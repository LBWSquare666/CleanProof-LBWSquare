#!/usr/bin/env python3
"""Real-robot trial (agent side, runs on the GPU workstation by default): the same TaskRunner, with the executor replaced by a skill server client and the camera replaced by the robot-pc camera service.

    agent (this process) -- HTTP --> skill server (robot-pc, the only RTDE writer)
                   `- HTTP --> camera service (robot-pc, RealSense)
    Decision: Nemotron 3 Super @ Nebius Token Factory; verification: Cosmos @ local vLLM 8001 (on-site perception on the GPU workstation)

This process creates no RTDE interface. Whether the robot may move and how fast is decided by the skill server's graded modes (shadow / dry_run / full) and
on-site arming; with --approve confirm, every decision that moves the robot needs a y typed in the terminal before it runs. Ctrl-C: asks the skill server for a software stop
and ends the loop (the physical E-stop remains the first resort).

  # prerequisite: an SSH tunnel maps robot-pc's 8100 / 8101 to the local machine (scripts/robot_pc/tunnel.sh)
  DEMO_MODE=1 python scripts/run_real.py --site ur3e_site --zones B2 B3 --approve confirm --record --operator <name>
  # self-test on the dev machine (the skill server uses the simulated backend, see tests/test_skill_server.py):
  python scripts/run_real.py --sim --skill-url http://127.0.0.1:8100 --camera-url http://127.0.0.1:8100 --policy rules --verifier cv
Result: runs/real/<run_id>/ (same format as simulation), and one row appended to eval/trials.csv (the on-site staff's visual verdict is a separate column, not filled in by the model).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.common.config import REPO_ROOT, runs_root  # noqa: E402

TRIAL_FIELDS = ["time_utc", "git", "run_id", "site", "robot", "robot_mode", "policy", "verifier", "approve", "targets",
                "forbidden", "status", "reason", "steps", "total_passes", "final_status", "robot_time_s", "wall_time_s",
                "decision_cost_usd", "human_verdict", "operator", "recording", "run_dir", "notes"]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def append_trial(row: dict[str, Any], path: Path) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TRIAL_FIELDS})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="ur3e_site")
    ap.add_argument("--skill-url", default="http://127.0.0.1:8100")
    ap.add_argument("--camera-url", default="http://127.0.0.1:8101")
    ap.add_argument("--zones", nargs="*", default=[], help="target zones; empty = all")
    ap.add_argument("--forbidden", nargs="*", default=[])
    ap.add_argument("--instruction", default="Clean the marked zones and prove they are clean.")
    ap.add_argument("--max-passes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--policy", default="nemotron", choices=("nemotron", "rules"))
    ap.add_argument("--verifier", default="vlm", choices=("vlm", "cv"))
    ap.add_argument("--approve", default="confirm", choices=("confirm", "auto"))
    ap.add_argument("--record", action="store_true", help="record the fixed camera for the whole trial (the camera service writes on robot-pc)")
    ap.add_argument("--operator", default="", help="on-site staff name (written to trials.csv)")
    ap.add_argument("--no-verdict", action="store_true", help="do not ask for the on-site visual verdict at the end")
    ap.add_argument("--max-drift-px", type=float, default=6.0, help="upper limit for the ArUco marker drift check before starting")
    ap.add_argument("--sim", action="store_true", help="the skill server is a simulated backend (self-test): reference image and camera come from it")
    ap.add_argument("--expect-backend", default="ur3e", help="without --sim the skill server must be this backend (prevents mistaking a local simulation service for the real robot)")
    ap.add_argument("--runs-dir", default=str(runs_root() / "real"))
    ap.add_argument("--trials-csv", default=str(REPO_ROOT / "eval" / "trials.csv"))
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if not args.sim:
        os.environ["CLEANPROOF_SITE"] = args.site
    from cleanproof.agent.loop import LoopOptions, TaskRunner
    from cleanproof.agent.runlog import RunLogger
    from cleanproof.common.schemas import TaskSpec
    from cleanproof.perception.residue_cv import ResidueDetector
    from cleanproof.perception.verifier import CVVerifier, VLMVerifier
    from cleanproof.agent.remote_job import RemoteSetupError, connect_robot
    from cleanproof.robot.server import auth_headers
    from cleanproof.sim.run import make_policy

    AUTH = auth_headers()
    try:
        link = connect_robot(args.skill_url, args.camera_url, None if args.sim else args.site, args.forbidden,
                             expect_backend=None if args.sim else args.expect_backend, max_drift_px=args.max_drift_px,
                             zones=args.zones)
    except RemoteSetupError as e:
        print(f"refused: {e}")
        return 2
    ex, health, grid, src, reference, forbidden, meta = link.ex, link.health, link.grid, link.src, link.reference, link.forbidden, link.meta
    print(f"skill server: {health['label']} mode={health['mode']} host={health.get('host', '-')} git={health['git']}")
    if "marker_drift" in meta:
        print("marker drift:", json.dumps(meta["marker_drift"], ensure_ascii=False))
    det = ResidueDetector(grid, reference_rgb=reference)

    client = None
    if args.policy == "nemotron" or args.verifier == "vlm":
        from cleanproof.common.llm_client import LLMClient

        client = LLMClient()
        if os.environ.get("DEMO_MODE") == "1":
            client.assert_demo_compliance()
    if os.environ.get("DEMO_MODE") == "1" and (args.policy != "nemotron" or args.verifier != "vlm"):
        print("refused: with DEMO_MODE=1 the decision must be Nemotron (Token Factory) and the verification must be Cosmos (local vLLM)")
        return 2
    ver = VLMVerifier(grid, det, client) if args.verifier == "vlm" else CVVerifier(grid, det)
    pol = make_policy(args.policy, client)
    # Do not delete forbidden zones here: TaskState removes forbidden zones and nonexistent cells; deleting first would turn "all named cells are forbidden" into an empty list = all zones
    task = TaskSpec(instruction=args.instruction, zones=list(args.zones), forbidden_zones=forbidden,
                    max_passes_per_zone=args.max_passes, max_steps=args.max_steps)
    from cleanproof.agent.state import TaskState

    if args.zones and not TaskState(task, grid).targets:
        print(f"refused: the named zones {args.zones} are all forbidden zones or not in the zone grid")
        ex.close()
        return 2

    tag = "sim" if args.sim else health["backend"]
    run_id = f"real-{tag}-{health['mode']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logger = RunLogger(Path(args.runs_dir), run_id=run_id)
    logger.image("reference", reference)
    try:
        import httpx

        meta["camera_service"] = httpx.get(f"{args.camera_url.rstrip('/')}/health", timeout=5.0, headers=AUTH).json()
    except Exception as e:  # noqa: BLE001
        meta["camera_service"] = {"error": str(e)[:200]}
    logger.write_json("reference_meta.json", {"source": src.name, **meta, "git": git_rev(), "args": vars(args)})

    def approve(decision, state) -> bool:
        d = decision.model_dump(mode="json")
        print(f"\n[step {state.step}] proposal: {d['action']} {d.get('zone_ids')} force={d.get('force_n')} "
              f"pattern={d.get('pattern')} passes={d.get('passes')} bbox={d.get('bbox_norm')}\n  reason: {d.get('reason')}")
        try:
            return input("  Execute? [y/N] ").strip().lower() == "y"
        except EOFError:
            return False

    runner = TaskRunner(task, ex, src, ver, pol, logger, LoopOptions(audit=False),
                        approve=approve if args.approve == "confirm" else None)

    def on_sigint(signum, frame):  # noqa: ARG001
        print("\nCtrl-C received: asking the skill server for a software stop, the loop will end after the current skill returns (press the physical E-stop if an immediate stop is needed)")
        ex.stop("operator Ctrl-C")
        runner.request_stop("operator Ctrl-C")

    signal.signal(signal.SIGINT, on_sigint)
    recording = None
    if args.record:
        try:
            import httpx

            recording = httpx.post(f"{args.camera_url.rstrip('/')}/record/start", json={"name": run_id}, timeout=5.0, headers=AUTH).json().get("recording")
            print("recording:", recording)
        except Exception as e:  # noqa: BLE001
            print("recording failed to start (trial continues):", e)
    crash = None
    try:
        summary = runner.run()
    except Exception as e:  # noqa: BLE001  a failed trial must still leave a row, it cannot vanish from the statistics
        crash = f"{type(e).__name__}: {e}"[:300]
        ex.stop(f"agent crashed: {crash}")
        summary = {"run_id": run_id, "status": "crashed", "reason": crash, "targets": runner.state.targets,
                   "steps": runner.state.step, "passes": runner.state.passes,
                   "final_status": {z: (runner.state.status(z).value if runner.state.status(z) else None) for z in runner.state.targets},
                   "robot_time_s": float("nan"), "wall_time_s": float("nan"), "decision_cost_usd": round(runner.llm_cost, 6)}
        logger.event("crashed", error=crash)
    finally:
        if args.record:
            try:
                import httpx

                rec = httpx.post(f"{args.camera_url.rstrip('/')}/record/stop", timeout=10.0, headers=AUTH).json()
                logger.write_json("recording.json", rec)
            except Exception as e:  # noqa: BLE001
                print("recording failed to stop:", e)
    print(f"\nresult: {summary['status']} {summary.get('reason', '')}  final={summary['final_status']}  passes={summary['passes']}")

    verdict = ""
    if not args.no_verdict and not args.sim:
        try:
            verdict = input("on-site visual check: are all target zones clean? [y=clean / n=residue left / s=skip] ").strip().lower()[:1]
        except EOFError:
            verdict = ""
        verdict = {"y": "clean", "n": "residue"}.get(verdict, "")
    elif args.sim:
        try:
            truth = __import__("httpx").get(f"{args.skill_url.rstrip('/')}/sim/truth", timeout=5.0, headers=AUTH).json()
            verdict = "sim_truth_clean" if all(truth[z] for z in summary["targets"]) else "sim_truth_residue"
        except Exception:  # noqa: BLE001
            verdict = ""
    summary["human_verdict"] = verdict
    logger.write_json("summary.json", summary)
    row = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_rev(), "run_id": run_id,
           "site": "" if args.sim else args.site, "robot": health["label"], "robot_mode": health["mode"],
           "policy": args.policy, "verifier": args.verifier, "approve": args.approve, "targets": " ".join(summary["targets"]),
           "forbidden": " ".join(forbidden), "status": summary["status"], "reason": summary.get("reason", "")[:200],
           "steps": summary["steps"], "total_passes": int(sum(summary["passes"].values())),
           "final_status": json.dumps(summary["final_status"]), "robot_time_s": summary["robot_time_s"],
           "wall_time_s": summary["wall_time_s"], "decision_cost_usd": summary["decision_cost_usd"],
           "human_verdict": verdict, "operator": args.operator, "recording": recording or "", "run_dir": str(logger.dir),
           "notes": args.notes}
    if args.trials_csv:
        append_trial(row, Path(args.trials_csv))
    print(f"run directory {logger.dir}")
    ex.close()
    if crash:
        print(f"agent crashed: {crash} (skill server asked for a software stop; confirm the robot state on site)")
    return 0 if summary["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main())
