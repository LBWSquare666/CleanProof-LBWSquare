#!/usr/bin/env python3
"""Run simulation scenarios and judge them against ground truth.

Examples:
  python scripts/run_sim.py --scenario all --policy rules --verifier cv
  python scripts/run_sim.py --scenario S2_dried --policy nemotron --verifier vlm --source render   # server: MUJOCO_GL=egl
  python scripts/run_sim.py --scenario all --robot rtde_fake              # skill → rtde_backend → fake RTDE (MuJoCo)
  python scripts/run_sim.py --scenario S1_light --robot rtde_fake --mode shadow   # only log commands, do not move
Result: runs/<run_id>/ (events, images, summary, scenario.json), and one row appended to eval/sim_results.csv (including robot and mode columns).
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.common.config import REPO_ROOT, runs_root  # noqa: E402
from cleanproof.robot.commissioning import MODES, CommissioningGate  # noqa: E402
from cleanproof.sim.run import POLICIES, VERIFIERS, run_sim_scenario  # noqa: E402
from cleanproof.sim.scenarios import ROBOT_BACKENDS, scenario_names  # noqa: E402

FIELDS = ["time_utc", "git", "scenario", "policy", "verifier", "source", "status", "passed", "targets_truly_clean",
          "false_clean", "total_passes", "robot_time_s", "decision_cost_usd", "run_dir", "robot", "mode", "platform"]
LEGACY_DEFAULTS = {"robot": "mujoco", "mode": "full", "platform": "ur3e"}   # old rows (missing these columns) are all mujoco backend, full mode, UR3e


def append_rows(out: Path, rows: list[dict]) -> None:
    """Append a result; if the old file lacks columns, first fill the old rows with defaults and then rewrite the header (no old data is lost)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    old_rows: list[dict] = []
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and set(reader.fieldnames) != set(FIELDS):
                old_rows = [{k: r.get(k, LEGACY_DEFAULTS.get(k, "")) for k in FIELDS} for r in reader]
        if old_rows:
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(old_rows)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="all", help=f"scenario name or all: {', '.join(scenario_names())}")
    ap.add_argument("--policy", default="rules", choices=POLICIES)
    ap.add_argument("--verifier", default="cv", choices=VERIFIERS)
    ap.add_argument("--source", default="topdown", choices=("topdown", "render"))
    ap.add_argument("--robot", default="mujoco", choices=ROBOT_BACKENDS,
                    help="mujoco: skills drive the simulation directly; rtde_fake: skill → rtde_backend → fake RTDE (MuJoCo)")
    ap.add_argument("--mode", default="full", choices=MODES, help="graded mode of the rtde backend (shadow only logs commands and does not move)")
    ap.add_argument("--enforce-gate", action="store_true", help="also enforce the commissioning gate for rtde_fake (rehearse the procedure)")
    ap.add_argument("--gate-path", default=None, help="gate evidence file path (default: configs/robot.yaml commissioning.gate_path)")
    ap.add_argument("--audit", action="store_true", help="write an audit with Nemotron 3 Ultra at the end (needs Token Factory)")
    ap.add_argument("--platform", default=None, help="robot platform (configs/platforms.yaml: ur3e / ur5e / panda / rizon4 / spot_arm); default ur3e")
    ap.add_argument("--runs-dir", default=str(runs_root()))
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    client = None
    if args.policy == "nemotron" or args.verifier == "vlm" or args.audit:
        from cleanproof.common.llm_client import LLMClient

        client = LLMClient()

    names = scenario_names() if args.scenario == "all" else [args.scenario]
    gate = CommissioningGate(args.gate_path) if args.gate_path else None
    if args.robot == "rtde_fake":
        print(f"robot backend rtde_fake = fake RTDE (MuJoCo), mode={args.mode}; not the real robot, not URSim")
    rows = []
    for name in names:
        res = run_sim_scenario(name, args.policy, args.verifier, args.source, Path(args.runs_dir), client, args.audit,
                               robot_backend=args.robot, mode=args.mode, enforce_gate=args.enforce_gate, gate=gate, platform=args.platform)
        j, s = res["judge"], res["summary"]
        rows.append({
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_rev(),
            "scenario": name, "policy": args.policy, "verifier": args.verifier, "source": args.source,
            "status": j["status"], "passed": j["passed"], "targets_truly_clean": j["targets_truly_clean"],
            "false_clean": len(j["false_clean_zones"]), "total_passes": j["total_passes"],
            "robot_time_s": j["robot_time_s"], "decision_cost_usd": s["decision_cost_usd"], "run_dir": res["run_dir"],
            "robot": args.robot, "mode": args.mode, "platform": j["robot"].get("platform", "ur3e"),
        })
        print(f"{name:14s} {j['status']:12s} passed={j['passed']!s:5s} clean={j['targets_truly_clean']!s:5s} "
              f"false_clean={len(j['false_clean_zones'])} passes={j['total_passes']} t={j['robot_time_s']:.0f}s "
              f"robot={args.robot}/{args.mode} checks={j['checks']}"
              + (f" rtde_calls={j['rtde_calls']}" if "rtde_calls" in j else ""))
    if not args.no_csv:
        append_rows(REPO_ROOT / "eval" / "sim_results.csv", rows)
    # shadow / dry_run never clean the tabletop anyway, so passed is not used as the basis for the exit code
    return 0 if all(r["passed"] for r in rows) or args.policy.startswith("fixed") or args.mode != "full" else 1


if __name__ == "__main__":
    raise SystemExit(main())
