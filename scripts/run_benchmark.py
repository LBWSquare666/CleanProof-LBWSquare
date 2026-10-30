#!/usr/bin/env python3
"""Randomized simulation benchmark (configs/benchmark.yaml): each variant runs three policies in turn, and results are appended row by row to eval/sim_benchmark.csv.

Policies (same config as the existing rows of eval/sim_results.csv, source=render):
  cleanproof = --policy nemotron --verifier vlm (Nemotron @ Token Factory decides; Cosmos @ local vLLM 8001 verifies)
  fixed1 / fixed2 = --policy fixed1|fixed2 --verifier cv (blind wipe, 1 / 2 passes, no check)

Examples:
  # pilot: variant 0 of each family x 3 policies = 18 runs
  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python scripts/run_benchmark.py --variants 0 --workers 2
  # new machine / new card: CLEANPROOF_GPU=0; blind wipes only (no GPU needed): MUJOCO_GL=osmesa CUDA_VISIBLE_DEVICES= ... --policies fixed1,fixed2
  # full run (resumable: (variant, policy) pairs that are done and have a matching scenario hash are skipped automatically)
  python scripts/run_benchmark.py --workers 2
  # show the plan only, do not run
  python scripts/run_benchmark.py --dry-run

Each run executes in its own subprocess (MuJoCo/EGL contexts are isolated, a crash does not affect other runs); a crash / timeout also records a row
(status=error|timeout), which does not count as done when resuming and is rerun, and the summary script lists these rows separately.
Spend guardrail: once the benchmark's cumulative decision spend (the larger of the sum of decision_cost_usd in the csv and the increment of the llm_client ledger over the baseline)
exceeds cost_guard.usd_limit in configs/benchmark.yaml, no new run is started and the exit code is 3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.common.config import REPO_ROOT, load_yaml  # noqa: E402
from cleanproof.common.jsonl import locked, read_jsonl  # noqa: E402
from cleanproof.sim.scenario_random import (  # noqa: E402
    Variant,
    config_hash,
    find_variant,
    generate_variant,
    load_benchmark,
    spec_hash,
    variant_list,
)

DEFAULT_CSV = REPO_ROOT / "eval" / "sim_benchmark.csv"
DEFAULT_WORK = REPO_ROOT / "runs" / "benchmark_v1"
PROMPT_DIR = REPO_ROOT / "cleanproof" / "agent" / "prompts"
PYTHON = sys.executable
ERROR_STATUSES = ("error", "timeout")

FIELDS = ["time_utc", "git", "bench", "config_hash", "variant_id", "family", "variant_index", "group", "seed",
          "scenario_hash", "policy", "sim_policy", "verifier", "source", "decision_model", "verifier_model",
          "prompt_version", "status", "passed", "targets_truly_clean", "false_done", "false_clean", "total_passes",
          "robot_time_s", "decision_cost_usd", "wall_s", "run_dir", "notes"]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def prompt_version() -> str:
    """Hash of the contents of the decision and verifier prompt files (the repo has no explicit version number)."""
    out = []
    for tag, name in (("dec", "decision_system.md"), ("ver", "verifier_zones.md")):
        out.append(f"{tag}-{hashlib.sha1((PROMPT_DIR / name).read_bytes()).hexdigest()[:7]}")
    return "/".join(out)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_row(path: Path, row: dict[str, Any]) -> None:
    with locked(path):
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})


# ---------------------------------------------------------------------------------------------
# subprocess: run one (variant, policy)
# ---------------------------------------------------------------------------------------------

def _scan_notes(run_dir: Path) -> str:
    """Find infrastructure problems in events.jsonl (decision model unreachable / fallback, verifier call failed / fell back to CV) and record them honestly in notes."""
    notes: list[str] = []
    dec_fallback = 0
    dec_errors: list[str] = []
    ver_errors = 0
    cv_fallback = 0
    for e in read_jsonl(run_dir / "events.jsonl"):
        if e.get("kind") == "decision":
            for m in (e.get("meta") or {}).get("policy", []):
                if m.get("fallback"):
                    dec_fallback += 1
                if m.get("error"):
                    dec_errors.append(str(m["error"])[:80])
        elif e.get("kind") == "inspection":
            for c in ((e.get("llm") or {}).get("calls") or []):
                if c.get("error"):
                    ver_errors += 1
            cv_fallback += sum(1 for p in (e.get("provenance") or {}).values() if p == "cv_fallback")
    if dec_fallback:
        notes.append(f"decision_fallback={dec_fallback}")
    if dec_errors:
        notes.append("decision_error=" + " ; ".join(dec_errors[:2]))
    if ver_errors:
        notes.append(f"verifier_call_errors={ver_errors}")
    if cv_fallback:
        notes.append(f"verifier_cv_fallback_zones={cv_fallback}")
    return "; ".join(notes)


def run_one(cfg: dict[str, Any], v: Variant, policy_name: str, runs_dir: Path, result_json: Path) -> int:
    from cleanproof.sim.run import run_sim_scenario

    pol = cfg["policies"][policy_name]
    spec = generate_variant(v, cfg)
    client = None
    if pol["policy"] == "nemotron" or pol["verifier"] == "vlm":
        from cleanproof.common.llm_client import LLMClient

        client = LLMClient()
    t0 = time.time()
    # experiment config (configs/exp/*.yaml) can give source / ablation (guardrail, final check, fixed force) per policy; the default is the same as benchmark_v1
    res = run_sim_scenario(v.variant_id, pol["policy"], pol["verifier"], str(pol.get("source", "render")), runs_dir, client, audit=False,
                           spec=spec, ablation=pol.get("ablation"))
    wall = time.time() - t0
    j, s = res["judge"], res["summary"]
    run_dir = Path(res["run_dir"])
    # also write the generated scenario spec to disk, for easy review
    (run_dir / "scenario_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    out = {
        "variant_id": v.variant_id, "policy": policy_name, "scenario_hash": spec_hash(spec),
        "status": j["status"], "passed": bool(j["passed"]), "targets_truly_clean": bool(j["targets_truly_clean"]),
        "false_clean": len(j["false_clean_zones"]), "total_passes": int(j["total_passes"]),
        "robot_time_s": float(j["robot_time_s"]), "decision_cost_usd": float(s.get("decision_cost_usd", 0.0)),
        "wall_s": round(wall, 1), "run_dir": str(run_dir), "notes": _scan_notes(run_dir), "checks": j["checks"],
    }
    result_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------
# parent process: plan, resume, parallelism, spend guardrail
# ---------------------------------------------------------------------------------------------

class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def parse_indices(text: Optional[str], n: int) -> Optional[set[int]]:
    if not text:
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise SystemExit(f"variant index out of range {bad} (0..{n - 1})")
    return out


def completed_keys(rows: list[dict[str, str]], bench: str, hashes: dict[str, str]) -> set[tuple[str, str]]:
    """Done = same benchmark, same (variant, policy), scenario hash matches the current generation, and not an error/timeout row."""
    done = set()
    for r in rows:
        if r.get("bench") != bench or r.get("status") in ERROR_STATUSES:
            continue
        if hashes.get(r["variant_id"]) != r.get("scenario_hash"):
            continue
        done.add((r["variant_id"], r["policy"]))
    return done


class CostGuard:
    """Cumulative decision spend of this benchmark: the larger of the csv total (exact for this benchmark) and the ledger increment (upper bound, includes other Token Factory calls in the same period)."""

    def __init__(self, csv_path: Path, bench: str, baseline_path: Path, limit_usd: float, log: Log):
        from cleanproof.common.llm_client import SpendLedger

        mcfg = load_yaml("models.yaml")
        b = mcfg.get("budget", {})
        self.ledger = SpendLedger(b.get("ledger_path", "~/.local/state/cleanproof/spend.json"), b.get("usd_limit", 40.0))
        self.csv_path, self.bench, self.limit, self.log = csv_path, bench, float(limit_usd), log
        if baseline_path.exists():
            self.baseline = float(json.loads(baseline_path.read_text())["total_usd"])
        else:
            self.baseline = self.ledger.total()
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps({"total_usd": self.baseline, "time_utc": now_utc(),
                                                 "note": "total ledger amount of llm_client before the benchmark started; later increments are treated as an upper bound of this benchmark's spend"}, indent=2))
        self.last: tuple[float, float] = (0.0, 0.0)

    def current(self) -> tuple[float, float]:
        csv_sum = sum(float(r.get("decision_cost_usd") or 0.0) for r in read_rows(self.csv_path) if r.get("bench") == self.bench)
        delta = self.ledger.total() - self.baseline
        self.last = (csv_sum, delta)
        return self.last

    def exceeded(self) -> bool:
        csv_sum, delta = self.current()
        if max(csv_sum, delta) > self.limit:
            self.log(f"COST CAP: decision spend of this benchmark csv=${csv_sum:.4f} ledger increment=${delta:.4f} > limit ${self.limit:.2f}, no new runs will be started")
            return True
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK), help="logs, ledger baseline, run directories (<work>/runs)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="maximum number of runs to start this time")
    ap.add_argument("--families", default=None, help="comma separated, e.g. S1_light,S5_obstacle")
    ap.add_argument("--variants", default=None, help="variant indices, e.g. 0 or 0-4 or 0,3,7 (default: all)")
    ap.add_argument("--policies", default=None, help="comma separated, default cleanproof,fixed1,fixed2")
    ap.add_argument("--timeout", type=float, default=1200.0, help="wall-clock limit of a single run (seconds)")
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    ap.add_argument("--one", nargs=2, metavar=("VARIANT_ID", "POLICY"), help="(internal) run one run in this process")
    ap.add_argument("--result-json", default=None, help="(internal) result file of --one")
    args = ap.parse_args()

    cfg = load_benchmark(Path(args.config))
    work = Path(args.work_dir)
    runs_dir = work / "runs"

    if args.one:
        v = find_variant(cfg, args.one[0])
        if args.one[1] not in cfg["policies"]:
            raise SystemExit(f"unknown policy {args.one[1]!r}")
        return run_one(cfg, v, args.one[1], runs_dir, Path(args.result_json or (work / "last_result.json")))

    # ---- environment guardrail: use only the one specified card (CLEANPROOF_GPU, default 2 on the dev server), EGL offscreen rendering; set CLEANPROOF_GPU when changing machine ----
    # For cluster jobs (the cluster PBS) set CLEANPROOF_GPU=sched: the card is assigned by the scheduler (CUDA_VISIBLE_DEVICES may be a UUID), and EGL uses the 0th card visible in the job
    gpu = os.environ.get("CLEANPROOF_GPU", "2")
    os.environ.setdefault("MUJOCO_GL", "egl")
    if gpu == "sched":
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", gpu)
    if gpu != "sched" and os.environ["MUJOCO_GL"] == "egl" and (os.environ["CUDA_VISIBLE_DEVICES"] != gpu or os.environ["MUJOCO_EGL_DEVICE_ID"] != gpu):
        raise SystemExit(f"only GPU {gpu} may be used (CUDA_VISIBLE_DEVICES={gpu}, MUJOCO_EGL_DEVICE_ID={gpu}; set CLEANPROOF_GPU to change the card)")

    log = Log(work / f"bench_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    csv_path = Path(args.csv)
    bench = cfg["name"]
    policies = (args.policies.split(",") if args.policies else list(cfg["policies"]))
    for p in policies:
        if p not in cfg["policies"]:
            raise SystemExit(f"unknown policy {p!r}, choose from {list(cfg['policies'])}")
    fams = set(args.families.split(",")) if args.families else None
    idx = parse_indices(args.variants, int(cfg["n_variants_per_family"]))

    variants = [v for v in variant_list(cfg) if (fams is None or v.family in fams) and (idx is None or v.index in idx)]
    log(f"generating {len(variants)} variants and computing scenario hashes ...")
    hashes = {v.variant_id: spec_hash(generate_variant(v, cfg)) for v in variant_list(cfg)}
    done = completed_keys(read_rows(csv_path), bench, hashes)
    # order: rotate through the families by variant index (complete group 1 first), the three policies of the same variant are adjacent
    plan = [(v, p) for v in sorted(variants, key=lambda v: (v.index, v.family)) for p in policies
            if (v.variant_id, p) not in done]
    if args.limit is not None:
        plan = plan[: args.limit]
    log(f"benchmark {bench} config_hash={config_hash(Path(args.config))} git={git_rev()} prompt={prompt_version()}; "
        f"done {len(done)}, planned this time {len(plan)} runs, workers={args.workers}")
    if args.dry_run:
        for v, p in plan:
            log(f"  {v.variant_id:18s} group={v.group} seed={v.seed} {p}")
        return 0
    if not plan:
        log("no runs to do")
        return 0

    guard = CostGuard(csv_path, bench, work / "ledger_baseline.json", float(cfg["cost_guard"]["usd_limit"]), log)
    mcfg = load_yaml("models.yaml")
    dec_model = mcfg["roles"]["decision"]["model"]
    ver_model = mcfg["roles"]["verifier"]["model"]
    pver = prompt_version()
    rev = git_rev()
    chash = config_hash(Path(args.config))
    (work / "logs").mkdir(parents=True, exist_ok=True)
    (work / "results").mkdir(parents=True, exist_ok=True)

    running: list[dict[str, Any]] = []
    queue = list(plan)
    n_done = n_fail = 0
    stop_launching = False
    t_start = time.time()

    def base_row(v: Variant, p: str) -> dict[str, Any]:
        pol = cfg["policies"][p]
        return {
            "time_utc": now_utc(), "git": rev, "bench": bench, "config_hash": chash,
            "variant_id": v.variant_id, "family": v.family, "variant_index": v.index, "group": v.group, "seed": v.seed,
            "scenario_hash": hashes[v.variant_id], "policy": p, "sim_policy": pol["policy"], "verifier": pol["verifier"],
            "source": "render", "decision_model": dec_model if pol["policy"] == "nemotron" else "",
            "verifier_model": ver_model if pol["verifier"] == "vlm" else "", "prompt_version": pver,
        }

    def launch(v: Variant, p: str) -> None:
        tag = f"{v.variant_id}-{p}"
        result = work / "results" / f"{tag}.json"
        if result.exists():
            result.unlink()
        logf = open(work / "logs" / f"{tag}.log", "a", encoding="utf-8")
        cmd = [PYTHON, str(Path(__file__).resolve()), "--config", args.config, "--work-dir", str(work),
               "--one", v.variant_id, p, "--result-json", str(result)]
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT), env=os.environ.copy())
        running.append({"v": v, "p": p, "proc": proc, "t0": time.time(), "result": result, "logf": logf, "tag": tag})
        log(f"started {tag} (pid {proc.pid})")

    def finish(job: dict[str, Any], timed_out: bool) -> None:
        nonlocal n_done, n_fail
        job["logf"].close()
        v, p = job["v"], job["p"]
        row = base_row(v, p)
        wall = round(time.time() - job["t0"], 1)
        if timed_out:
            row.update(status="timeout", wall_s=wall, notes=f"killed after {args.timeout:.0f}s; log={job['logf'].name}")
            n_fail += 1
        elif job["proc"].returncode != 0 or not job["result"].exists():
            tail = ""
            try:
                tail = Path(job["logf"].name).read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1][:160]
            except Exception:
                pass
            row.update(status="error", wall_s=wall, notes=f"exit={job['proc'].returncode}; {tail}")
            n_fail += 1
        else:
            r = json.loads(job["result"].read_text(encoding="utf-8"))
            row.update({k: r[k] for k in ("status", "passed", "targets_truly_clean", "false_clean", "total_passes",
                                           "robot_time_s", "decision_cost_usd", "wall_s", "run_dir", "notes")})
            row["false_done"] = bool(r["status"] == "done" and not r["targets_truly_clean"])
            row["scenario_hash"] = r["scenario_hash"]
            n_done += 1
        append_row(csv_path, row)
        log(f"finished {job['tag']}: status={row.get('status')} passed={row.get('passed', '')} "
            f"clean={row.get('targets_truly_clean', '')} false_done={row.get('false_done', '')} "
            f"passes={row.get('total_passes', '')} robot_t={row.get('robot_time_s', '')} "
            f"cost=${float(row.get('decision_cost_usd') or 0):.4f} wall={row.get('wall_s')}s "
            f"{('notes=' + row['notes']) if row.get('notes') else ''}")

    try:
        while queue or running:
            # reap
            for job in list(running):
                rc = job["proc"].poll()
                if rc is not None:
                    running.remove(job)
                    finish(job, timed_out=False)
                elif time.time() - job["t0"] > args.timeout:
                    job["proc"].kill()
                    job["proc"].wait()
                    running.remove(job)
                    finish(job, timed_out=True)
            # launch
            while queue and len(running) < args.workers and not stop_launching:
                v, p = queue[0]
                if cfg["policies"][p]["policy"] == "nemotron" and guard.exceeded():
                    stop_launching = True
                    break
                queue.pop(0)
                launch(v, p)
            if stop_launching and not running:
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("interrupt received, terminating subprocesses ...")
        for job in running:
            job["proc"].kill()
        raise
    csv_sum, delta = guard.current()
    log(f"finished: {n_done} recorded successfully, error/timeout {n_fail}, not started {len(queue)}, elapsed {(time.time() - t_start) / 60:.1f} min; "
        f"decision spend of this benchmark csv=${csv_sum:.4f} ledger increment=${delta:.4f}")
    return 3 if stop_launching else (0 if n_fail == 0 else 2)


if __name__ == "__main__":
    raise SystemExit(main())
