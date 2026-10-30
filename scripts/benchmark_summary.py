#!/usr/bin/env python3
"""Summarize randomized benchmark results: eval/sim_benchmark.csv → eval/benchmark_summary.json + eval/benchmark_summary.md.

For each policy:
- proportions of right_outcome (correct outcome = judge passed) and false_done (status=done but the target zones were not actually clean), with Wilson 95 % intervals;
- mean and 95 % interval (t interval, per job; per-group means are also given for cross-checking) of wipes per job and robot time;
- proportions per group (5 groups × 12 jobs); number of correct outcomes per family.
Only rows whose scenario hash matches what the current configs/benchmark.yaml generates and that are not error/timeout are counted; if one (variant, policy) has several rows only the first is used
(pre-registered seeds may not be rerun to cherry-pick results); duplicate and errored rows are listed separately.

Example: python scripts/benchmark_summary.py [--csv eval/sim_benchmark.csv] [--out eval/benchmark_summary.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.common.config import REPO_ROOT  # noqa: E402
from cleanproof.sim.scenario_random import (  # noqa: E402
    FAMILIES,
    config_hash,
    generate_variant,
    load_benchmark,
    spec_hash,
    variant_list,
)

Z95 = 1.959963984540054
# two-sided 95 % t quantiles (degrees of freedom 1..30, normal approximation with a small correction beyond)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df in _T95:
        return _T95[df]
    # Cornish–Fisher first-order correction, error < 0.002 for df > 30
    return Z95 + (Z95 ** 3 + Z95) / (4 * df)


def wilson(k: int, n: int, z: float = Z95) -> dict[str, Any]:
    """Wilson score interval (no continuity correction). Returns an empty interval when n = 0."""
    if n <= 0:
        return {"k": 0, "n": 0, "rate": None, "ci95": [None, None], "half_width": None}
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return {"k": int(k), "n": int(n), "rate": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "half_width": round((hi - lo) / 2, 4)}


def mean_ci(values: list[float]) -> dict[str, Any]:
    """Mean and t interval (per job)."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None], "sum": 0.0}
    m = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": round(m, 3), "sd": None, "ci95": [None, None], "sum": round(sum(values), 3)}
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = t95(n - 1) * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 3), "sd": round(sd, 3), "ci95": [round(m - h, 3), round(m + h, 3)],
            "sum": round(sum(values), 3), "ci95_method": "t over runs"}


def _b(x: Any) -> bool:
    return str(x).strip().lower() in ("true", "1", "yes")


def select_rows(rows: list[dict[str, str]], bench: str, hashes: dict[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Valid rows, errored rows, duplicate / stale rows."""
    valid, errors, dropped = [], [], []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        if r.get("bench") != bench:
            dropped.append({**r, "_why": "other bench"})
            continue
        if r.get("status") in ("error", "timeout"):
            errors.append(r)
            continue
        if hashes.get(r.get("variant_id", "")) != r.get("scenario_hash"):
            dropped.append({**r, "_why": "scenario_hash mismatch (stale generator/config)"})
            continue
        key = (r["variant_id"], r["policy"])
        if key in seen:
            dropped.append({**r, "_why": "duplicate (first row kept)"})
            continue
        seen.add(key)
        valid.append(r)
    return valid, errors, dropped


def summarize(rows: list[dict[str, str]], cfg: dict[str, Any], hashes: dict[str, str],
              extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    bench = cfg["name"]
    valid, errors, dropped = select_rows(rows, bench, hashes)
    groups = int(cfg["groups"])
    n_expected = int(cfg["n_variants_per_family"]) * len(FAMILIES)
    policies: dict[str, Any] = {}
    for p in cfg["policies"]:
        rs = [r for r in valid if r["policy"] == p]
        passed = [_b(r["passed"]) for r in rs]
        fdone = [_b(r.get("false_done")) for r in rs]
        wipes = [float(r["total_passes"]) for r in rs]
        rtime = [float(r["robot_time_s"]) for r in rs]
        per_group = []
        for g in range(1, groups + 1):
            gr = [r for r in rs if int(r["group"]) == g]
            per_group.append({
                "group": g, "n": len(gr),
                "right_outcome": wilson(sum(_b(r["passed"]) for r in gr), len(gr)),
                "false_done": wilson(sum(_b(r.get("false_done")) for r in gr), len(gr)),
                "wipes_mean": round(sum(float(r["total_passes"]) for r in gr) / len(gr), 3) if gr else None,
                "robot_time_mean_s": round(sum(float(r["robot_time_s"]) for r in gr) / len(gr), 2) if gr else None,
            })
        per_family = {}
        for fam in FAMILIES:
            fr = [r for r in rs if r["family"] == fam]
            per_family[fam] = {"n": len(fr), "right_outcome_k": sum(_b(r["passed"]) for r in fr),
                               "false_done_k": sum(_b(r.get("false_done")) for r in fr),
                               "statuses": sorted({r["status"] for r in fr}),
                               "wipes_sum": int(sum(float(r["total_passes"]) for r in fr)),
                               "robot_time_sum_s": round(sum(float(r["robot_time_s"]) for r in fr), 1)}
        policies[p] = {
            "config": dict(cfg["policies"][p]), "n": len(rs), "n_expected": n_expected,
            "complete": len(rs) == n_expected,
            "right_outcome": wilson(sum(passed), len(rs)),
            "false_done": wilson(sum(fdone), len(rs)),
            "wipes": mean_ci(wipes), "robot_time_s": mean_ci(rtime),
            "decision_cost_usd_sum": round(sum(float(r.get("decision_cost_usd") or 0) for r in rs), 4),
            "wall_s_mean": round(sum(float(r.get("wall_s") or 0) for r in rs) / len(rs), 1) if rs else None,
            "per_group": per_group, "per_family": per_family,
            "models": sorted({r.get("decision_model", "") for r in rs} | {r.get("verifier_model", "") for r in rs}) ,
            "prompt_versions": sorted({r.get("prompt_version", "") for r in rs}),
            "git": sorted({r.get("git", "") for r in rs}),
            "notes_rows": [{"variant_id": r["variant_id"], "notes": r["notes"]} for r in rs if r.get("notes")],
        }
    out = {
        "benchmark": bench, "n_per_policy": n_expected, "groups": groups,
        "variants_per_family": int(cfg["n_variants_per_family"]), "families": list(FAMILIES),
        "ci_method": {"rates": "Wilson score interval, 95 %", "means": "t interval over runs, 95 %"},
        "policies": policies,
        "errors": [{k: r.get(k, "") for k in ("variant_id", "policy", "status", "notes", "time_utc")} for r in errors],
        "dropped_rows": [{k: r.get(k, "") for k in ("variant_id", "policy", "status", "scenario_hash", "time_utc", "_why")} for r in dropped],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.update(extra or {})
    return out


def _pct(w: dict[str, Any]) -> str:
    if w["rate"] is None:
        return "–"
    return f"{100 * w['rate']:.0f} % ({w['k']}/{w['n']}, 95 % CI {100 * w['ci95'][0]:.0f}–{100 * w['ci95'][1]:.0f} %)"


def _mean(m: dict[str, Any], unit: str = "", nd: int = 1) -> str:
    if m["mean"] is None:
        return "–"
    ci = f" (95 % CI {m['ci95'][0]:.{nd}f}–{m['ci95'][1]:.{nd}f})" if m["ci95"][0] is not None else ""
    return f"{m['mean']:.{nd}f}{unit}{ci}"


def to_markdown(s: dict[str, Any]) -> str:
    L = [f"# Randomized simulation benchmark {s['benchmark']} summary", "",
         f"Generated {s['generated_utc']}; git {s.get('git', '?')}; configs/benchmark.yaml hash {s.get('config_hash', '?')}.",
         f"Design: {len(s['families'])} families × {s['variants_per_family']} random variants = {s['n_per_policy']} jobs per policy, in {s['groups']} groups. "
         "Proportions use Wilson 95 % intervals; means use per-job t intervals. **All results are MuJoCo simulation and do not represent real-robot performance.**", ""]
    names = {"cleanproof": "CleanProof (Nemotron + Cosmos)", "fixed1": "Blind wipe ×1", "fixed2": "Blind wipe ×2"}
    L += ["| Metric | " + " | ".join(names.get(p, p) for p in s["policies"]) + " |",
          "|---|" + "---|" * len(s["policies"])]
    P = s["policies"]
    L.append("| Jobs completed n | " + " | ".join(f"{P[p]['n']} / {P[p]['n_expected']}" for p in P) + " |")
    L.append("| Correct outcome (judge passed) | " + " | ".join(_pct(P[p]["right_outcome"]) for p in P) + " |")
    L.append("| Reports done but still dirty (false done) | " + " | ".join(_pct(P[p]["false_done"]) for p in P) + " |")
    L.append("| Wipes per job | " + " | ".join(_mean(P[p]["wipes"], "", 2) for p in P) + " |")
    L.append("| Robot time per job | " + " | ".join(_mean(P[p]["robot_time_s"], " s", 0) for p in P) + " |")
    L.append("| Total wipes / total robot time | " + " | ".join(
        f"{P[p]['wipes']['sum']:.0f} / {P[p]['robot_time_s']['sum']:.0f} s" for p in P) + " |")
    L.append("| Total decision cost (USD) | " + " | ".join(f"{P[p]['decision_cost_usd_sum']:.4f}" for p in P) + " |")
    L.append("| Mean wall clock / job | " + " | ".join(f"{P[p]['wall_s_mean']} s" if P[p]['wall_s_mean'] is not None else "–" for p in P) + " |")
    L += ["", "## Correct outcomes / false done per family", "",
          "| Family | " + " | ".join(names.get(p, p) for p in P) + " |", "|---|" + "---|" * len(P)]
    for fam in s["families"]:
        L.append(f"| {fam} | " + " | ".join(
            f"{P[p]['per_family'][fam]['right_outcome_k']}/{P[p]['per_family'][fam]['n']} · fd {P[p]['per_family'][fam]['false_done_k']}"
            for p in P) + " |")
    L += ["", "## Per-group proportions (for cross-checking)", ""]
    for p in P:
        L.append(f"- {names.get(p, p)}: " + "; ".join(
            f"group {g['group']} n={g['n']} correct {g['right_outcome']['k']}/{g['n']} fd {g['false_done']['k']} wipes {g['wipes_mean']} t {g['robot_time_mean_s']}"
            for g in P[p]["per_group"]))
    if any(P[p]["notes_rows"] for p in P):
        L += ["", "## Rows with notes (infrastructure issues etc.; results still counted)", ""]
        for p in P:
            for r in P[p]["notes_rows"]:
                L.append(f"- {p} {r['variant_id']}: {r['notes']}")
    if s["errors"]:
        L += ["", "## error / timeout rows (not counted in the proportions; a resumed run retries them)", ""]
        for r in s["errors"]:
            L.append(f"- {r['variant_id']} {r['policy']} {r['status']} {r['notes']}")
    if s["dropped_rows"]:
        L += ["", "## Excluded rows (scenario hash mismatch / duplicate / other benchmark)", ""]
        for r in s["dropped_rows"]:
            L.append(f"- {r['variant_id']} {r['policy']} {r['status']} {r['_why']}")
    L += ["", f"Data source: eval/sim_benchmark.csv; policy models {sorted({m for p in P for m in P[p]['models'] if m})}; "
          f"prompt versions {sorted({v for p in P for v in P[p]['prompt_versions'] if v})}."]
    return "\n".join(L) + "\n"


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    ap.add_argument("--csv", default=str(REPO_ROOT / "eval" / "sim_benchmark.csv"))
    ap.add_argument("--out", default=str(REPO_ROOT / "eval" / "benchmark_summary.json"))
    ap.add_argument("--md", default=None, help="defaults to the same name as --out with .md")
    args = ap.parse_args()
    cfg = load_benchmark(Path(args.config))
    hashes = {v.variant_id: spec_hash(generate_variant(v, cfg)) for v in variant_list(cfg)}
    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s = summarize(rows, cfg, hashes, extra={"git": git_rev(), "config_hash": config_hash(Path(args.config)),
                                            "csv": str(Path(args.csv).relative_to(REPO_ROOT)) if Path(args.csv).is_relative_to(REPO_ROOT) else args.csv})
    out = Path(args.out)
    out.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    md = Path(args.md) if args.md else out.with_suffix(".md")
    md.write_text(to_markdown(s), encoding="utf-8")
    print(to_markdown(s))
    print(f"wrote {out} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
