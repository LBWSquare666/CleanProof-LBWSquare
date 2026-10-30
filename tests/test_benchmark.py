"""Unit tests for the randomized simulation benchmark (configs/benchmark.yaml, cleanproof/sim/scenario_random.py, scripts/benchmark_summary.py)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cleanproof.sim import scenario_random as sr  # noqa: E402


def _summary_mod():
    spec = importlib.util.spec_from_file_location("benchmark_summary", ROOT / "scripts" / "benchmark_summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)     # type: ignore[union-attr]
    return m


def test_variant_table_is_preregistered():
    cfg = sr.load_benchmark()
    vs = sr.variant_list(cfg)
    assert len(vs) == 6 * int(cfg["n_variants_per_family"]) == 60
    groups = {}
    for v in vs:
        groups.setdefault(v.group, []).append(v)
    assert sorted(groups) == [1, 2, 3, 4, 5]
    assert all(len(g) == 12 for g in groups.values())          # 12 per group, 2 per family


def test_generator_is_deterministic():
    cfg = sr.load_benchmark()
    v = sr.variant_list(cfg)[7]
    a, b = sr.generate_variant(v, cfg), sr.generate_variant(v, cfg)
    assert sr.spec_hash(a) == sr.spec_hash(b)


def test_wilson_interval_values():
    m = _summary_mod()
    w = m.wilson(60, 60)
    assert w["rate"] == 1.0 and 0.93 < w["ci95"][0] < 0.95 and w["ci95"][1] == 1.0
    w = m.wilson(30, 60)
    assert abs(w["rate"] - 0.5) < 1e-9 and w["ci95"][0] < 0.5 < w["ci95"][1]
