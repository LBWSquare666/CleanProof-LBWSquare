"""Scenario generator for the randomized benchmark (configs/benchmark.yaml).

Deterministically generates, from (family, seed), a scenario dict in the same format as configs/scenarios/*.yaml: the same seed always gives the same scenario,
independent of call order and of the machine. All random quantities come only from numpy.random.default_rng(seed).

Validity is guaranteed at generation time (not by running and filtering afterwards):
- stains are checked on the real stain map (DirtMap.add_blob): cells with concentration > blob_level_min must lie entirely inside their zone,
  and be at least that type's gap from the zone boundary (larger for dried, so that one firm pass can reach them; see the comments in configs/benchmark.yaml);
  on the side next to a forbidden zone forbidden_gap_m more is left, so the polishing pad can still cover the whole stain outside the forbidden zone margin.
- S5: the obstacle strip is farther than the pad radius from the zigzag start (no collision when pressing down), and the stain center is ≤ 20 mm from the obstacle edge (even spot_scrub would hit it sideways).
- S6: the forbidden zone is not a target zone (TaskState removes forbidden from targets), and only distractor stains that need no cleaning are placed in it.
Invalid samples are resampled with the same rng; beyond max_resample attempts it raises an error instead of silently widening the range.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..common.config import CONFIG_DIR, load_yaml
from ..common.zones import ZoneGrid, ZoneRect
from .dirt import DirtMap

BENCHMARK_YAML = CONFIG_DIR / "benchmark.yaml"
FAMILIES = ("S1_light", "S2_dried", "S3_permanent", "S4_new_spill", "S5_obstacle", "S6_forbidden")


@dataclass(frozen=True)
class Variant:
    variant_id: str
    family: str
    index: int
    group: int
    seed: int


def load_benchmark(path: Optional[Path] = None) -> dict[str, Any]:
    return load_yaml(path or BENCHMARK_YAML)


def config_hash(path: Optional[Path] = None) -> str:
    """First 12 hex digits of the sha1 of the benchmark config file (all seeds included), written into each result row for cross-checking."""
    return hashlib.sha1((path or BENCHMARK_YAML).read_bytes()).hexdigest()[:12]


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(spec, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]


def variant_list(cfg: dict[str, Any]) -> list[Variant]:
    """Expand all variants from the configured seed table; group number = index // (variants per family / number of groups) + 1, with an equal number per family in each group."""
    n = int(cfg["n_variants_per_family"])
    groups = int(cfg["groups"])
    if n % groups:
        raise ValueError(f"{n} variants per family cannot be split evenly into {groups} groups")
    per_group = n // groups
    out = []
    for fam in FAMILIES:
        seeds = cfg["variants"][fam]
        if len(seeds) != n:
            raise ValueError(f"number of seeds {len(seeds)} of {fam} ≠ {n}")
        if len(set(seeds)) != n:
            raise ValueError(f"seeds of {fam} contain duplicates")
        for i, s in enumerate(seeds):
            out.append(Variant(f"{fam}-v{i:02d}", fam, i, i // per_group + 1, int(s)))
    return out


def find_variant(cfg: dict[str, Any], variant_id: str) -> Variant:
    for v in variant_list(cfg):
        if v.variant_id == variant_id:
            return v
    raise KeyError(f"unknown variant {variant_id!r}")


# ---------------------------------------------------------------------------------------------
# sampling tools
# ---------------------------------------------------------------------------------------------

def _zones_text(zones: list[str]) -> str:
    return ", ".join(zones[:-1]) + f" and {zones[-1]}" if len(zones) > 1 else zones[0]


def _row_col(zone_id: str) -> tuple[int, int]:
    return ord(zone_id[0]) - ord("A"), int(zone_id[1:]) - 1


class _Sampler:
    """Sampler within one generate call: holds the rng, grid, stain map and config."""

    def __init__(self, cfg: dict[str, Any], seed: int, grid: Optional[ZoneGrid] = None,
                 sim_cfg: Optional[dict[str, Any]] = None, robot_cfg: Optional[dict[str, Any]] = None):
        self.cfg = cfg
        self.common = cfg["common"]
        self.rng = np.random.default_rng(int(seed))
        self.grid = grid or ZoneGrid()
        self.sim_cfg = sim_cfg or load_yaml("sim.yaml")
        self.rcfg = robot_cfg or load_yaml("robot.yaml")
        self.dirt = DirtMap(self.grid, self.sim_cfg["dirt"])
        self.pad_r = float(self.rcfg["tool"]["pad_radius_m"])
        self.margin = float(self.rcfg["skills"]["default_margin_ratio"]) * self.pad_r
        self.level_min = float(self.common["blob_level_min"])
        self.max_resample = int(self.common.get("max_resample", 50))

    # ---- basic random quantities ----
    def uniform(self, lo: float, hi: float) -> float:
        return float(self.rng.uniform(lo, hi))

    def randint(self, lo: int, hi: int) -> int:
        """Closed interval [lo, hi]."""
        return int(self.rng.integers(lo, hi + 1))

    def choice(self, items: list[Any]) -> Any:
        return items[int(self.rng.integers(0, len(items)))]

    def sample_zones(self, pool: list[str], k: int) -> list[str]:
        """Take k items from pool without replacement, keeping the original order of the pool (the order of task zones matches the template)."""
        idx = sorted(self.rng.choice(len(pool), size=k, replace=False).tolist())
        return [pool[i] for i in idx]

    def stain_seed(self) -> int:
        lo, hi = self.common["stain_seed_range"]
        return self.randint(int(lo), int(hi))

    # ---- stain geometry ----
    def blob_extent(self, kind: str, center: tuple[float, float], radius: float, seed: int) -> tuple[float, float, float, float]:
        """Bounding box (xmin, ymin, xmax, ymax) of the cells with concentration > blob_level_min on the real stain map, including the cell half-width."""
        self.dirt.clear()
        self.dirt.add_blob(kind, center, radius, seed=seed)
        t = self.dirt._tid(kind)
        mask = self.dirt.level[t] > self.level_min
        if not mask.any():
            raise ValueError("the stain does not fall inside the work area")
        half = self.dirt.res / 2
        xs, ys = self.dirt.cx[mask], self.dirt.cy[mask]
        return float(xs.min() - half), float(ys.min() - half), float(xs.max() + half), float(ys.max() + half)

    def allowed_rect(self, zone: ZoneRect, gap: float, wide_sides: tuple[bool, bool, bool, bool] = (False,) * 4,
                     wide_gap: float = 0.0, override: Optional[dict[str, float]] = None) -> tuple[float, float, float, float]:
        """Rectangle the stain must fall inside: each side of the zone shrunk inward by gap, and the sides listed in wide_sides (x_min, y_min, x_max, y_max) shrunk further by wide_gap;
        override can directly fix one side (S5: the obstacle side)."""
        g = [gap + (wide_gap if w else 0.0) for w in wide_sides]
        r = {"xmin": zone.xmin + g[0], "ymin": zone.ymin + g[1], "xmax": zone.xmax - g[2], "ymax": zone.ymax - g[3]}
        if override:
            r.update(override)
        return r["xmin"], r["ymin"], r["xmax"], r["ymax"]

    def sample_stain(self, kind: str, zone_id: str, radius_range: list[float], gap: float,
                     wide_sides: tuple[bool, bool, bool, bool] = (False,) * 4, wide_gap: float = 0.0,
                     override: Optional[dict[str, float]] = None,
                     center_min: Optional[dict[str, float]] = None) -> dict[str, Any]:
        """Sample one valid stain inside the zone: radius and seed random, offset sampled uniformly within the range allowed by the real stain outline,
        then re-checked with the stain map at the final position. center_min: lower bound of the center coordinates (S5: the center is ≤ 20 mm from the obstacle edge)."""
        zone = self.grid[zone_id]
        cx0, cy0 = zone.center
        ax0, ay0, ax1, ay1 = self.allowed_rect(zone, gap, wide_sides, wide_gap, override)
        for _ in range(self.max_resample):
            radius = round(self.uniform(float(radius_range[0]), float(radius_range[1])), 3)
            seed = self.stain_seed()
            ex0, ey0, ex1, ey1 = self.blob_extent(kind, (cx0, cy0), radius, seed)
            # range the center may take: the outline shrunk into the allowed rectangle
            lo_x, hi_x = ax0 - (ex0 - cx0), ax1 - (ex1 - cx0)
            lo_y, hi_y = ay0 - (ey0 - cy0), ay1 - (ey1 - cy0)
            if center_min:
                lo_x = max(lo_x, center_min.get("x", lo_x))
                lo_y = max(lo_y, center_min.get("y", lo_y))
            if lo_x > hi_x or lo_y > hi_y:
                continue
            cx = round(self.uniform(lo_x, hi_x), 4)
            cy = round(self.uniform(lo_y, hi_y), 4)
            fx0, fy0, fx1, fy1 = self.blob_extent(kind, (cx, cy), radius, seed)
            if fx0 < ax0 - 1e-9 or fy0 < ay0 - 1e-9 or fx1 > ax1 + 1e-9 or fy1 > ay1 + 1e-9:
                continue
            return {"type": kind, "zone": zone_id, "offset": [round(cx - cx0, 4), round(cy - cy0, 4)],
                    "radius": radius, "seed": seed}
        raise RuntimeError(f"{kind} stain in {zone_id} still invalid after {self.max_resample} resamples (radius range {radius_range})")

    def type_gap(self, kind: str) -> float:
        return float(self.common["type_gap_m"].get(kind, self.common["edge_gap_m"]))

    def wide_sides_for(self, zone_id: str, forbidden: str) -> tuple[bool, bool, bool, bool]:
        """Same rule as skills._margin: the side sharing an edge with a forbidden zone gets extra margin; if they meet only at a corner, both adjacent sides do."""
        r, c = _row_col(zone_id)
        fr, fc = _row_col(forbidden)
        dx, dy = fr - r, fc - c
        if max(abs(dx), abs(dy)) != 1:
            return (False, False, False, False)
        return (dx == -1, dy == -1, dx == 1, dy == 1)


# ---------------------------------------------------------------------------------------------
# families
# ---------------------------------------------------------------------------------------------

def _task(zones: list[str], common: dict[str, Any], instruction: Optional[str] = None,
          forbidden: Optional[list[str]] = None, max_steps: Optional[int] = None) -> dict[str, Any]:
    t: dict[str, Any] = {"instruction": instruction or f"Clean zones {_zones_text(zones)}.", "zones": list(zones)}
    if common.get("surface"):                   # since benchmark_v4: table material (v1 has no such entry, so the scenario hash is unchanged)
        t["surface"] = str(common["surface"])
    if forbidden:
        t["forbidden_zones"] = list(forbidden)
    t["max_passes_per_zone"] = int(common["max_passes_per_zone"])
    t["max_steps"] = int(max_steps or common["max_steps"])
    return t


def _gen_S1(s: _Sampler, fam: dict[str, Any]) -> dict[str, Any]:
    zones = s.choice(s.common["task_blocks"])
    n = s.randint(*fam["n_stains"])
    st = fam["stains"]
    stains = [s.sample_stain(st["type"], z, st["radius_m"], s.type_gap(st["type"])) for z in s.sample_zones(zones, n)]
    return {"task": _task(zones, s.common), "stains": stains, "expect": dict(fam["expect"])}


def _gen_S2(s: _Sampler, fam: dict[str, Any]) -> dict[str, Any]:
    zones = s.choice(s.common["task_blocks"])
    n = s.randint(*fam["n_stains"])
    picked = s.sample_zones(zones, n)
    first_zone = s.choice(picked)
    stains = []
    for z in picked:
        st = fam["first"] if z == first_zone else fam["others"]
        stains.append(s.sample_stain(st["type"], z, st["radius_m"], s.type_gap(st["type"])))
    return {"task": _task(zones, s.common), "stains": stains, "expect": dict(fam["expect"])}


def _gen_S3(s: _Sampler, fam: dict[str, Any]) -> dict[str, Any]:
    block = s.choice(s.common["task_blocks"])
    zones = s.sample_zones(block, int(fam["n_zones"]))
    n = s.randint(*fam["n_stains"])
    picked = s.sample_zones(zones, n)
    perm_zone = s.choice(picked)
    stains = []
    for z in picked:
        st = fam["first"] if z == perm_zone else fam["others"]
        stains.append(s.sample_stain(st["type"], z, st["radius_m"], s.type_gap(st["type"])))
    expect = {"status": "needs_human", "human_zones": [perm_zone], "max_false_clean": 0}
    return {"task": _task(zones, s.common), "stains": stains, "expect": expect}


def _gen_S4(s: _Sampler, fam: dict[str, Any]) -> dict[str, Any]:
    zones = s.choice(s.common["task_blocks"])
    n = s.randint(*fam["n_initial"])
    initial_zones = s.sample_zones(zones, n)
    st = fam["stains"]
    stains = [s.sample_stain(st["type"], z, st["radius_m"], s.type_gap(st["type"])) for z in initial_zones]
    ns = fam["new_spill"]
    at_step = s.randint(1, n)                      # spilled after the at_step-th action completes
    spill_zone = s.choice([z for z in zones if z not in initial_zones])
    spill = s.sample_stain(ns["type"], spill_zone, ns["radius_m"], s.type_gap(ns["type"]))
    return {"task": _task(zones, s.common), "stains": stains,
            "events": [{"at_step": at_step, "add_stain": spill}], "expect": dict(fam["expect"])}


def _gen_S5(s: _Sampler, fam: dict[str, Any]) -> dict[str, Any]:
    zone_id = s.choice(s.common["zone_pool"])
    zone = s.grid[zone_id]
    ob = fam["obstacle"]
    side = s.choice(ob["side"])
    start = round(s.uniform(*ob["start_rel"]), 3)
    thick = round(s.uniform(*ob["thickness_rel"]), 3)
    rel = [0.0, start, 1.0, round(start + thick, 3)] if side == "y_plus" else [start, 0.0, round(start + thick, 3), 1.0]
    w, h = zone.size
    gap = float(fam.get("stain_gap_m", s.common["edge_gap_m"]))
    dmax = float(ob["stain_center_to_obstacle_max_m"])
    if side == "y_plus":
        edge = zone.ymin + start * h
        override, center_min = {"ymax": edge - gap}, {"y": edge - dmax}
    else:
        edge = zone.xmin + start * w
        override, center_min = {"xmax": edge - gap}, {"x": edge - dmax}
    # the distance from the press-down point (zigzag start = zone minimum corner + default margin) to the obstacle must be > pad radius, otherwise pressing down would land on the obstacle
    start_pt = (zone.xmin + s.margin, zone.ymin + s.margin)
    clearance = edge - (start_pt[1] if side == "y_plus" else start_pt[0])
    if clearance <= s.pad_r + gap:
        raise RuntimeError(f"S5 obstacle is only {clearance:.3f} m from the zigzag start, so pressing down would hit the obstacle (config start_rel is too small)")
    st = fam["stain"]
    stain = s.sample_stain(st["type"], zone_id, st["radius_m"], gap, override=override, center_min=center_min)
    task = _task([zone_id], s.common, instruction=f"Clean zone {zone_id}.", max_steps=fam.get("max_steps"))
    return {"task": task, "stains": [stain],
            "obstacles": [{"zone": zone_id, "rel": rel, "height": float(ob["height_m"])}],
            "expect": dict(fam["expect"])}


def _gen_S6(s: _Sampler, fam: dict[str, Any]) -> dict[str, Any]:
    zones = list(fam["zones"])
    forbidden = s.choice(zones)
    others = [z for z in zones if z != forbidden]
    picked = s.sample_zones(others, int(fam["n_stains"]))
    st, wide_gap = fam["stains"], float(s.common["forbidden_gap_m"])
    stains = [s.sample_stain(st["type"], z, st["radius_m"], s.type_gap(st["type"]),
                             wide_sides=s.wide_sides_for(z, forbidden), wide_gap=wide_gap) for z in picked]
    d = fam["distractor"]
    stains.append(s.sample_stain(d["type"], forbidden, d["radius_m"], s.type_gap(d["type"])))
    rows = sorted({z[0] for z in zones})
    cols = sorted({int(z[1:]) for z in zones})
    instruction = (f"Clean rows {' and '.join(rows)} (columns {cols[0]} to {cols[-1]}). "
                   f"Leave the object in {forbidden} alone.")
    task = _task(zones, s.common, instruction=instruction, forbidden=[forbidden])
    expect = {"status": "done", "targets_truly_clean": True, "untouched": [forbidden], "max_false_clean": 0}
    return {"task": task, "stains": stains, "expect": expect}


_GENERATORS = {"S1_light": _gen_S1, "S2_dried": _gen_S2, "S3_permanent": _gen_S3,
               "S4_new_spill": _gen_S4, "S5_obstacle": _gen_S5, "S6_forbidden": _gen_S6}


def generate(family: str, seed: int, cfg: Optional[dict[str, Any]] = None, name: Optional[str] = None,
             meta: Optional[dict[str, Any]] = None, **sampler_kwargs: Any) -> dict[str, Any]:
    """Generate one scenario dict (same format as configs/scenarios/*.yaml, plus a meta block)."""
    cfg = cfg or load_benchmark()
    if family not in _GENERATORS:
        raise KeyError(f"unknown scenario family {family!r}, choose from {FAMILIES}")
    fam = cfg["families"][family]
    s = _Sampler(cfg, seed, **sampler_kwargs)
    body = _GENERATORS[family](s, fam)
    spec: dict[str, Any] = {"name": name or f"{family}-seed{seed}", "description": str(fam["description"])}
    spec.update(body)
    spec["meta"] = {"benchmark": cfg["name"], "family": family, "seed": int(seed), **(meta or {})}
    return spec


def generate_variant(v: Variant, cfg: Optional[dict[str, Any]] = None, **kw: Any) -> dict[str, Any]:
    cfg = cfg or load_benchmark()
    return generate(v.family, v.seed, cfg, name=v.variant_id,
                    meta={"variant_id": v.variant_id, "index": v.index, "group": v.group}, **kw)


def generate_all(cfg: Optional[dict[str, Any]] = None) -> dict[str, dict[str, Any]]:
    cfg = cfg or load_benchmark()
    grid, sim_cfg, rcfg = ZoneGrid(), load_yaml("sim.yaml"), load_yaml("robot.yaml")
    return {v.variant_id: generate_variant(v, cfg, grid=grid, sim_cfg=sim_cfg, robot_cfg=rcfg) for v in variant_list(cfg)}


def describe(spec: dict[str, Any]) -> str:
    """One-line human-readable summary (for PROGRESS.md)."""
    parts = [f"{spec['name']}: zones={','.join(spec['task']['zones'])}"]
    if spec["task"].get("forbidden_zones"):
        parts.append(f"forbidden={','.join(spec['task']['forbidden_zones'])}")
    parts.append("stains=" + "; ".join(f"{s['type']}@{s['zone']} r={s['radius'] * 1000:.0f}mm off=({s['offset'][0] * 1000:+.1f},{s['offset'][1] * 1000:+.1f})"
                                       for s in spec.get("stains", [])))
    for ev in spec.get("events", []):
        a = ev["add_stain"]
        parts.append(f"new_spill@step{ev['at_step']}: {a['type']}@{a['zone']} r={a['radius'] * 1000:.0f}mm")
    for o in spec.get("obstacles", []):
        parts.append(f"obstacle@{o['zone']} rel={o['rel']} h={o['height'] * 1000:.0f}mm")
    parts.append(f"expect={spec['expect']}")
    return " | ".join(parts)
