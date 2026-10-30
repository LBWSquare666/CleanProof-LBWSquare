"""Stain × surface force table (configs/force_table.yaml): starting force, escalation rule and per-surface cap.

The planner prompt, guardrail, rule policy and skill server all share the rules here:
- starting force = forces[stain][surface]; when the stain type cannot be told (unknown), start from that surface's wet_spill;
- still dirty after a re-check: previous force + escalate_step_n, not above that surface's max_n; if already at max_n and still dirty, ask_human;
- always within [min_n, hard_max_n].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from ..common.config import load_yaml

UNKNOWN_FALLBACK = "wet_spill"
EPS = 1e-6
ESCALATE_RANGE_N = (1.0, 5.0)        # same range as the "extra force per re-wipe" slider in the settings page (cleanproof/ui/operator.py read_settings)
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForceTable:
    hard_max_n: float
    min_n: float
    step_n: float
    surfaces: dict[str, dict[str, Any]]
    stains: dict[str, dict[str, Any]]
    forces: dict[str, dict[str, float]]

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "ForceTable":
        t = cls(float(cfg["hard_max_n"]), float(cfg["min_n"]), float(cfg["escalate_step_n"]),
                dict(cfg["surfaces"]), dict(cfg["stains"]),
                {s: {k: float(v) for k, v in row.items()} for s, row in cfg["forces"].items()})
        t.validate()
        return t

    def validate(self) -> None:
        for stain, row in self.forces.items():
            if stain not in self.stains:
                raise ValueError(f"force table stain {stain!r} has no label")
            if set(row) != set(self.surfaces):
                raise ValueError(f"surfaces of force table row {stain} do not match surfaces")
            for surface, f in row.items():
                if not self.min_n - EPS <= f <= self.cap(surface) + EPS:
                    raise ValueError(f"{stain} × {surface} = {f} N is outside [{self.min_n}, {self.cap(surface)}] N")

    def check_surface(self, surface: str) -> None:
        if surface not in self.surfaces:
            raise ValueError(f"unknown surface {surface!r}, choose from {sorted(self.surfaces)}")

    def cap(self, surface: str) -> float:
        self.check_surface(surface)
        return min(float(self.surfaces[surface]["max_n"]), self.hard_max_n)

    def start(self, stain: Optional[str], surface: str) -> float:
        self.check_surface(surface)
        key = stain if stain in self.forces else UNKNOWN_FALLBACK
        return self.forces[key][surface]

    def next(self, prev_n: float, surface: str) -> Optional[float]:
        """Next force after a re-check that is still dirty; returns None once the surface cap is reached (should ask_human)."""
        cap = self.cap(surface)
        if prev_n >= cap - EPS:
            return None
        return round(min(prev_n + self.step_n, cap), 3)

    def rules_for(self, surface: str) -> dict[str, Any]:
        """Compact rules for the planner model: the column for this job's surface plus the escalation rule."""
        self.check_surface(surface)
        return {
            "surface": surface, "surface_label": self.surfaces[surface].get("label", surface),
            "start_force_n_by_stain": {s: self.forces[s][surface] for s in self.forces},
            "unknown_stain_starts_as": UNKNOWN_FALLBACK,
            "escalate_step_n": self.step_n, "surface_max_n": self.cap(surface), "min_n": self.min_n,
        }


def load_force_table(site: Optional[str] = None) -> ForceTable:
    """Read the force table (configs/site/<site>/force_table.yaml can override surface caps etc.; call reload_force_table after saving on the software settings page).
    site=None: use the current process site (current_site()); with a site given, read that site's table ("" = no site overlay).
    Read once when a job starts, using the job's site, and keep it for the whole job (TaskRunner / policy table); switching site elsewhere midway does not affect this job."""
    from ..common.config import current_site
    return _load_force_table(current_site() if site is None else site)


def clamp_site_limits(cfg: dict[str, Any], base: dict[str, Any]) -> list[str]:
    """Clamp the force-table limits of a site override to the range the settings page allows (edits cfg in place) and return a list of readable warnings.
    Same range as the settings page: the global limit does not exceed the default config's hard_max_n; each surface's max_n lies in [that surface's largest starting force, global limit]; the extra force per re-wipe lies in [1, 5] N.
    Only limit-type values are handled; a starting force in a cell that is out of range is still reported by ForceTable.validate."""
    warns: list[str] = []
    base_hard = float(base["hard_max_n"])
    hard = float(cfg["hard_max_n"])
    if hard > base_hard + EPS:
        warns.append(f"hard_max_n = {hard:g} N exceeds the allowed {base_hard:g} N; using {base_hard:g} N")
        cfg["hard_max_n"] = hard = base_hard
    for surface, row in cfg["surfaces"].items():
        starts = [float(r[surface]) for r in cfg["forces"].values() if surface in r]
        lo = max(starts) if starts else float(cfg["min_n"])
        v = float(row["max_n"])
        new = min(max(v, lo), hard)
        if abs(new - v) > EPS:
            warns.append(f"surfaces.{surface}.max_n = {v:g} N is outside the allowed {lo:g}-{hard:g} N; using {new:g} N")
            row["max_n"] = new
    step = float(cfg["escalate_step_n"])
    lo_s, hi_s = ESCALATE_RANGE_N
    new_s = min(max(step, lo_s), hi_s)
    if abs(new_s - step) > EPS:
        warns.append(f"escalate_step_n = {step:g} N is outside the allowed {lo_s:g}-{hi_s:g} N; using {new_s:g} N")
        cfg["escalate_step_n"] = new_s
    return warns


@lru_cache(maxsize=8)
def _load_force_table(site: str) -> ForceTable:
    cfg = load_yaml("force_table.yaml", site=site)
    if site:                          # a site override file may have been edited by hand: values outside the settings range are clamped with a warning, not accepted silently
        for w in clamp_site_limits(cfg, load_yaml("force_table.yaml", site="")):
            log.warning("force_table.yaml (site %s): %s", site, w)
    return ForceTable.from_cfg(cfg)


def reload_force_table() -> None:
    _load_force_table.cache_clear()


@lru_cache(maxsize=1)
def _legacy_presets() -> dict[str, float]:
    return {k: float(v) for k, v in load_yaml("robot.yaml").get("force_presets_n", {}).items()}


def decision_force_n(d: Optional[dict[str, Any]]) -> Optional[float]:
    """Target force (N) of a decision in a run file: force_n since v8; older run files only have force_preset, converted from the old three levels."""
    if not d:
        return None
    if d.get("force_n") is not None:
        return float(d["force_n"])
    p = d.get("force_preset")
    return _legacy_presets().get(str(p)) if p else None


def force_label(d: Optional[dict[str, Any]]) -> str:
    f = decision_force_n(d)
    return "" if f is None else f"{f:g} N"
