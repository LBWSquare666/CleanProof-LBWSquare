"""Task state: zone status, wipe count per zone, step history, and a compact summary for the decision model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..common.schemas import Decision, TaskSpec, VerifierZoneResult, ZoneStatus
from ..common.zones import ZoneGrid


@dataclass
class StepRecord:
    step: int
    decision: dict[str, Any]
    policy: str
    guard: str
    outcome: dict[str, Any]
    inspected: dict[str, str] = field(default_factory=dict)


class TaskState:
    def __init__(self, task: TaskSpec, grid: ZoneGrid):
        self.task = task
        self.grid = grid
        forbidden = set(task.forbidden_zones)
        base = task.zones or grid.ids
        self.targets = [z for z in base if z in grid and z not in forbidden]
        self.forbidden = sorted(forbidden)
        self.zone: dict[str, VerifierZoneResult] = {}
        self.provenance: dict[str, str] = {}
        self.passes: dict[str, int] = {z: 0 for z in self.targets}
        self.last_force: dict[str, dict[str, Any]] = {}
        self.last_force_n: dict[str, float] = {}     # target force (N) of the last wipe per zone
        self.wipe_stain: dict[str, str] = {}         # stain type used at the first wipe per zone (for report and video)
        self.last_pattern: dict[str, str] = {}       # zigzag / spiral / edge_trace / spot
        self.closeup_done: set[str] = set()
        self.history: list[StepRecord] = []
        self.human_requests: list[dict[str, Any]] = []
        self.step = 0
        self.initial_status: dict[str, str] = {}
        self.skill_rejections: dict[str, str] = {}   # action key → reason the executor rejected it
        self.final_checks = 0

    # ---- Update ----
    def update_inspection(self, results: dict[str, VerifierZoneResult], provenance: dict[str, str]) -> None:
        for z, r in results.items():
            if z not in self.targets:
                continue
            prev = self.zone.get(z)
            self.zone[z] = r
            self.provenance[z] = provenance.get(z, "")
            if z not in self.initial_status:
                self.initial_status[z] = r.status.value
            if prev is not None and prev.status != r.status:
                self.closeup_done.discard(z)

    def record_wipe(self, zone_ids: list[str], passes: int, force: dict[str, dict[str, Any]],
                    force_n: Optional[float] = None, pattern: Optional[str] = None) -> None:
        for z in zone_ids:
            self.passes[z] = self.passes.get(z, 0) + passes
            if z not in self.wipe_stain and z in self.zone:
                self.wipe_stain[z] = self.zone[z].stain_type.value
            if force_n is not None:
                self.last_force_n[z] = float(force_n)
            if pattern:
                self.last_pattern[z] = pattern
        self.last_force.update(force)

    # ---- Query ----
    def status(self, z: str) -> Optional[ZoneStatus]:
        r = self.zone.get(z)
        return r.status if r else None

    def dirty(self) -> list[str]:
        return [z for z in self.targets if self.status(z) in (ZoneStatus.residue, ZoneStatus.uncertain)]

    def all_clean(self) -> bool:
        return all(self.status(z) == ZoneStatus.clean for z in self.targets)

    def remaining_passes(self, z: str) -> int:
        return self.task.max_passes_per_zone - self.passes.get(z, 0)

    def steps_left(self) -> int:
        return self.task.max_steps - self.step

    def summary(self, force_rules: dict[str, Any]) -> dict[str, Any]:
        """Summary for the decision model: expand only non-clean zones, list clean zones by ID only."""
        zones = []
        for z in self.targets:
            r = self.zone.get(z)
            if r is None or r.status == ZoneStatus.clean:
                continue
            item = {
                "zone_id": z, "status": r.status.value, "confidence": r.confidence,
                "stain_type": r.stain_type.value,
                "likely_cause": r.likely_cause.value, "residue_bbox_norm": r.residue_bbox_norm,
                "evidence": r.evidence, "passes_done": self.passes.get(z, 0),
                "passes_left": self.remaining_passes(z), "closeup_done": z in self.closeup_done,
                "source": self.provenance.get(z, ""),
                "last_force_n": self.last_force_n.get(z),
                "last_pattern": self.last_pattern.get(z),
            }
            if z in self.last_force:
                f = self.last_force[z]
                item["last_wipe"] = {k: round(float(v), 3) for k, v in f.items()}
            zones.append(item)
        rejected = [{"action": k, "robot_reason": v} for k, v in self.skill_rejections.items()]
        last = None
        if self.history:
            h = self.history[-1]
            last = {"decision": h.decision, "guard": h.guard, "outcome_status": h.outcome.get("status"),
                    "outcome_reason": h.outcome.get("reason", ""), "inspected_after": h.inspected}
        return {
            "task": {"instruction": self.task.instruction, "surface": self.task.surface, "target_zones": self.targets,
                     "forbidden_zones": self.forbidden, "max_passes_per_zone": self.task.max_passes_per_zone},
            "step": self.step, "steps_left": self.steps_left(),
            "zones_needing_attention": zones,
            "clean_zones": [z for z in self.targets if self.status(z) == ZoneStatus.clean],
            "last_step": last,
            "actions_rejected_by_robot": rejected,
            "force_rules": force_rules,
        }


def decision_to_dict(d: Decision) -> dict[str, Any]:
    return d.model_dump(mode="json", exclude_none=True)


def decision_key(d: Decision) -> str:
    """Identity of an action (without the reason text), used to recognize a rejected action being proposed again."""
    bbox = [round(v, 2) for v in d.bbox_norm] if d.bbox_norm else None
    return f"{d.action.value}|{','.join(d.zone_ids)}|{'' if d.force_n is None else f'{d.force_n:g}'}|" \
           f"{d.pattern.value if d.pattern else ''}|{d.passes}|{bbox}"
