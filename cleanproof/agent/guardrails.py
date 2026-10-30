"""Decision guardrail: the model only proposes; the proposal is checked here against the task state. A non-compliant decision gets a reason, and the loop re-asks or turns it into ask_human.

Checks: the zone exists and belongs to the task, and is not a forbidden zone; only wipe residue / uncertain zones; per-zone pass budget;
no finish until everything is clean; at most one close look per zone per round; step budget;
force follows the stain × surface force table (configs/force_table.yaml): within [min_n, surface cap]; the first wipe does not exceed the table starting force;
a zone that still has residue after a wipe must go up exactly one level (previous + escalate_step_n, capped at the surface cap; only with missed_coverage may the same force be kept with a different wipe pattern),
otherwise it is replaced with a zigzag one level up; if already at the cap and still dirty, ask_human.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..common.schemas import Action, Decision, LikelyCause, Pattern, ZoneStatus
from .force_table import EPS, ForceTable, load_force_table
from .state import TaskState, decision_key


@dataclass
class GuardResult:
    ok: bool
    reason: str = ""
    fallback: Optional[Decision] = None   # replacement decision when it cannot be corrected (ask_human / finish)


def ask_human(reason: str, zones: Optional[list[str]] = None) -> Decision:
    return Decision(action=Action.ask_human, zone_ids=zones or [], reason=reason[:500])


SMALL_RESIDUE_AREA = 0.25   # upper bound of residue box area as a fraction of the zone: below it counts as "local" (same as rule policy and prompt)


def _bbox_area(b: Optional[list[float]]) -> float:
    return (b[2] - b[0]) * (b[3] - b[1]) if b else 1.0


def _force_problem(decision: Decision, state: TaskState, table: ForceTable) -> Optional[GuardResult]:
    """Force check: range, starting force of the first wipe, escalation after a re-check that is still dirty."""
    f = float(decision.force_n)
    surface = state.task.surface
    cap = table.cap(surface)
    if f < table.min_n - EPS or f > cap + EPS:
        return GuardResult(False, f"force {f:g} N is outside the allowed range {table.min_n:g}–{cap:g} N for {surface}")
    for z in decision.zone_ids:
        r = state.zone.get(z)
        prev = state.last_force_n.get(z)
        if prev is None:
            start = table.start(r.stain_type.value if r else None, surface)
            if f > start + EPS:
                stain = r.stain_type.value if r else "unknown"
                return GuardResult(False, f"{z}: first wipe of a {stain} stain on {surface} starts at {start:g} N "
                                          f"(force table), not {f:g} N")
            continue
        nxt = table.next(prev, surface)
        if r is None or r.status != ZoneStatus.residue:          # uncertain: may wipe once more at the same force, at most one level up
            if f > (nxt if nxt is not None else prev) + EPS:
                return GuardResult(False, f"{z}: at most one step above the last wipe ({prev:g} N)")
            continue
        new_pattern = "spot" if decision.action == Action.spot_scrub else (decision.pattern.value if decision.pattern else "zigzag")
        # The missed-coverage exception only holds for local residue: large residue that survives a same-force wipe means too little force, not a missed spot (S2 demo take3)
        if (abs(f - prev) <= EPS and decision.action == Action.wipe_zone
                and r.likely_cause == LikelyCause.missed_coverage and new_pattern != state.last_pattern.get(z)
                and _bbox_area(r.residue_bbox_norm) <= SMALL_RESIDUE_AREA):
            continue
        if nxt is not None and abs(f - nxt) <= EPS:
            continue
        if nxt is None:
            fb = ask_human(f"{z} still shows residue after a {prev:g} N wipe, the limit for {surface}", [z])
            return GuardResult(False, f"{z} still shows residue at the {surface} limit ({cap:g} N); ask a person", fb)
        fb = Decision(action=Action.wipe_zone, zone_ids=[z], force_n=nxt, pattern=Pattern.zigzag, passes=1,
                      reason=f"guardrail: residue survived a {prev:g} N wipe, escalate to {nxt:g} N")
        return GuardResult(False, f"{z} still shows residue after a {prev:g} N {state.last_pattern.get(z, '')} wipe; "
                                  f"the next wipe must use {nxt:g} N (last + {table.step_n:g} N, surface limit {cap:g} N; "
                                  f"same force only for missed_coverage with a different wipe_zone pattern)", fb)
    return None


def check(decision: Decision, state: TaskState, table: Optional[ForceTable] = None) -> GuardResult:
    a = decision.action
    table = table or load_force_table()
    if state.steps_left() <= 0:
        if state.all_clean():
            return GuardResult(False, "step budget exhausted", Decision(action=Action.finish, reason="all target zones clean"))
        return GuardResult(False, "step budget exhausted", ask_human("step budget exhausted with zones still dirty", state.dirty()))

    for z in decision.zone_ids:
        if z not in state.grid:
            return GuardResult(False, f"unknown zone {z}")
        if z in state.forbidden:
            return GuardResult(False, f"{z} is a forbidden zone")
        if a != Action.ask_human and z not in state.targets:
            return GuardResult(False, f"{z} is not part of this task")

    key = decision_key(decision)
    if key in state.skill_rejections:
        return GuardResult(False, f"the robot already rejected this same action: {state.skill_rejections[key]}",
                           ask_human(f"robot rejected the planned action: {state.skill_rejections[key]}",
                                     decision.zone_ids))

    if a in (Action.wipe_zone, Action.spot_scrub):
        for z in decision.zone_ids:
            st = state.status(z)
            if st == ZoneStatus.clean:
                return GuardResult(False, f"{z} is already judged clean, no need to wipe")
            if state.remaining_passes(z) < decision.passes:
                return GuardResult(False, f"{z} has too few wipe passes left ({state.remaining_passes(z)} left)",
                                   ask_human(f"{z} still dirty after {state.passes.get(z, 0)} passes", [z]))
        if a == Action.spot_scrub:
            z = decision.zone_ids[0]
            r = state.zone.get(z)
            area = _bbox_area(r.residue_bbox_norm if r else None)
            if area > SMALL_RESIDUE_AREA:
                return GuardResult(False, f"{z} residue covers {area:.0%} of the zone; spot_scrub is only for residue "
                                          f"under {SMALL_RESIDUE_AREA:.0%} — use wipe_zone")
        esc = _force_problem(decision, state, table)
        if esc is not None:
            return esc
    elif a == Action.inspect_closeup:
        z = decision.zone_ids[0]
        if z in state.closeup_done:
            return GuardResult(False, f"{z} was already closely inspected this round; decide to wipe or ask a human directly")
    elif a == Action.finish:
        dirty = state.dirty()
        if dirty:
            return GuardResult(False, f"zones {dirty} are still not clean, cannot finish")
    return GuardResult(True)
