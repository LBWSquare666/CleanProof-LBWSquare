"""Decision policies.

- NemotronPolicy: Nemotron 3 Super (Nebius Token Factory) outputs the next skill as structured output; the decision maker used in the demo;
- RulePolicy: deterministic rules (tests, offline regression, and the comparison when Token Factory is unreachable);
- FixedPolicy: comparison baseline "wipe once / twice, no checking".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from ..common.llm_client import LLMClient, classify_model_error
from ..common.schemas import Action, Decision, LikelyCause, Pattern, ZoneStatus
from .force_table import ForceTable, load_force_table
from .guardrails import SMALL_RESIDUE_AREA, ask_human
from .state import TaskState, decision_key

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class PolicyDecision:
    decision: Decision
    meta: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    name: str
    uses_guard: bool

    def decide(self, state: TaskState, feedback: Optional[str] = None) -> PolicyDecision: ...


def _bbox_area(b: Optional[list[float]]) -> float:
    return (b[2] - b[0]) * (b[3] - b[1]) if b else 1.0


class RulePolicy:
    name = "rules"
    uses_guard = True

    def __init__(self, table: Optional[ForceTable] = None):
        self.table = table or load_force_table()

    def decide(self, state: TaskState, feedback: Optional[str] = None) -> PolicyDecision:
        if state.all_clean():
            return PolicyDecision(Decision(action=Action.finish, reason="all target zones are clean"))
        dirty = state.dirty()
        residue = [z for z in dirty if state.status(z) == ZoneStatus.residue]
        uncertain = [z for z in dirty if state.status(z) == ZoneStatus.uncertain]
        for z in uncertain:
            if z not in state.closeup_done:
                return PolicyDecision(Decision(action=Action.inspect_closeup, zone_ids=[z],
                                               reason=f"{z} is uncertain; take a closer look"))
        # Wipe the cells that still have options first; a cell that cannot be wiped off (cap reached / enough passes / all skills rejected) is not handed to a human right away, only after the other cells are done and nothing else can be tried
        stuck: list[PolicyDecision] = []
        for z in residue + [u for u in uncertain if u not in residue]:
            cand = self._candidate(state, z)
            if decision_key(cand.decision) in state.skill_rejections and cand.decision.action == Action.spot_scrub:
                cand = PolicyDecision(Decision(action=Action.wipe_zone, zone_ids=[z], force_n=cand.decision.force_n,
                                               pattern=Pattern.zigzag, reason=f"{z}: spot scrub was rejected; wipe the zone"))
            if decision_key(cand.decision) in state.skill_rejections:
                cand = PolicyDecision(ask_human(f"{z}: robot rejected every planned wipe", [z]))
            if cand.decision.action != Action.ask_human:
                return cand
            stuck.append(cand)
        if len(stuck) == 1:
            return stuck[0]
        return PolicyDecision(ask_human("; ".join(c.decision.reason for c in stuck), [z for c in stuck for z in c.decision.zone_ids]))

    def _candidate(self, state: TaskState, z: str) -> PolicyDecision:
        if state.remaining_passes(z) < 1:
            return PolicyDecision(ask_human(f"{z} still not clean after {state.passes.get(z, 0)} passes", [z]))
        r = state.zone[z]
        surface = state.task.surface
        last = state.last_force_n.get(z)
        small = bool(r.residue_bbox_norm) and _bbox_area(r.residue_bbox_norm) < SMALL_RESIDUE_AREA
        if last is None:
            force = self.table.start(r.stain_type.value, surface)
        elif r.status != ZoneStatus.residue:            # uncertain: wipe once more at the same force
            force = last
        elif (r.likely_cause == LikelyCause.missed_coverage and small
              and state.last_pattern.get(z) != Pattern.edge_trace.value):
            return PolicyDecision(Decision(action=Action.wipe_zone, zone_ids=[z], force_n=last,
                                           pattern=Pattern.edge_trace, reason=f"{z}: residue at edges (missed coverage)"))
        else:
            nxt = self.table.next(last, surface)
            if nxt is None:
                return PolicyDecision(ask_human(f"{z} residue survived a {last:g} N wipe, the limit for {surface}; "
                                                f"likely permanent", [z]))
            force = nxt
        if small:
            return PolicyDecision(Decision(action=Action.spot_scrub, zone_ids=[z], force_n=force,
                                           bbox_norm=r.residue_bbox_norm, reason=f"{z}: small localized residue"))
        return PolicyDecision(Decision(action=Action.wipe_zone, zone_ids=[z], force_n=force, pattern=Pattern.zigzag,
                                       reason=f"{z}: {r.status.value} ({r.stain_type.value}, {r.likely_cause.value})"))


class FixedPolicy:
    """Comparison baseline: wipe each task zone with a fixed number of passes (fixed force force_n, zigzag), ignore verification results, then finish."""

    uses_guard = False

    def __init__(self, passes: int = 1, force_n: float = 10.0):
        self.passes, self.force_n = passes, float(force_n)
        self.name = f"fixed_{passes}x"
        self._i = 0

    def decide(self, state: TaskState, feedback: Optional[str] = None) -> PolicyDecision:
        if self._i < len(state.targets):
            z = state.targets[self._i]
            self._i += 1
            return PolicyDecision(Decision(action=Action.wipe_zone, zone_ids=[z], force_n=self.force_n,
                                           pattern=Pattern.zigzag, passes=self.passes, reason="fixed program"))
        return PolicyDecision(Decision(action=Action.finish, reason="fixed program complete"))


class NemotronPolicy:
    name = "nemotron"
    uses_guard = True

    def __init__(self, client: LLMClient, table: Optional[ForceTable] = None, role: str = "decision",
                 prompt_file: str = "decision_system.md"):
        self.client, self.role = client, role
        self.table = table or load_force_table()
        self.prompt_file = prompt_file                    # E16: decision_system_v2.md (pick force per zone + budget rules), evaluated only on new seeds
        self.system_prompt = (PROMPT_DIR / prompt_file).read_text(encoding="utf-8")

    def _ask(self, state: TaskState, feedback: Optional[str]) -> tuple[Optional[Decision], dict[str, Any], Optional[str]]:
        user = json.dumps(state.summary(self.table.rules_for(state.task.surface)), ensure_ascii=False)
        if feedback:
            user += f"\n\nYour previous choice was rejected: {feedback}\nChoose again."
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        res = self.client.chat(self.role, messages, schema=Decision, tag=f"step{state.step}")
        meta = {"request_id": res.request_id, "model": res.model, "backend": res.backend,
                "latency_s": round(res.latency_s, 3), "cost_usd": round(res.cost_usd, 6),
                "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens}
        return (res.parsed if res.schema_ok else None), meta, res.schema_error

    def decide(self, state: TaskState, feedback: Optional[str] = None) -> PolicyDecision:
        metas = []
        try:
            d, meta, err = self._ask(state, feedback)
            metas.append(meta)
            if d is None:
                d, meta, err2 = self._ask(state, f"invalid JSON or parameters ({err}). Reply with JSON only.")
                metas.append(meta)
                if d is None:
                    return PolicyDecision(ask_human(f"decision model returned invalid output twice: {err2}"),
                                          {"calls": metas, "fallback": True})
        except Exception as e:  # Token Factory unreachable / key missing or rejected / over budget: do not guess. This is not "ask a person to check the dirt"; the loop ends with model_unavailable based on model_error
            ce = classify_model_error(e)
            return PolicyDecision(ask_human(f"{ce['message']} {ce['fix']}"),
                                  {"calls": metas, "error": f"{type(e).__name__}: {e}"[:300], "fallback": True, "model_error": ce})
        return PolicyDecision(d, {"calls": metas})
