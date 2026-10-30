"""Main task loop: survey → inspect → decide (guardrail) → [approve] → execute → re-check ... → finish → audit.

Executor, camera source, verifier and decision policy are all injected through parameters; simulation and the real robot use the same loop.
All events are written to runs/<run_id>/events.jsonl; the console and the report read the same record.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from ..common.llm_client import LLMClient
from ..common.schemas import Action, Decision, TaskSpec
from ..robot.skills import SkillExecutor, SkillOutcome
from . import guardrails
from .force_table import ForceTable
from .policy import Policy
from .runlog import RunLogger
from .state import StepRecord, TaskState, decision_key, decision_to_dict

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class LoopOptions:
    rescan_all: bool = True        # every re-check also screens already-clean zones through the cheap channel (finds newly spilled stains, scene S4); not needed when recheck_all is set
    recheck_all: bool = True       # after every wipe pass the verifier (Cosmos 3) re-checks all zones; False = old logic, only re-check the wiped cells,
                                   # the other already-clean cells are screened through the cheap channel (benchmark_v4 / V4_fixed15 used the old logic)
    audit: bool = False            # after finishing, call the audit role (Nemotron 3 Ultra) to write an audit summary
    save_traces: bool = True
    # Ablation switches (for simulation experiments E4 / E5); the defaults are the official system
    guard: bool = True             # False: the guardrail only records "would have blocked" (guard_would_reject), the action still runs (skill-layer safety limits still apply)
    final_check: bool = True       # False: after the planner says finish, only one recording-purpose final check is done; residue found does not continue the loop


class InspectionAborted(RuntimeError):
    """Failed to return to the survey pose (skill rejected / fault / network): the arm may still block the camera, so this image cannot be used for inspection."""

    def __init__(self, outcome: SkillOutcome):
        super().__init__(f"survey {outcome.status}: {outcome.reason}")
        self.outcome = outcome


Injector = Callable[[int, TaskState], None]
Approver = Callable[[Decision, TaskState], bool]
TruthFn = Callable[[], dict[str, bool]]   # simulation only: whether each zone is truly clean; written to the log for evaluation only, not used in decisions


class TaskRunner:
    def __init__(self, task: TaskSpec, executor: SkillExecutor, source: Any, verifier: Any, policy: Policy,
                 logger: Optional[RunLogger] = None, options: Optional[LoopOptions] = None,
                 approve: Optional[Approver] = None, audit_client: Optional[LLMClient] = None,
                 injector: Optional[Injector] = None, truth: Optional[TruthFn] = None,
                 force_table: Optional[ForceTable] = None):
        self.task, self.ex, self.src, self.ver, self.policy = task, executor, source, verifier, policy
        # Force table for this job (read once at job start, for this job's site): used by the guardrail at every step; None = old behavior (read per step for the current process site)
        self.table = force_table
        self.log = logger or RunLogger()
        self.opt = options or LoopOptions()
        self.approve, self.audit_client, self.injector = approve, audit_client, injector
        self.truth = truth
        self.state = TaskState(task, executor.grid)
        self.prev_img: Optional[np.ndarray] = None
        self.llm_cost = 0.0
        self.stop_reason: Optional[str] = None     # E-STOP / stop request from the console (written once, across threads)
        self.stop_kind = "estop"
        self._run_gate = threading.Event()         # set = running; cleared = paused (the operator clicked Pause in the software; stops before the next step starts)
        self._run_gate.set()

    def request_stop(self, reason: str = "operator E-STOP", kind: str = "estop") -> None:
        """Ask the loop to stop as soon as possible (callable from other threads). kind: estop (emergency stop, outcome estopped) or stop (normal operator stop, outcome stopped).

        Only sets a flag: the main loop checks it at the start of each step, after approval and after a skill returns; a running skill is ended by
        the backend's emergency stop (e.g. MujocoRobot.emergency_stop) being judged fatal by SafetyMonitor.
        """
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stop_kind = kind
        self._run_gate.set()                       # must be able to stop while paused too
        self.log.event("stop_requested", step=self.state.step, reason=reason, stop_kind=kind)

    def request_pause(self, reason: str = "operator pause") -> None:
        """Pause: stop after the current skill finishes and before the next decision (the robot stays at the survey pose). Callable from other threads."""
        if self._run_gate.is_set():
            self._run_gate.clear()
            self.log.event("pause_requested", step=self.state.step, reason=reason)

    def resume(self) -> None:
        if not self._run_gate.is_set():
            self.log.event("resumed", step=self.state.step)
            self._run_gate.set()

    @property
    def paused(self) -> bool:
        return not self._run_gate.is_set()

    def _wait_if_paused(self) -> None:
        if self._run_gate.is_set():
            return
        self.log.event("paused", step=self.state.step)
        while not self._run_gate.wait(0.2):
            if self.stop_reason is not None:
                return

    # ---------- Tools ----------
    def _skill_event(self, out: SkillOutcome, step: int) -> dict[str, Any]:
        trace_rel = self.log.trace(f"step{step:02d}_{out.skill}", out.trace) if self.opt.save_traces else None
        return self.log.event("skill", step=step, trace=trace_rel, **out.to_dict())

    def _inspect(self, step: int, zones: list[str], force: Optional[dict[str, Any]] = None,
                 closeups: Optional[dict[str, np.ndarray]] = None, label: str = "after") -> np.ndarray:
        out = self.ex.survey()
        self._skill_event(out, step)
        if not out.ok:
            raise InspectionAborted(out)
        img = self.src.rectified()
        rel = self.log.image(f"step{step:02d}_{label}", img)
        check = set(zones)
        new_spills: list[str] = []
        if self.opt.rescan_all:
            clean_now = [z for z in self.state.targets if self.state.status(z) is not None
                         and self.state.status(z).value == "clean" and z not in check]
            new_spills = self.ver.screen(img, clean_now) if clean_now else []
            check |= set(new_spills)
        order = [z for z in self.state.targets if z in check]
        vout = self.ver.inspect(order, after=img, before=self.prev_img, force=force, closeups=closeups)
        self.state.update_inspection(vout.results, vout.provenance)
        self.log.event("inspection", step=step, image=rel, zones=order, new_spills=new_spills,
                       results={z: r.model_dump(mode="json") for z, r in vout.results.items()},
                       provenance=vout.provenance, verifier=self.ver.name, llm=vout.llm,
                       cv_fraction={z: round(float(r.residue_fraction), 5) for z, r in vout.cv.items()},
                       vlm_raw=vout.raw or None,
                       truth_clean=({z: v for z, v in self.truth().items() if z in order} if self.truth else None))
        self.prev_img = img
        return img

    def _decide(self) -> tuple[Decision, dict[str, Any], str]:
        pd = self.policy.decide(self.state)
        metas = [pd.meta]
        if not self.policy.uses_guard:
            return pd.decision, {"policy": metas}, "not_applied"
        if not self.opt.guard:                                   # ablation: guardrail off, only record the actions it would have blocked
            g0 = guardrails.check(pd.decision, self.state, self.table)
            if not g0.ok:
                self.log.event("guard_would_reject", step=self.state.step, decision=decision_to_dict(pd.decision), reason=g0.reason)
            return pd.decision, {"policy": metas}, "ablated" if g0.ok else f"ablated_would_reject: {g0.reason}"
        g = guardrails.check(pd.decision, self.state, self.table)
        if g.ok:
            return pd.decision, {"policy": metas}, "ok"
        self.log.event("guard_reject", step=self.state.step, decision=decision_to_dict(pd.decision), reason=g.reason)
        pd2 = self.policy.decide(self.state, feedback=g.reason)
        metas.append(pd2.meta)
        g2 = guardrails.check(pd2.decision, self.state, self.table)
        if g2.ok:
            return pd2.decision, {"policy": metas}, f"ok_after_retry: {g.reason}"
        self.log.event("guard_reject", step=self.state.step, decision=decision_to_dict(pd2.decision), reason=g2.reason)
        fb = g2.fallback or g.fallback or guardrails.ask_human(f"planner proposed invalid actions twice: {g2.reason}")
        return fb, {"policy": metas}, f"fallback: {g2.reason}"

    def _cost(self, meta: dict[str, Any]) -> None:
        for m in meta.get("policy", []):
            for c in m.get("calls", []):
                self.llm_cost += float(c.get("cost_usd", 0.0) or 0.0)

    # ---------- Main flow ----------
    def run(self) -> dict[str, Any]:
        st, ex = self.state, self.ex
        t_robot0, t_wall0 = ex.robot.now(), time.time()
        self.log.event("task_start", task=self.task.model_dump(), targets=st.targets, policy=self.policy.name,
                       verifier=self.ver.name, source=getattr(self.src, "name", ""), robot=ex.robot.backend_name)
        problems = ex.startup_problems()
        if problems:
            self.log.event("startup_rejected", problems=problems)
            return self._finish("rejected", t_robot0, t_wall0, reason="; ".join(problems))
        try:
            self._inspect(0, list(st.targets), label="survey")
            status, reason = self._loop()
        except InspectionAborted as e:
            o = e.outcome
            status = "fault" if o.status == "fault" else "needs_human"
            reason = f"move to survey pose failed ({o.status}): {o.reason}"[:300]
            if status == "needs_human":
                st.human_requests.append({"step": st.step, "zones": [], "reason": reason})
            self.log.event("inspection_aborted", step=st.step, status=o.status, reason=o.reason)
        except Exception as e:
            if self.stop_reason is None:
                raise
            status, reason = "estopped", f"{self.stop_reason} ({type(e).__name__}: {e})"[:300]
        if self.stop_reason is not None and status != "estopped":
            status, reason = "estopped", self.stop_reason
        if status == "estopped" and self.stop_kind == "stop":
            status = "stopped"
        return self._finish(status, t_robot0, t_wall0, reason=reason)

    def _loop(self) -> tuple[str, str]:
        st, ex = self.state, self.ex
        status, reason = "done", ""
        while True:
            if self.stop_reason is not None:
                return "estopped", self.stop_reason
            self._wait_if_paused()
            if self.stop_reason is not None:
                return "estopped", self.stop_reason
            if self.injector:
                self.injector(st.step, st)
            st.step += 1
            decision, meta, guard = self._decide()
            self._cost(meta)
            self.log.event("decision", step=st.step, decision=decision_to_dict(decision), guard=guard,
                           policy=self.policy.name, meta=meta)
            if self.approve is not None and decision.action not in (Action.finish, Action.ask_human):
                if not self.approve(decision, st):
                    if self.stop_reason is not None:      # E-stop while waiting for approval: not counted as operator rejection
                        return "estopped", self.stop_reason
                    self.log.event("operator_rejected", step=st.step)
                    decision = guardrails.ask_human("operator rejected the proposed action", decision.zone_ids)
            if self.stop_reason is not None:
                return "estopped", self.stop_reason
            outcome: dict[str, Any] = {"status": "ok"}
            inspected: dict[str, str] = {}
            a = decision.action
            if a == Action.finish:
                st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome))
                if not self.policy.uses_guard or not self.opt.final_check:     # blind-wipe baseline / ablation: no forced final-check loop
                    self._inspect(st.step, list(st.targets), label="final")
                    break
                # The report must reflect the final all-zone check: if residue is found (e.g. a stain spilled midway), keep going
                self._inspect(st.step, list(st.targets), label="final")
                st.final_checks += 1
                if st.all_clean() or st.final_checks >= 3:
                    break
                self.log.event("final_check_found_residue", step=st.step, zones=st.dirty())
                continue
            merr = next((m.get("model_error") for m in meta.get("policy", []) if m.get("model_error")), None)
            if a == Action.ask_human and merr:                 # planning model unavailable: its own outcome, not a Staff check (there is no dirt for a person to look at)
                st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome))
                self.log.event("model_unavailable", step=st.step, error_kind=merr.get("kind"), message=merr.get("message"), fix=merr.get("fix"),
                               error_type=merr.get("error_type"), http_status=merr.get("http_status"))
                status, reason = "model_unavailable", f"{merr['message']} {merr['fix']}"
                break
            if a == Action.ask_human:
                st.human_requests.append({"step": st.step, "zones": decision.zone_ids, "reason": decision.reason})
                st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome))
                status, reason = "needs_human", decision.reason
                break
            if a == Action.inspect_closeup:
                z = decision.zone_ids[0]
                out = ex.closeup(z)
                self._skill_event(out, st.step)
                closeups = {}
                if out.ok:
                    try:
                        frame = self.src.frame("wrist")
                    except Exception as e:  # real robot has no wrist camera / stream failed: record honestly, re-check with the top-down image only
                        frame = None
                        self.log.event("closeup_unavailable", step=st.step, zone=z, error=f"{type(e).__name__}: {e}"[:200])
                    if frame is not None:
                        self.log.image(f"step{st.step:02d}_closeup_{z}", frame)
                        closeups[z] = frame
                st.closeup_done.add(z)
                self._inspect(st.step, [z], closeups=closeups)
                inspected = {z: st.status(z).value}
                outcome = {"status": out.status, "reason": out.reason}
            else:  # wipe_zone / spot_scrub
                out = ex.wipe(decision.zone_ids, float(decision.force_n),
                              decision.pattern.value if decision.pattern else "zigzag",
                              decision.passes, decision.bbox_norm if a == Action.spot_scrub else None)
                self._skill_event(out, st.step)
                outcome = {"status": out.status, "reason": out.reason, "force": out.force}
                if self.stop_reason is not None:   # during E-stop the skill exits with fault: record it as E-stop, not as a fault
                    st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome))
                    return "estopped", self.stop_reason
                if out.status == "rejected":
                    st.skill_rejections[decision_key(decision)] = out.reason
                if out.status in ("ok", "aborted", "no_contact"):
                    st.record_wipe(decision.zone_ids, decision.passes, out.force, float(decision.force_n),
                                   "spot" if a == Action.spot_scrub else (decision.pattern.value if decision.pattern else "zigzag"))
                if out.status == "fault":
                    st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome))
                    status, reason = "fault", out.reason
                    break
                if out.status in ("no_contact", "aborted"):
                    st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome))
                    st.human_requests.append({"step": st.step, "zones": decision.zone_ids, "reason": out.reason})
                    status, reason = "needs_human", out.reason
                    break
                if out.status == "ok":
                    zones = list(st.targets) if self.opt.recheck_all else list(decision.zone_ids)
                    self._inspect(st.step, zones, force=out.force)
                    inspected = {z: st.status(z).value for z in decision.zone_ids}
            st.history.append(StepRecord(st.step, decision_to_dict(decision), self.policy.name, guard, outcome, inspected))
            if st.step >= self.task.max_steps + 2:   # last safety net beyond the guardrail
                status, reason = "needs_human", "step budget exhausted"
                break
        return status, reason

    def _finish(self, status: str, t_robot0: float, t_wall0: float, reason: str = "") -> dict[str, Any]:
        st = self.state
        if status == "done" and not st.all_clean() and self.policy.uses_guard:
            status = "needs_human"
        summary = {
            "run_id": self.log.run_id, "status": status, "reason": reason, "policy": self.policy.name,
            "verifier": self.ver.name, "robot": self.ex.robot.backend_name, "task": self.task.model_dump(),
            "targets": st.targets, "steps": st.step,
            "initial_status": st.initial_status,
            "final_status": {z: (st.status(z).value if st.status(z) else None) for z in st.targets},
            "passes": st.passes, "human_requests": st.human_requests,
            "surface": self.task.surface, "last_force_n": st.last_force_n, "stain_at_first_wipe": st.wipe_stain,
            "robot_time_s": round(self.ex.robot.now() - t_robot0, 2), "wall_time_s": round(time.time() - t_wall0, 2),
            "decision_cost_usd": round(self.llm_cost, 6),
        }
        if self.opt.audit and self.audit_client is not None and self.policy.name == "nemotron" and status != "model_unavailable":   # the audit is Nemotron 3 Ultra summarising the Nemotron decision process: not called for the rule planner or a fixed-procedure planner
            summary["audit"] = self._audit(summary)
        self.log.write_json("summary.json", summary)
        self.log.event("task_end", **{k: summary[k] for k in ("status", "reason", "steps", "final_status")})
        return summary

    def _audit(self, summary: dict[str, Any]) -> dict[str, Any]:
        record = {"summary": summary, "history": [h.__dict__ for h in self.state.history]}
        messages = [{"role": "system", "content": (PROMPT_DIR / "audit.md").read_text(encoding="utf-8")},
                    {"role": "user", "content": json.dumps(record, ensure_ascii=False, default=str)[:60000]}]
        try:
            res = self.audit_client.chat("audit", messages, tag="audit")
            from ..common.llm_client import _THINK_RE

            text = _THINK_RE.sub("", res.text).strip()
            self.log.write_text("audit.md", text)
            return {"ok": True, "model": res.model, "request_id": res.request_id, "cost_usd": res.cost_usd}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
