"""Read runs/<run_id>/ and organize it into a structure shared by the report and the console.

Data comes only from files in the run directory: events.jsonl, summary.json, scenario.json, reference_meta.json,
audit.md, images/, traces/, plus runs/llm_calls.jsonl (to fill in tokens and cost by request_id).
No metric is derived here: a missing field is None.
A directory still in progress (no summary.json yet) can also be read, for the live console mode.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..agent.force_table import force_label
from ..common.config import load_yaml, runs_root
from ..common.jsonl import read_jsonl

STATE_NAMES = {1: "tare", 2: "approach", 3: "confirm", 4: "ramp", 5: "control", 6: "reapproach", 7: "release"}
WIPE_SKILLS = ("wipe_zone", "spot_scrub")


@dataclass
class ZoneView:
    status: Optional[str] = None          # clean / residue / uncertain / None (not checked)
    confidence: Optional[float] = None
    source: str = ""                      # vlm / fused / cv / oracle ... (verifier provenance)
    likely_cause: str = ""
    evidence: str = ""
    residue_bbox_norm: Optional[list[float]] = None
    passes: int = 0
    last_preset: Optional[str] = None
    truth_clean: Optional[bool] = None    # simulation only


@dataclass
class Snapshot:
    """All zone statuses after one inspection (the console drags the timeline by it)."""

    step: int
    label: str                            # survey / after / final / closeup
    image: Optional[str]
    zones: dict[str, ZoneView]


@dataclass
class Step:
    index: int
    decision: Optional[dict[str, Any]] = None
    guard: str = ""
    policy: str = ""
    calls: list[dict[str, Any]] = field(default_factory=list)
    guard_rejects: list[dict[str, Any]] = field(default_factory=list)
    skills: list[dict[str, Any]] = field(default_factory=list)
    inspections: list[dict[str, Any]] = field(default_factory=list)
    operator: list[dict[str, Any]] = field(default_factory=list)     # operator_approved / operator_rejected / stop_requested
    notes: list[str] = field(default_factory=list)                   # final_check_found_residue etc.

    @property
    def wipes(self) -> list[dict[str, Any]]:
        return [s for s in self.skills if s.get("skill") in WIPE_SKILLS]


@dataclass
class RunData:
    run_dir: Path
    run_id: str
    events: list[dict[str, Any]]
    summary: Optional[dict[str, Any]]
    judge: Optional[dict[str, Any]]
    reference_meta: Optional[dict[str, Any]]
    audit_text: Optional[str]
    task_start: Optional[dict[str, Any]]
    task_end: Optional[dict[str, Any]]
    steps: list[Step]
    snapshots: list[Snapshot]
    calls: list[dict[str, Any]]
    grid: dict[str, Any]

    # ---- Convenience properties ----
    @property
    def robot(self) -> str:
        if self.summary:
            return str(self.summary.get("robot", ""))
        return str((self.task_start or {}).get("robot", ""))

    @property
    def is_sim(self) -> bool:
        src = str((self.task_start or {}).get("source", ""))
        # When the robot computer runs a simulation backend (deployment self-test), the executor name is "remote:<backend>", and the report likewise says Simulation
        return self.robot.removeprefix("remote:") in ("mujoco", "rtde_fake", "ursim") or src.startswith("sim")

    @property
    def status(self) -> str:
        if self.summary:
            return str(self.summary.get("status", ""))
        if self.task_end:
            return str(self.task_end.get("status", ""))
        return "running"

    @property
    def targets(self) -> list[str]:
        if self.summary:
            return list(self.summary.get("targets", []))
        return list((self.task_start or {}).get("targets", []))

    @property
    def forbidden(self) -> list[str]:
        task = (self.summary or self.task_start or {}).get("task", {})
        return list(task.get("forbidden_zones", []))

    @property
    def task(self) -> dict[str, Any]:
        return dict((self.summary or self.task_start or {}).get("task", {}))

    @property
    def final_snapshot(self) -> Optional[Snapshot]:
        return self.snapshots[-1] if self.snapshots else None

    @property
    def initial_snapshot(self) -> Optional[Snapshot]:
        return self.snapshots[0] if self.snapshots else None

    def image_path(self, rel: Optional[str]) -> Optional[Path]:
        if not rel:
            return None
        p = self.run_dir / rel
        return p if p.is_file() else None

    def total_cost_usd(self) -> Optional[float]:
        """Sum of all known costs; returns None if any single one is unknown (no made-up numbers)."""
        total = 0.0
        for c in self.calls:
            if c.get("cost_usd") is None:
                return None
            total += float(c["cost_usd"])
        return round(total, 6)


# ---------------- Reading ----------------
def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None    # being written


def _grid_info() -> dict[str, Any]:
    cfg = load_yaml("zones.yaml")
    wa = cfg["work_area"]
    rows, cols = int(wa["rows"]), int(wa["cols"])
    ids = [[f"{chr(ord('A') + r)}{c + 1}" for c in range(cols)] for r in range(rows)]
    return {"rows": rows, "cols": cols, "ids": ids, "size_xy": list(wa["size_xy"]), "origin_xy": list(wa["origin_xy"])}


class _CallLog:
    """Index of runs/llm_calls.jsonl by request_id (metadata only, no request / response bodies)."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.index: dict[str, dict[str, Any]] = {}
        if path is not None and path.is_file():
            for rec in read_jsonl(path):
                rid = rec.get("request_id")
                if rid:
                    self.index[rid] = {k: v for k, v in rec.items() if k not in ("request", "response_text")}

    def enrich(self, call: dict[str, Any]) -> dict[str, Any]:
        rec = self.index.get(str(call.get("request_id", "")))
        if not rec:
            return call
        out = dict(call)
        for k in ("backend", "model", "latency_s", "prompt_tokens", "completion_tokens", "cost_usd", "schema_ok", "role"):
            if out.get(k) is None and rec.get(k) is not None:
                out[k] = rec[k]
        return out


def _call_record(step: Optional[int], role: str, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step, "role": role,
        "backend": raw.get("backend"), "model": raw.get("model"), "request_id": raw.get("request_id"),
        "latency_s": raw.get("latency_s"), "prompt_tokens": raw.get("prompt_tokens"),
        "completion_tokens": raw.get("completion_tokens"), "cost_usd": raw.get("cost_usd"),
        "schema_ok": raw.get("schema_ok"),
    }


def load_run(run_dir: str | Path, llm_log: Optional[Path] = runs_root() / "llm_calls.jsonl") -> RunData:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run dir not found: {run_dir}")
    events = read_jsonl(run_dir / "events.jsonl")
    summary = _read_json(run_dir / "summary.json")
    judge = _read_json(run_dir / "scenario.json")
    ref = _read_json(run_dir / "reference_meta.json")
    from ..common.config import use_site
    use_site(((ref or {}).get("scenario") or {}).get("site"))     # site geometry registered by the scene (zone grid)
    audit_path = run_dir / "audit.md"
    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.is_file() else None
    calllog = _CallLog(llm_log)

    task_start = next((e for e in events if e["kind"] == "task_start"), None)
    task_end = next((e for e in events if e["kind"] == "task_end"), None)
    targets = list((summary or task_start or {}).get("targets", []))

    steps: dict[int, Step] = {}
    snapshots: list[Snapshot] = []
    calls: list[dict[str, Any]] = []
    zones: dict[str, ZoneView] = {z: ZoneView() for z in targets}

    def step_of(e: dict[str, Any]) -> Step:
        i = int(e.get("step", 0) or 0)
        if i not in steps:
            steps[i] = Step(i)
        return steps[i]

    for e in events:
        k = e["kind"]
        if k == "decision":
            s = step_of(e)
            s.decision, s.guard, s.policy = e.get("decision"), str(e.get("guard", "")), str(e.get("policy", ""))
            for m in (e.get("meta") or {}).get("policy", []):
                for c in m.get("calls", []):
                    rec = calllog.enrich(_call_record(s.index, "decision", c))
                    s.calls.append(rec)
                    calls.append(rec)
        elif k == "guard_reject":
            step_of(e).guard_rejects.append({"decision": e.get("decision"), "reason": e.get("reason", "")})
        elif k == "skill":
            s = step_of(e)
            rec = {kk: vv for kk, vv in e.items() if kk not in ("kind", "ts")}
            s.skills.append(rec)
            if e.get("skill") in WIPE_SKILLS and e.get("status") in ("ok", "aborted", "no_contact"):
                for z in e.get("zone_ids", []):
                    if z in zones:
                        zones[z].passes += 1    # each skill call counts as 1 pass (a round trip with passes=2 counts as the same call)
        elif k == "inspection":
            s = step_of(e)
            rec = {kk: vv for kk, vv in e.items() if kk not in ("kind", "ts")}
            label = Path(str(e.get("image", ""))).stem.split("_", 1)[-1] if e.get("image") else "after"
            rec["label"] = label
            s.inspections.append(rec)
            for c in ((e.get("llm") or {}).get("calls") or []):
                crec = calllog.enrich(_call_record(s.index, "verifier", c))
                s.calls.append(crec)
                calls.append(crec)
            prov = e.get("provenance") or {}
            truth = e.get("truth_clean") or {}
            for z, r in (e.get("results") or {}).items():
                if z not in zones:
                    zones[z] = ZoneView()
                v = zones[z]
                v.status = r.get("status")
                v.confidence = r.get("confidence")
                v.source = str(prov.get(z, ""))
                v.likely_cause = str(r.get("likely_cause", ""))
                v.evidence = str(r.get("evidence", ""))
                v.residue_bbox_norm = r.get("residue_bbox_norm")
                if z in truth:
                    v.truth_clean = bool(truth[z])
            snapshots.append(Snapshot(s.index, label, e.get("image"), copy.deepcopy(zones)))
        elif k in ("operator_approved", "operator_rejected", "stop_requested", "estop"):
            step_of(e).operator.append({kk: vv for kk, vv in e.items() if kk != "ts"})
        elif k == "final_check_found_residue":
            step_of(e).notes.append(f"Final check found residue in {', '.join(e.get('zones', []))}; continuing.")
        elif k == "startup_rejected":
            step_of(e).notes.append("Startup rejected: " + "; ".join(e.get("problems", [])))

    # Target force recorded in the decision (last_preset for the zone view, shown as "10 N"; old run files are converted from the three levels)
    for s in steps.values():
        d = s.decision or {}
        if d.get("action") in WIPE_SKILLS and any(w.get("status") == "ok" for w in s.wipes):
            for z in d.get("zone_ids", []):
                if z in zones:
                    zones[z].last_preset = force_label(d) or None

    if summary and summary.get("audit", {}).get("ok"):
        a = summary["audit"]
        calls.append(calllog.enrich(_call_record(None, "audit", {"model": a.get("model"), "request_id": a.get("request_id"),
                                                                "cost_usd": a.get("cost_usd"), "backend": None})))

    return RunData(run_dir=run_dir, run_id=run_dir.name, events=events, summary=summary, judge=judge, reference_meta=ref,
                   audit_text=audit_text, task_start=task_start, task_end=task_end,
                   steps=[steps[i] for i in sorted(steps)], snapshots=snapshots, calls=calls, grid=_grid_info())


# ---------------- Trajectories ----------------
def load_trace(run_dir: str | Path, rel: str, max_points: int = 800) -> Optional[dict[str, Any]]:
    """Read a skill trajectory npz (t, x, y, z, f_n, f_ref, state), evenly subsampled to at most max_points."""
    path = Path(run_dir) / rel
    if not path.is_file():
        return None
    with np.load(path) as d:
        arrays = {k: np.asarray(d[k], dtype=float) for k in d.files}
    n = len(arrays.get("t", []))
    if n == 0:
        return None
    idx = np.arange(n) if n <= max_points else np.linspace(0, n - 1, max_points).round().astype(int)
    out: dict[str, Any] = {k: [round(float(v), 4) for v in a[idx]] for k, a in arrays.items()}
    out["n_samples"] = n
    out["t0"] = float(arrays["t"][0])
    out["duration_s"] = round(float(arrays["t"][-1] - arrays["t"][0]), 3)
    out["state_names"] = STATE_NAMES
    return out


# ---------------- Serialization (console API) ----------------
def snapshot_to_dict(s: Snapshot) -> dict[str, Any]:
    return {"step": s.step, "label": s.label, "image": s.image,
            "zones": {z: vars(v) for z, v in s.zones.items()}}


def step_to_dict(s: Step) -> dict[str, Any]:
    return {"index": s.index, "decision": s.decision, "guard": s.guard, "policy": s.policy, "calls": s.calls,
            "guard_rejects": s.guard_rejects, "skills": s.skills, "inspections": s.inspections,
            "operator": s.operator, "notes": s.notes}


def run_to_dict(r: RunData) -> dict[str, Any]:
    return {
        "run_id": r.run_id, "run_dir": str(r.run_dir), "status": r.status, "is_sim": r.is_sim, "robot": r.robot,
        "task": r.task, "targets": r.targets, "forbidden": r.forbidden, "task_start": r.task_start,
        "task_end": r.task_end, "summary": r.summary, "judge": r.judge,
        "scenario": (r.reference_meta or {}).get("scenario"), "audit_text": r.audit_text,
        "steps": [step_to_dict(s) for s in r.steps], "snapshots": [snapshot_to_dict(s) for s in r.snapshots],
        "calls": r.calls, "total_cost_usd": r.total_cost_usd(), "grid": r.grid, "n_events": len(r.events),
    }
