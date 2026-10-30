"""Recorded runs: reads a recording directory (frames.jsonl, meta.json) plus its run events and summarizes them for the operator software.

`Take` gives the UI state of a recorded job at any time t (the events written before that frame); `summarize_events` and `report_row` condense a run.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any, Optional

from ..common.config import REPO_ROOT, runs_root
from ..report.provenance import FALLBACK_NOTE, check_kind, check_name, planner_name



def _find_run_dir(rec_dir: Path, run_dir: str) -> Path:
    """The run_dir in meta.json may be an absolute path on another machine (the cluster): look for it nearby by directory name."""
    p = Path(run_dir)
    if (p / "events.jsonl").is_file():
        return p
    for root in (rec_dir.parent.parent.parent, rec_dir.parent.parent, runs_root(), REPO_ROOT / "runs" / "platforms" / "runs"):
        q = root / p.name
        if (q / "events.jsonl").is_file():
            return q
    raise FileNotFoundError(f"run directory {p.name} not found (recording {rec_dir})")


def _short_reason(reason: str) -> str:
    import re
    m = re.search(r"must use (\d+(?:\.\d+)?) N", reason or "")
    if m:
        return f"next wipe must be {float(m.group(1)):g} N"
    m = re.search(r"covers (\d+)% of the zone; spot_scrub", reason or "")
    if m:
        return f"residue {m.group(1)}% too large for a spot scrub"
    m = re.search(r"starts at (\d+(?:\.\d+)?) N", reason or "")
    if m:
        return f"first wipe starts at {float(m.group(1)):g} N"
    return (reason or "")[:60]


# scenario → site name shown in the console (multi-platform segment, cleanproof/robot/env_scenes.py)
SCENE_OF = {"W1_wall_marker": "Office desk", "G1_window_glass": "Car wash · van window", "R1_rail_dried": "Footbridge handrail", "L1_line_steel": "Factory line",
            "S2_dried": "Home dining table", "F1_force_wood": "Home dining table", "S3_permanent": "Home dining table",
            "T1_surface_wood": "Home dining table", "T2_surface_stone": "Home dining table", "T3_surface_steel": "Home dining table"}
SURFACE_LABEL = {"glass": "glass", "whiteboard": "white laminate", "sealed_wood": "sealed wood", "stainless_steel": "stainless steel", "rough_stone": "stone"}


def _wipe_cut(e: dict[str, Any], evs: list[dict[str, Any]]) -> str:
    """A wipe with no contact segment (e-stop / stop interruption, skill rejected): honestly record that it did not wipe; if this job ended by e-stop, write E-stop."""
    end = next((x for x in evs if x.get("kind") == "task_end"), None)
    if end is not None and end.get("status") == "estopped":
        return "wipe stopped (E-stop)"
    return f"wipe {str(e.get('status') or 'not completed').replace('_', ' ')}"


def summarize_events(evs: list[dict[str, Any]], targets: list[str], audited: bool = False) -> dict[str, Any]:
    """Events → task state (zone states, activity log, wipe / check counts, model cost, outcome). Shared by recording replay (Take.state) and live jobs (operator._live_state)."""
    zones: dict[str, dict[str, Any]] = {z: {"status": None} for z in targets}
    log: list[dict[str, Any]] = []
    wipes = checks = 0
    cost = 0.0
    target_n: Optional[float] = None
    status = "running"
    wiped: set[str] = set()
    last_image: Optional[str] = None
    # who a plan is sent back to when the guardrails reject it: with Nemotron planning (the jobs in the recordings) say Nemotron, with rule-based planning say the planner; without task_start assume Nemotron (old recordings)
    policy = next((str(e.get("policy") or "") for e in evs if e.get("kind") == "task_start"), "nemotron") or "nemotron"
    back_to = "plan sent back to Nemotron" if policy == "nemotron" else "plan sent back to the planner"
    for e in evs:
        k = e["kind"]
        if k == "inspection":
            checks += 1
            last_image = str(e.get("image") or "") or last_image
            res = e.get("results") or {}
            cvf = e.get("cv_fraction") or {}
            for z, r in res.items():
                zones[z] = {"status": r.get("status"), "residue_pct": round(100 * float(cvf[z])) if z in cvf and cvf[z] is not None else None,
                            "stain_type": r.get("stain_type")}                 # live mode force-library screening looks it up by this (type not recognizable = unknown)
            dirty = [z for z, r in res.items() if r.get("status") == "residue"]
            first = checks == 1
            calls = (e.get("llm") or {}).get("calls") or []
            lat = sum(float(c.get("latency_s") or 0) for c in calls)
            kd = check_kind(e.get("provenance"), e.get("verifier") or "", len(calls))        # written from the actual source in the event, not guessed from the configuration
            who = check_name(kd) if kd != "mixed" else "Cosmos 3"
            if dirty:                                    # the first inspection says "needs cleaning", only a re-check says "still dirty"; the residue ratio comes from the pixel check and goes into the metadata
                txt = " · ".join(f"{z} {'needs cleaning' if first else 'still dirty'}" for z in dirty)
            else:
                txt = "All checked zones clean"
            log.append({"who": who, "kind": "check", "text": txt, "ok": not dirty,
                        "meta": (f"{lat:.1f} s · local NVIDIA GPU" if kd in ("cosmos", "mixed") else FALLBACK_NOTE if kd == "fallback" else "classical vision")
                        })                                     # the residue ratio from the pixel check is kept out of the log (it would suggest the verdict was not made by Cosmos)
        elif k == "decision":
            d = e.get("decision") or {}
            calls = [c for p in ((e.get("meta") or {}).get("policy") or []) for c in (p.get("calls") or [])]
            c_usd = sum(float(c.get("cost_usd") or 0) for c in calls)
            lat = sum(float(c.get("latency_s") or 0) for c in calls)
            cost += c_usd
            act = d.get("action")
            if act in ("wipe_zone", "spot_scrub"):
                target_n = float(d.get("force_n") or 0) or target_n
                zs_ = list(d.get("zone_ids") or [])
                again = any(z in wiped for z in zs_)            # only a new plan for already wiped zones counts as re-planning
                txt = f"{'Re-plan · ' if again else ''}{act.replace('_', ' ')} {' + '.join(zs_)} · {float(d.get('force_n') or 0):g} N"
            elif act == "finish":
                txt = "Finish · every zone clean"
            elif act == "ask_human":
                txt = f"Ask a person · {' + '.join(d.get('zone_ids') or [])}"
            else:
                txt = str(act)
            merr = next((m.get("model_error") for m in ((e.get("meta") or {}).get("policy") or []) if m.get("model_error")), None)
            if merr:                                           # model unavailable: its own row, not counted as "ask a person to check"
                log.append({"who": "Model unavailable", "kind": "error", "ok": False, "text": str(merr.get("message") or "The planning model is not available."),
                            "meta": str(merr.get("fix") or "")})
                continue
            log.append({"who": "Nemotron 3 Super" if calls else planner_name(policy) if policy != "nemotron" else "Planner", "kind": "plan", "text": txt,
                        "action": str(act), "zones": list(d.get("zone_ids") or []),       # the UI plays the force-library screening only for wipe plans, looked up by the cell in the plan
                        "meta": (f"{lat:.1f} s · ${c_usd:.4f} · Nebius Token Factory" if calls else "rules")})
            if e.get("guard") == "ok" and act in ("wipe_zone", "spot_scrub"):
                log.append({"who": "Guardrail", "kind": "guard", "text": f"approved · {float(d.get('force_n') or 0):g} N within limits", "ok": True,
                            "meta": "force database · surface limit"})
        elif k == "guard_reject":
            log.append({"who": "Guardrail", "kind": "guard", "text": "blocked · " + _short_reason(str(e.get("reason", ""))), "ok": False,
                        "meta": back_to})
        elif k == "skill" and e.get("skill") in ("wipe_zone", "spot_scrub"):
            wipes += 1
            fz = e.get("force") or {}
            wiped.update(fz.keys())
            parts = [f"{z} {v.get('mean_normal_n', 0):.1f} N" for z, v in fz.items()]
            cov = min((float(v.get("coverage_ratio", 0)) for v in fz.values()), default=0.0)
            log.append({"who": "Skill server", "kind": "wipe", "ok": e.get("status") == "ok",
                        # no contact segment (e-stop / stop interruption, skill rejected): honestly record that it did not wipe, rather than an empty "wiped"
                        "text": "wiped " + " · ".join(parts) if parts else _wipe_cut(e, evs),
                        "meta": f"force control · coverage {100 * cov:.0f}%" if fz else "force control · coverage unknown"})   # no contact segment (interrupted by an E-stop, etc.): coverage is unknown, not 0%
        elif k == "task_end":
            status = str(e.get("status", "done"))
            fs = e.get("final_status") or {}
            for z, s in fs.items():
                zones.setdefault(z, {})["status"] = s
            log.append({"who": "Report", "kind": "end", "ok": status == "done",
                        "text": "All zones clean · report with image proof" if status == "done" else f"Ended: {status}",
                        "meta": "audited by Nemotron 3 Ultra" if audited else ""})
    return {"zones": zones, "log": log, "status": status, "target_n": target_n, "wipes": wipes, "checks": checks,
            "cost_usd": round(cost, 4), "last_image": last_image}


class Take:
    """One run of one platform (recording + events)."""

    def __init__(self, key: str, rec_dir: str | Path):
        self.key = key
        self.rec_dir = Path(rec_dir)
        self.frames = [json.loads(ln) for ln in (self.rec_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.meta = json.loads((self.rec_dir / "meta.json").read_text(encoding="utf-8"))
        self.run_dir = _find_run_dir(self.rec_dir, self.meta["run_dir"])
        self.events = [json.loads(ln) for ln in (self.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.ts = [float(f["t_sim"]) for f in self.frames]
        start = next((e for e in self.events if e["kind"] == "task_start"), {})
        self.task = start.get("task") or {}
        self.targets = list(start.get("targets") or self.task.get("zones") or [])

    def event_times(self) -> list[dict[str, Any]]:
        """Simulation time at which each event first appears in the recording (the frame of frames.jsonl whose n_events passes that event index)."""
        out, j = [], 0
        for k, e in enumerate(self.events):
            while j < len(self.frames) and int(self.frames[j].get("n_events", 0)) <= k:
                j += 1
            t = self.ts[min(j, len(self.ts) - 1)]
            d = e.get("decision") or {}
            out.append({"k": k, "kind": e["kind"], "t": t, "action": d.get("action"), "zones": d.get("zone_ids"), "force_n": d.get("force_n"),
                        "guard": e.get("guard"), "skill": e.get("skill")})
        return out

    @property
    def t0(self) -> float:
        return self.ts[0]

    @property
    def t1(self) -> float:
        return self.ts[-1]

    def index(self, t: float) -> int:
        return max(0, min(len(self.frames) - 1, bisect_right(self.ts, t) - 1))

    def frame_path(self, t: float) -> Path:
        """Prefer the re-rendered frame with the white-background theme (cleanproof/robot/platform_render.py, state taken from the same recording), otherwise use the original recorded frame."""
        i = self.index(t)
        hq = self.rec_dir / "frames_hq" / f"demo_{int(self.frames[i]['i']):06d}.jpg"
        return hq if hq.is_file() else self.rec_dir / self.frames[i]["demo"]

    def state(self, t: float, nev: Optional[int] = None) -> dict[str, Any]:
        i = self.index(t)
        fr = self.frames[i]
        # task_end (final check, audit) is written only after the last recorded frame: when replay reaches the end of the recording, count all events
        evs = self.events if i >= len(self.frames) - 1 else self.events[: int(fr.get("n_events", 0))]
        if nev is not None:                                  # several events written at the same simulation time (check, decision, guardrail) are revealed one by one following the narration
            evs = evs[: max(0, int(nev))]
        sm = summarize_events(evs, self.targets, (self.run_dir / "audit.md").is_file())
        zones, log, status, target_n = sm["zones"], sm["log"], sm["status"], sm["target_n"]
        wipes, checks, cost, last_image = sm["wipes"], sm["checks"], sm["cost_usd"], sm["last_image"]
        trace = [(round(self.ts[j] - fr["t_sim"], 3), round(float(self.frames[j].get("f_n") or 0.0), 2))
                 for j in range(max(0, i - 400), i + 1, 4)]
        return {"platform": self.key, "t": round(float(fr["t_sim"]), 3), "t0": self.t0, "t1": self.t1, "status": status,
                "zones": zones, "log": log, "f_n": round(float(fr.get("f_n") or 0.0), 2), "in_contact": bool(fr.get("in_contact")),
                "target_n": target_n, "trace": trace, "wipes": wipes, "checks": checks, "cost_usd": round(cost, 4),
                "task": {"instruction": self.task.get("instruction"), "surface": self.task.get("surface")},
                "run_id": self.run_dir.name, "last_image": last_image}


def report_row(run_dir: Path, robot_label: str) -> dict[str, Any]:
    """One row of the report list (all taken from events.jsonl in the run directory): end time, site, robot, zone count, outcome, wipe count, decision cost."""
    from datetime import datetime, timezone
    evs = [json.loads(ln) for ln in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    start = next((e for e in evs if e["kind"] == "task_start"), {})
    end = next((e for e in reversed(evs) if e["kind"] == "task_end"), {})
    scen = run_dir.name.split("-")[1]                        # sim-<scenario>-<policy>-...
    zones = list(start.get("targets") or (start.get("task") or {}).get("zones") or [])
    ts = end.get("ts") or start.get("ts")
    when = datetime.fromisoformat(ts).astimezone(timezone.utc) if ts else None
    wipes = sum(1 for e in evs if e["kind"] == "skill" and e.get("skill") in ("wipe_zone", "spot_scrub"))
    cost = sum(float(c.get("cost_usd") or 0) for e in evs if e["kind"] == "decision"
               for p_ in ((e.get("meta") or {}).get("policy") or []) for c in (p_.get("calls") or []))
    surf = str((start.get("task") or {}).get("surface") or "")
    return {"run_id": run_dir.name, "site": SCENE_OF.get(scen, "Site"), "surface": SURFACE_LABEL.get(surf, surf.replace("_", " ")),
            "robot": robot_label, "zones": len(zones), "status": str(end.get("status", "running")), "wipes": wipes,
            "cost_usd": round(cost, 4), "ended_utc": when.strftime("%Y-%m-%d %H:%M") if when else "",
            "staff_zones": [z for z, v in (end.get("final_status") or {}).items() if v not in ("clean", None)]}
