"""Web console: replay API, approve / reject / emergency-stop flow of live simulation (all offline, no model calls)."""

import time

import pytest
from fastapi.testclient import TestClient

from cleanproof.common.jsonl import read_jsonl
from cleanproof.robot.io import SAFETY_MODE_ROBOT_EMERGENCY_STOP
from cleanproof.ui.app import create_app
from cleanproof.ui.session import LiveSession, ReplaySession


def _wait(pred, timeout=60.0, dt=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


def test_replay_endpoints(quick_run_dir):
    client = TestClient(create_app(ReplaySession(quick_run_dir)))
    assert "CleanProof Console" in client.get("/").text
    st = client.get("/api/status").json()
    assert st["mode"] == "replay" and st["live"] is None
    run = client.get("/api/run").json()
    assert run["status"] == "done" and run["is_sim"] and run["snapshots"][0]["label"] == "survey"
    img = client.get("/api/image/" + run["snapshots"][0]["image"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.get("/api/image/../summary.json").status_code in (404, 400)
    trace_rel = next(sk["trace"] for s in run["steps"] for sk in s["skills"] if sk.get("trace"))
    tr = client.get("/api/trace", params={"path": trace_rel, "max_points": 120}).json()
    assert len(tr["t"]) <= 120 and "f_n" in tr
    assert client.get("/api/frame").status_code == 404
    rep = client.get("/api/report")
    assert rep.status_code == 200 and "Cleaning Report" in rep.text
    assert client.post("/api/estop").status_code == 409        # replay has no live control


def test_live_approve_then_estop(tmp_path):
    s = LiveSession("S1_light", "rules", "cv", "topdown", run_root=tmp_path, require_approval=True)
    client = TestClient(create_app(s))
    assert client.get("/api/run").json()["status"] == "starting"
    assert client.post("/api/approve").json()["ok"] is False      # not started yet, nothing pending approval
    client.post("/api/start")
    assert _wait(lambda: client.get("/api/status").json()["live"]["pending"] is not None)
    st = client.get("/api/status").json()["live"]
    assert st["pending"]["decision"]["action"] in ("wipe_zone", "spot_scrub") and st["running"]
    assert client.get("/api/frame").status_code == 200           # live top-down view
    assert client.post("/api/approve").json()["ok"] is True
    # press emergency stop when the second decision awaiting approval appears
    assert _wait(lambda: (st2 := client.get("/api/status").json()["live"])["pending"] is not None or st2["finished"])
    assert not s.finished, "S1 should have at least two actions needing approval"
    r = client.post("/api/estop").json()
    assert r["ok"] is True and r["live"]["estopped"]
    assert _wait(lambda: s.finished, timeout=60)
    assert s.error is None, s.error
    summary = s.result["summary"]
    assert summary["status"] == "estopped" and "E-STOP" in summary["reason"]
    assert s.robot.get_safety_mode() == SAFETY_MODE_ROBOT_EMERGENCY_STOP
    kinds = [e["kind"] for e in read_jsonl(s.run_dir / "events.jsonl")]
    assert "operator_approved" in kinds and "stop_requested" in kinds and "operator_rejected" not in kinds
    assert kinds[-1] == "task_end"
    run = client.get("/api/run").json()
    assert run["status"] == "estopped" and client.get("/api/status").json()["live"]["final_status"] == "estopped"
    assert "estopped" in client.get("/api/report").text


def test_live_reject_hands_off_to_human(tmp_path):
    s = LiveSession("S1_light", "rules", "cv", "topdown", run_root=tmp_path, require_approval=True)
    client = TestClient(create_app(s))
    client.post("/api/start")
    assert _wait(lambda: s.pending() is not None)
    assert client.post("/api/reject").json()["ok"] is True
    assert _wait(lambda: s.finished, timeout=60)
    assert s.error is None, s.error
    assert s.result["summary"]["status"] == "needs_human"
    kinds = [e["kind"] for e in read_jsonl(s.run_dir / "events.jsonl")]
    assert "operator_rejected" in kinds


def test_live_estop_mid_wipe_without_approval(tmp_path):
    """When no approval is needed, an emergency stop during wiping: the skill exits as fatal, the loop is recorded as estopped, and the robot enters the e-stop state."""
    s = LiveSession("S1_light", "rules", "cv", "topdown", run_root=tmp_path, require_approval=False, pace=0.0)
    s.start()
    # emergency stop immediately after the first wipe decision is written (the skill is most likely executing then)
    assert _wait(lambda: s.run_dir is not None and any(e["kind"] == "decision" for e in read_jsonl(s.run_dir / "events.jsonl")))
    assert s.estop("test E-STOP")
    assert _wait(lambda: s.finished, timeout=60)
    assert s.error is None, s.error
    assert s.result["summary"]["status"] == "estopped"
    assert s.robot.get_safety_mode() == SAFETY_MODE_ROBOT_EMERGENCY_STOP
    # no more motion allowed after the emergency stop
    assert s.robot.move_j(s.robot.get_q(), 0.5, 0.5) is False


def test_live_pause_resume_then_stop(tmp_path):
    """Pause: stops before the next step once the current skill finishes; Resume continues; Stop wraps up normally (outcome stopped, robot does not enter e-stop state)."""
    s = LiveSession("S1_light", "rules", "cv", "topdown", run_root=tmp_path, require_approval=True)
    s.start()
    assert _wait(lambda: s.pending() is not None)
    assert s.pause() is True
    assert s.approve() is True                                  # the approved step still executes as usual, then stops after it finishes
    assert _wait(lambda: any(e["kind"] == "paused" for e in read_jsonl(s.run_dir / "events.jsonl")))
    assert s.status()["live"]["paused"] is True and s.pending() is None
    n_dec = sum(1 for e in read_jsonl(s.run_dir / "events.jsonl") if e["kind"] == "decision")
    time.sleep(0.5)
    assert sum(1 for e in read_jsonl(s.run_dir / "events.jsonl") if e["kind"] == "decision") == n_dec   # no new decisions while paused
    assert s.resume() is True
    assert _wait(lambda: s.pending() is not None or s.finished)
    if not s.finished:
        assert s.stop() is True
    assert _wait(lambda: s.finished, timeout=60)
    assert s.error is None, s.error
    if s.result["summary"]["status"] != "done":
        assert s.result["summary"]["status"] == "stopped"
        assert s.robot.get_safety_mode() != SAFETY_MODE_ROBOT_EMERGENCY_STOP
    kinds = [e["kind"] for e in read_jsonl(s.run_dir / "events.jsonl")]
    assert "pause_requested" in kinds and "resumed" in kinds
