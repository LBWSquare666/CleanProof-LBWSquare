"""The software runs a job against a paired robot computer (cleanproof/ui/session.RemoteSession): the skill server uses the simulated backend (mujoco) and is started in this process as
a uvicorn (thread, free port), with the token in a temporary CLEANPROOF_CONFIG_DIR (0600). No real robot, no camera, no model calls.

Covers: a whole job to completion (approve, pause / resume), reject, normal stop (no /stop sent), E-stop during skill execution reaching the skill server, the hint when starting a job after an E-stop,
human review hand-over and the showcase view /frame/view for the home dining table site (Spot platform), the original text of the connection checks, hints for a wrong token / unreachable host, pace_robot, the config directory."""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

mujoco = pytest.importorskip("mujoco")

import httpx  # noqa: E402

from cleanproof.robot.server import build_service, create_app, pace_robot  # noqa: E402
from cleanproof.ui.session import RemoteSession  # noqa: E402

TOKEN = "unit-test-token"
SITE = "home_dining"
S1_TASK = {"instruction": "Clean zones B2, B3, C2 and C3.", "surface": "sealed_wood", "zones": ["B2", "B3", "C2", "C3"],
           "forbidden_zones": [], "max_passes_per_zone": 3, "max_steps": 20}
DINING_TASK = {"instruction": "Clean the table.", "surface": "sealed_wood", "zones": [], "forbidden_zones": [],
               "max_passes_per_zone": 3, "max_steps": 24}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    """Temporary user config directory (token 0600); never touches the real ~/.config/cleanproof."""
    from cleanproof.common.config import use_site

    d = tmp_path / "config"
    d.mkdir()
    (d / "robot_token").write_text(TOKEN + "\n", encoding="utf-8")
    os.chmod(d / "robot_token", 0o600)
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(d))
    monkeypatch.delenv("CLEANPROOF_SITE", raising=False)
    yield d
    use_site(None)                                    # the site registered by the simulated scenario (dining table) is not left to later tests


@contextmanager
def skill_server(tmp_path: Path, scenario: str, pace: float = 0.0):
    import uvicorn

    svc = build_service("mujoco", scenario=scenario, log_dir=tmp_path / f"ss-{scenario}", pace=pace)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield svc, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        th.join(timeout=10)
        svc.close()


def _hdr() -> dict[str, str]:
    return {"X-CleanProof-Token": TOKEN}


def _wait(pred, timeout: float = 60.0, step: float = 0.05) -> Any:
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = pred()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out")


def _events(run_dir) -> list[dict[str, Any]]:
    p = Path(run_dir) / "events.jsonl"
    if not p.is_file():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            break
    return out


def _kinds(run_dir) -> list[str]:
    return [e["kind"] for e in _events(run_dir)]


def _drive(rs: RemoteSession, timeout: float = 120.0) -> None:
    """Approve every pending plan until this job ends."""
    t0 = time.time()
    while rs.running:
        if rs.pending():
            rs.approve()
        time.sleep(0.05)
        assert time.time() - t0 < timeout, "job did not finish"
    rs.join(10)


def _log(rs: RemoteSession) -> list[dict[str, Any]]:
    """The page activity log (the same function as for the built-in simulation and recording replay)."""
    from cleanproof.ui.fleet import summarize_events

    evs = _events(rs.run_dir)
    start = next(e for e in evs if e["kind"] == "task_start")
    return summarize_events(evs, list(start["targets"]))["log"]


def _ss_log(svc) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in open(svc.log_dir / "skill_server.jsonl", encoding="utf-8")]


# ---------------------------------------------------------------- whole job
def test_full_job_with_approval_pause_resume(cfg, tmp_path):
    with skill_server(tmp_path, "S1_light") as (svc, url):
        rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", require_approval=True, robot="ur3e")
        assert rs.status()["live"]["robot_computer"] == {"skill_url": url, "camera_url": url, "backend": None, "mode": None,
                                                         "label": None, "simulated": None}
        rs.start()
        _wait(rs.pending)
        assert rs.pause() and rs.approve()                  # pause: stops after this step is done, before the next decision
        _wait(lambda: "paused" in _kinds(rs.run_dir))
        assert rs.status()["live"]["paused"] is True
        n = len(_kinds(rs.run_dir))
        time.sleep(0.8)
        assert len(_kinds(rs.run_dir)) == n and rs.pending() is None
        assert rs.resume()
        _drive(rs)

        st = rs.status()["live"]
        assert st["final_status"] == "done" and st["error"] is None and st["judge"] == {"targets_truly_clean": True}
        assert st["robot_computer"] == {"skill_url": url, "camera_url": url, "backend": "mujoco", "mode": "full", "label": "mujoco",
                                        "simulated": True}
        rd = Path(rs.run_dir)
        assert rd.parent == tmp_path / "runs" and rd.name.startswith(f"live-{SITE}-mujoco-full-")
        meta = json.loads((rd / "reference_meta.json").read_text(encoding="utf-8"))
        assert meta["site"] == SITE and meta["robot"]["platform"] == "ur3e" and meta["scenario"]["site"] == SITE
        summ = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
        assert summ["status"] == "done" and summ["robot_mode"] == "full" and summ["robot_label"] == "mujoco"
        kinds = _kinds(rd)
        assert kinds[0] == "task_start" and kinds[-1] == "task_end"
        assert {"pause_requested", "paused", "resumed", "operator_approved", "inspection", "decision", "skill"} <= set(kinds)
        insp = next(e for e in _events(rd) if e["kind"] == "inspection")
        assert set(insp["truth_clean"]) == set(insp["zones"])        # simulated backend: carries ground truth, like the built-in simulation
        log = _log(rs)                                                # the page log has the same shape as the built-in simulation: check, plan, guardrail, wipe (contact force), report
        assert {"check", "plan", "guard", "wipe", "end"} <= {x["kind"] for x in log}
        assert any(x["kind"] == "check" and "needs cleaning" in x["text"] for x in log)
        assert any(x["kind"] == "plan" and x["action"] == "wipe_zone" and x["zones"] for x in log)
        assert any(x["kind"] == "wipe" and re.match(r"wiped [A-E]\d \d+\.\d N", x["text"]) for x in log)
        assert log[-1] == {"who": "Report", "kind": "end", "ok": True, "text": "All zones clean · report with image proof", "meta": ""}

        jpg = rs.frame()                                              # UR3e (a built-in platform) has no showcase view: fixed camera
        assert jpg[:2] == b"\xff\xd8" and rs._has_view is False
        assert rs.frame() is jpg                                      # reused from the cache within 0.2 s
        assert httpx.get(url + "/frame/view", headers=_hdr()).status_code == 404
        h = httpx.get(url + "/health", headers=_hdr()).json()
        assert h["safety_mode"] == 1 and h["stopped"] is None and h["platform"] == "ur3e"
        assert "stop" not in {r["kind"] for r in _ss_log(svc)}

        n = len(_kinds(rd))                                           # pressing E-stop after it ends: only /stop is sent, the already finished job is not changed
        assert rs.estop("after the job") and rs.estop_delivered is True
        assert len(_kinds(rd)) == n and rs.status()["live"]["final_status"] == "done" and rs.estopped is False
        assert httpx.get(url + "/health", headers=_hdr()).json()["stopped"] == "after the job"


def test_reject_then_graceful_stop_does_not_stop_robot(cfg, tmp_path):
    with skill_server(tmp_path, "S1_light") as (svc, url):
        rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", require_approval=True, robot="ur3e")
        rs.start()
        _wait(rs.pending)
        assert rs.reject()
        rs.join(60)
        assert rs.status()["live"]["final_status"] == "needs_human"
        summ = json.loads((Path(rs.run_dir) / "summary.json").read_text(encoding="utf-8"))
        assert summ["human_requests"][0]["reason"] == "operator rejected the proposed action"
        assert "operator_rejected" in _kinds(rs.run_dir)

        rs2 = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", require_approval=True, robot="ur3e")
        rs2.start()
        _wait(rs2.pending)
        assert rs2.stop()                                             # normal stop: no /stop sent
        rs2.join(60)
        st = rs2.status()["live"]
        assert st["final_status"] == "stopped" and st["estopped"] is False and st["estop_delivered"] is None
        assert httpx.get(url + "/health", headers=_hdr()).json()["stopped"] is None
        assert "stop" not in {r["kind"] for r in _ss_log(svc)}
        assert not any(r["kind"] == "outcome" and r.get("skill") in ("wipe_zone", "spot_scrub") for r in _ss_log(svc))


def test_estop_reaches_skill_server_mid_skill(cfg, tmp_path):
    from cleanproof.ui.deploy_checks import check_skill_server

    with skill_server(tmp_path, "S1_light", pace=1.0) as (svc, url):   # about real time: wiping takes several seconds, the E-stop lands in the middle of a skill
        rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", require_approval=False, robot="ur3e")
        rs.start()
        _wait(lambda: (svc.busy or "").startswith("wipe_zone"), timeout=60, step=0.02)
        t0 = time.time()
        assert rs.estop("unit-test E-stop")
        assert time.time() - t0 < 3.5                                # does not wait for the skill to finish
        rs.join(30)
        assert not rs.running
        st = rs.status()["live"]
        assert st["final_status"] == "estopped" and st["estopped"] is True and st["estop_delivered"] is True
        h = httpx.get(url + "/health", headers=_hdr()).json()
        assert h["stopped"] == "unit-test E-stop" and h["safety_mode"] != 1
        stops = [r for r in _ss_log(svc) if r["kind"] == "stop"]
        assert len(stops) == 1 and stops[0]["reason"] == "unit-test E-stop" and stops[0]["busy"].startswith("wipe_zone")
        wipe = next(e for e in _events(rs.run_dir) if e["kind"] == "skill" and e["skill"] == "wipe_zone")
        assert wipe["status"] == "fault"
        assert _log(rs)[-1]["text"] == "Ended: estopped"

        rs2 = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", require_approval=False, robot="ur3e")
        rs2.start()
        rs2.join(30)
        err = rs2.status()["live"]["error"]
        assert err.startswith("The robot computer is stopped (E-stop). Reset it on site before the next job.") and "\n" not in err
        assert rs2.run_dir is None and rs2.status()["live"]["final_status"] is None
        assert "stopped (E-stop)" in check_skill_server(url)["detail"]
        from cleanproof.robot.remote import RemoteSkillExecutor
        ex = RemoteSkillExecutor(url)                                 # a rejected skill: the reason given to the operator is in English, the original text only goes in the events
        out = ex.survey()
        ex.close()
        assert out.status == "aborted" and out.reason == "the robot computer is stopped (E-stop); reset it on site (HTTP 423)"
        assert out.events[0]["http_status"] == 423 and "software stop" in out.events[0]["detail"]

        assert httpx.post(url + "/reset", json={"by": "tester"}, headers=_hdr()).status_code == 200   # after the on-site reset a job can be started again
        h = httpx.get(url + "/health", headers=_hdr()).json()
        assert h["stopped"] is None and h["safety_mode"] == 1


def test_dining_job_staff_check_and_display_view(cfg, tmp_path):
    """Home dining table site (Spot platform): coffee wiped clean, a mark that cannot be removed is handed to a person for review; the camera picture is a showcase render from the same view as the built-in simulation."""
    from cleanproof.robot.platforms import load_platforms
    from cleanproof.ui.deploy_checks import check_adapter, check_sim_camera, check_skill_server

    with skill_server(tmp_path, "H5_dining_permanent") as (svc, url):
        h = httpx.get(url + "/health", headers=_hdr()).json()
        assert h["safety_mode"] == 1 and h["platform"] == "spot_arm" and h["startup_problems"] == []
        sk = check_skill_server(url)                                  # the original text of the three Setup → Connect checks (these are the sentences in the video)
        assert sk["ok"] and sk["detail"] == "connected · token OK"
        label = load_platforms()["spot_arm"].label
        assert check_adapter(sk, "spot_arm")["detail"] == f"{label} · ready · software adapter"
        assert check_sim_camera(url)["detail"] == "1 camera stream live"
        ur3e = load_platforms()["ur3e"].label
        assert check_adapter(sk, "ur3e") == {"ok": False, "detail": f"{label} on the robot computer, not {ur3e}",
                                             "facts": {"platform": "ur3e", "label": ur3e, "via": "software adapter"}}
        bad = RemoteSession(url, url, SITE, DINING_TASK, run_root=tmp_path / "runs", robot="ur3e")      # the robot registered for the site differs: no job is started
        bad.start()
        bad.join(30)
        assert bad.status()["live"]["error"] == (f"The robot computer runs {label}, but this site is set up for {ur3e}. "
                                                 "Choose the matching site or robot.") and svc.n_requests == 0
        r = httpx.get(url + "/frame/view", headers=_hdr(), timeout=30.0)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and r.content[:2] == b"\xff\xd8"

        rs = RemoteSession(url, url, SITE, DINING_TASK, run_root=tmp_path / "runs", require_approval=True, robot="spot_arm")
        rs.start()
        _drive(rs)
        st = rs.status()["live"]
        assert st["final_status"] == "needs_human" and st["error"] is None and st["robot"] == "spot_arm"
        summ = json.loads((Path(rs.run_dir) / "summary.json").read_text(encoding="utf-8"))
        assert [r["zones"] for r in summ["human_requests"]] == [["B3"]]
        assert summ["final_status"]["C2"] == "clean"
        meta = json.loads((Path(rs.run_dir) / "reference_meta.json").read_text(encoding="utf-8"))
        assert meta["site"] == SITE and meta["robot"]["platform"] == "spot_arm" and meta["robot"]["platform_label"] == label
        log = _log(rs)
        assert {"check", "plan", "guard", "wipe", "end"} <= {x["kind"] for x in log}
        assert any(x["kind"] == "plan" and x["text"] == "Ask a person · B3" for x in log)   # request to hand over for human review
        assert any(x["kind"] == "plan" and x["text"].startswith("Re-plan · ") for x in log)  # re-wipe (increase force)
        assert log[-1]["text"] == "Ended: needs_human"
        jpg = rs.frame()
        assert jpg[:2] == b"\xff\xd8" and rs._has_view is True


def test_stop_or_estop_while_connecting_does_not_start(cfg, tmp_path, monkeypatch):
    """Click Stop during connection: no job is started and no /stop is sent; click E-stop: no job is started, and /stop goes straight to the skill server by address."""
    from cleanproof.agent import remote_job

    orig, entered = remote_job.connect_robot, threading.Event()

    def slow_connect(*a, **k):
        entered.set()
        time.sleep(0.8)
        return orig(*a, **k)

    monkeypatch.setattr(remote_job, "connect_robot", slow_connect)
    with skill_server(tmp_path, "S1_light") as (svc, url):
        for kind in ("stop", "estop"):
            entered.clear()
            rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", require_approval=False)
            rs.start()
            assert entered.wait(10)
            assert (rs.stop() if kind == "stop" else rs.estop("connect-time E-stop"))
            rs.join(30)
            st = rs.status()["live"]
            assert st["final_status"] == ("stopped" if kind == "stop" else "estopped") and st["error"] is None
            assert rs.run_dir is None and not (tmp_path / "runs").exists()
            h = httpx.get(url + "/health", headers=_hdr()).json()
            assert h["stopped"] == (None if kind == "stop" else "connect-time E-stop")
        assert rs.estop_delivered is True and svc.n_requests == 0


# ---------------------------------------------------------------- the force table is bound to this job's site
@pytest.fixture()
def force_sites(tmp_path, monkeypatch):
    """A temporary copy of configs/ with two sites added: zz_force_low caps the wood limit at 8 N (the starting force also lowered to <= 8 N), zz_force_high uses the default table (15 N).
    Does not touch configs/site in the repo."""
    import shutil

    from cleanproof.agent.force_table import reload_force_table
    from cleanproof.common import config

    cfgs = tmp_path / "configs"
    shutil.copytree(config.CONFIG_DIR, cfgs, ignore=shutil.ignore_patterns("*.png", "*.jpg", "*.npz"))
    (cfgs / "site" / "zz_force_low").mkdir()
    (cfgs / "site" / "zz_force_low" / "force_table.yaml").write_text(
        "surfaces: {sealed_wood: {max_n: 8.0}}\n"
        "forces: {dried_stain: {sealed_wood: 8}, ink_mark: {sealed_wood: 8}, grease: {sealed_wood: 8}}\n", encoding="utf-8")
    (cfgs / "site" / "zz_force_high").mkdir()
    monkeypatch.setattr(config, "CONFIG_DIR", cfgs)
    reload_force_table()
    yield cfgs
    reload_force_table()


def _forces_and_guard(rs) -> tuple[list[float], list[dict[str, Any]]]:
    evs = _events(rs.run_dir)
    forces = [float(e["decision"]["force_n"]) for e in evs if e["kind"] == "decision"
              and e["decision"]["action"] in ("wipe_zone", "spot_scrub") and "B3" in e["decision"]["zone_ids"]]
    return forces, [e for e in evs if e["kind"] == "guard_reject"]


@pytest.mark.parametrize("where", ["robot_computer", "builtin_sim"])
def test_force_limits_follow_the_jobs_site(cfg, force_sites, tmp_path, where):
    """This job is on zz_force_low (wood limit 8 N), the process's current site is another one (home_dining, registered by the simulated scenario), and after the job starts it switches to zz_force_high:
    planning and guardrails follow the zz_force_low table for the whole job - a mark that cannot be removed goes to a person once the force reaches 8 N, and no plan is bounced by the guardrail using another table."""
    from cleanproof.agent.force_table import load_force_table
    from cleanproof.common.config import current_site, use_site
    from cleanproof.sim.scenarios import load_scenario
    from cleanproof.ui.session import LiveSession

    def run(rs):
        rs.start()
        _wait(rs.pending)
        use_site("zz_force_high")                                  # someone elsewhere switches the site after the job starts (setup wizard, opening a report ...)
        assert load_force_table().cap("sealed_wood") == 15.0 and current_site() == "zz_force_high"
        _drive(rs)

    if where == "robot_computer":
        with skill_server(tmp_path, "H5_dining_permanent") as (svc, url):
            assert current_site() == "home_dining"                 # the robot computer (same process) switched the current site to the scenario's site
            rs = RemoteSession(url, url, "zz_force_low", DINING_TASK, run_root=tmp_path / "runs", require_approval=True,
                               robot="spot_arm")
            run(rs)
    else:
        spec = {**load_scenario("H5_dining_permanent"), "operator_site": "zz_force_low"}
        rs = LiveSession("H5_dining_permanent", "rules", "cv", "topdown", tmp_path / "runs", require_approval=True, spec=spec)
        assert rs.site == "zz_force_low"
        run(rs)
    st = rs.status()["live"]
    assert st["error"] is None and st["final_status"] == "needs_human", st
    forces, rejects = _forces_and_guard(rs)
    assert rejects == [] and max(forces) == 8.0 and forces[-1] == 8.0, forces   # B3 goes up to this site's limit and then goes to a person (the default table would go to 15 N)
    summ = json.loads((Path(rs.run_dir) / "summary.json").read_text(encoding="utf-8"))
    assert [r["zones"] for r in summ["human_requests"]] == [["B3"]] and summ["last_force_n"]["B3"] == 8.0
    meta = json.loads((Path(rs.run_dir) / "reference_meta.json").read_text(encoding="utf-8"))
    assert meta["site"] == "zz_force_low"


# ---------------------------------------------------------------- hints when a job is refused
def test_unreachable_and_bad_token_are_readable(cfg, tmp_path):
    port = _free_port()
    rs = RemoteSession(f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port}", SITE, S1_TASK, run_root=tmp_path / "runs")
    rs.start()
    rs.join(30)
    err = rs.status()["live"]["error"]
    assert err.startswith(f"Robot computer not reachable at http://127.0.0.1:{port}.") and "\n" not in err
    assert rs.frame() is None and rs.estop() is True and rs.estop_delivered is False

    with skill_server(tmp_path, "S1_light") as (svc, url):
        (cfg / "robot_token").write_text("not-the-token\n", encoding="utf-8")       # the client's token differs from the robot computer's
        rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs")
        rs.start()
        rs.join(30)
        err = rs.status()["live"]["error"]
        assert err.startswith("The robot computer rejected this computer's access token.") and "\n" not in err
        os.chmod(cfg / "robot_token", 0o644)                                        # token file permissions too open
        rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs")
        rs.start()
        rs.join(30)
        assert rs.status()["live"]["error"].startswith("The robot access token file must be private")
        os.chmod(cfg / "robot_token", 0o600)
        (cfg / "robot_token").write_text(TOKEN + "\n", encoding="utf-8")
        rs = RemoteSession(url, url, SITE, {**S1_TASK, "zones": ["B2"], "forbidden_zones": ["B2"]}, run_root=tmp_path / "runs")
        rs.start()
        rs.join(30)
        assert rs.status()["live"]["error"].startswith("None of the zones named in the instruction can be cleaned here")
        for task, msg in (({**S1_TASK, "forbidden_zones": ["Z9"]}, "No-wipe zone Z9 is not on the robot computer's surface (its grid is "),
                          ({**S1_TASK, "zones": ["B2", "Q7", "Q8"]}, "Zones Q7, Q8 are not on the robot computer's surface")):
            rs = RemoteSession(url, url, SITE, task, run_root=tmp_path / "runs")      # a cell that is not among the robot computer's zones: refused before the job starts
            rs.start()
            rs.join(30)
            assert rs.status()["live"]["error"].startswith(msg), rs.status()["live"]["error"]
        assert svc.n_requests == 0                                                  # a refused job sends no skill


def test_uncalibrated_robot_computer_is_not_ready(cfg, tmp_path, monkeypatch):
    """Real-robot backend (ur3e, shadow), site not calibrated: the connection check explains why, and the job is refused without sending any skill. The real-robot factory is replaced by fake RTDE."""
    import uvicorn

    from cleanproof.robot.backends import rtde_backend
    from cleanproof.robot.backends.fake_rtde import FakeRTDE
    from cleanproof.robot.platforms import load_platforms
    from cleanproof.ui.deploy_checks import check_adapter, check_skill_server

    orig = rtde_backend.make_rtde_robot
    monkeypatch.setattr(rtde_backend, "make_rtde_robot",
                        lambda backend, mode="full", rcfg=None, host=None, z_surface=None, arming=None, **kw:
                        orig("rtde_fake", mode, rcfg=rcfg, z_surface=z_surface, fake=FakeRTDE()))
    monkeypatch.setenv("CLEANPROOF_SITE", "zz_uncalibrated_test_site")             # a nonexistent site directory = not calibrated
    svc = build_service("ur3e", mode="shadow", log_dir=tmp_path / "ss")
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(svc), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    _wait(lambda: server.started, timeout=10)
    url = f"http://127.0.0.1:{port}"
    try:
        sk = check_skill_server(url)
        assert not sk["ok"] and sk["detail"] == "connected · token OK · site not calibrated · calibrate it in the Site step"
        assert check_adapter(sk, "ur3e")["detail"] == f"{load_platforms()['ur3e'].label} · not ready · site not calibrated"
        rs = RemoteSession(url, url, SITE, S1_TASK, run_root=tmp_path / "runs", robot="ur3e")
        rs.start()
        rs.join(30)
        err = rs.status()["live"]["error"]
        assert err.startswith("This site is not calibrated on the robot computer yet; calibrate it in Setup (Site step), "
                              "then start the job.") and "\n" not in err
        assert rs.run_dir is None and svc.n_requests == 0
    finally:
        server.should_exit = True
        th.join(timeout=10)
        svc.close()


# ---------------------------------------------------------------- small items
def test_pace_robot_paces_and_does_not_catch_up():
    class FakeRobot:
        def __init__(self):
            self.t = 0.0

        def now(self) -> float:
            return self.t

        def wait_tick(self) -> None:
            self.t += 0.004

    r = FakeRobot()
    pace_robot(r, 2.0)
    t0 = time.time()
    for _ in range(100):                                           # 0.4 s of simulated time, 2x real time → about 0.2 s
        r.wait_tick()
    assert 0.17 < time.time() - t0 < 0.6
    time.sleep(0.4)                                                # idle time between skills is not made up: afterwards it still runs at 2x real time
    t0 = time.time()
    for _ in range(50):
        r.wait_tick()
    assert time.time() - t0 > 0.07


def test_builtin_sim_throttle_realigns_after_waits():
    """Throttling of the built-in simulation (LiveSession._throttle): after waiting for approval / pause it does not catch up in one burst, and still follows pace times real time."""
    from cleanproof.ui.session import LiveSession

    class FakeRobot:
        def __init__(self):
            self.t = 0.0

        def now(self) -> float:
            return self.t

        def wait_tick(self) -> None:
            self.t += 0.004

    r = FakeRobot()
    LiveSession("S1_light", pace=2.0)._throttle(r)
    time.sleep(0.4)
    t0 = time.time()
    for _ in range(50):                                           # 0.2 s of simulated time, 2x real time → about 0.1 s, not 0
        r.wait_tick()
    assert time.time() - t0 > 0.07


def test_builtin_sim_job_records_operator_site(tmp_path):
    """One job of the built-in simulation writes the site chosen by the operator into reference_meta.json (several sites can share one simulated scenario)."""
    from types import SimpleNamespace

    from cleanproof.agent.runlog import RunLogger
    from cleanproof.ui.session import LiveSession

    log = RunLogger(tmp_path, run_id="sim-H5_dining_permanent-rules-cv-test")
    log.write_json("reference_meta.json", {"source": "sim-topdown", "scenario": {"site": "home_dining"}, "robot": {"platform": "ur3e"}})
    for ls in (LiveSession("H5_dining_permanent", site="office_stairs"),
               LiveSession("H5_dining_permanent", spec={"task": {}, "operator_site": "office_stairs"})):
        ls._on_runner(SimpleNamespace(log=log, src=None), SimpleNamespace(robot=None, meta={"platform": "ur3e"}))
        meta = json.loads((log.dir / "reference_meta.json").read_text(encoding="utf-8"))
        assert meta["site"] == "office_stairs" and meta["scenario"] == {"site": "home_dining"} and meta["robot"]["platform"] == "ur3e"


def test_refusal_and_startup_texts_are_english():
    from cleanproof.agent.remote_job import drift_problem, readiness_error
    from cleanproof.robot.remote import refusal_reason, startup_problem_text

    assert refusal_reason(423, "software stop active: x; reset after on-site confirmation", "survey") == "the robot computer is stopped (E-stop); reset it on site"
    assert refusal_reason(423, "unreset fault exists: y", "wipe_zone").startswith("the robot computer has an unreset fault")
    assert refusal_reason(423, "motion authorization expired at 12:00:00, please re-arm", "survey").startswith("the robot is not armed")
    assert refusal_reason(422, "unknown forbidden zone id ['Q9']", "survey") == "the robot computer does not know some of the no-wipe zones"
    assert refusal_reason(409, "busy executing survey", "survey") == "the robot computer is busy with another skill"
    assert refusal_reason(500, "RuntimeError: boom", "wipe_zone").startswith("the robot computer failed while running wipe_zone")
    assert startup_problem_text("teach-pendant speed slider is 0.50, must be 100%") == "the teach-pendant speed slider is not at 100%"
    msg = readiness_error({"startup_problems": ["teach-pendant speed slider is 0.50, must be 100%"]})
    assert msg == "Robot start-up check failed: the teach-pendant speed slider is not at 100%."
    assert all(ord(c) < 0x3000 for c in msg)
    assert drift_problem({"max_px": 0.4, "missing": []}, 6.0) is None
    assert "a marker is missing" in drift_problem({"max_px": 0.4, "missing": ["3"]}, 6.0)
    assert drift_problem({"max_px": None, "missing": ["0", "1", "2", "3"]}, 6.0) and drift_problem({"max_px": 7.0, "missing": []}, 6.0)


def test_config_dir_env_is_read_at_call_time(tmp_path, monkeypatch):
    from cleanproof.common import config
    from cleanproof.robot.server import read_token, token_path

    monkeypatch.delenv("CLEANPROOF_CONFIG_DIR", raising=False)
    assert token_path() == Path("~/.config/cleanproof/robot_token").expanduser()
    assert config.secrets_files()[0] == Path("~/.config/cleanproof/secrets.env").expanduser()
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(tmp_path))
    assert token_path() == tmp_path / "robot_token" and read_token() is None
    (tmp_path / "robot_token").write_text("abc\n", encoding="utf-8")
    os.chmod(tmp_path / "robot_token", 0o600)
    assert read_token() == "abc"
    assert config.secrets_files() == (tmp_path / "secrets.env", config.REPO_ROOT / ".env")
    (tmp_path / "secrets.env").write_text("CP_UNIT_TEST_DUMMY=1\n", encoding="utf-8")
    monkeypatch.delenv("CP_UNIT_TEST_DUMMY", raising=False)
    assert config._load_secrets((tmp_path / "secrets.env",)) == (str(tmp_path / "secrets.env"),)
    assert os.environ["CP_UNIT_TEST_DUMMY"] == "1"
    assert callable(config.load_secrets.cache_clear)


def _flaky_executor(fail_times, calls):
    """Skip the constructor (it connects to a service): build a client over a transport that fails fail_times times first."""
    import httpx

    from cleanproof.robot import remote as R

    def handler(req):
        calls.append(req.url.path)
        if len(calls) <= fail_times:
            raise httpx.ConnectError("boom")
        if req.url.path == "/health":
            return httpx.Response(200, json={"backend": "mujoco"})
        return httpx.Response(200, json={"skill": "x", "status": "done", "reason": ""})

    ex = R.RemoteSkillExecutor.__new__(R.RemoteSkillExecutor)
    ex.http = httpx.Client(base_url="http://t", transport=httpx.MockTransport(handler))
    ex.forbidden, ex.fault, ex.cfg = set(), None, {"skills": {"zone_timeout_s": 5}}
    return ex


def test_idempotent_requests_retry_but_motion_skills_do_not(monkeypatch):
    """: survey / close-up / health are retried briefly when the connection drops; wiping is never retried."""
    from cleanproof.robot import remote as R
    monkeypatch.setattr(R, "RETRY_BACKOFF_S", 0.0)
    calls: list[str] = []
    assert _flaky_executor(2, calls).survey().status == "done" and len(calls) == 3
    calls = []
    assert _flaky_executor(3, calls).survey().status == "aborted" and len(calls) == 3      # gives up only after the retries are used up
    calls = []
    assert _flaky_executor(1, calls).health()["backend"] == "mujoco" and len(calls) == 2
    calls = []
    out = _flaky_executor(1, calls).wipe(["A1"], 5.0, "zigzag")
    assert out.status == "aborted" and len(calls) == 1                                     # wiping is sent only once
