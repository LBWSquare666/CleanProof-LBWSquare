"""Offline tests for the real-robot deployment chain: skill server (simulation backend), HTTP client, camera service (replay), pure functions for site calibration, site config overrides,
RTDE software stop. No robot connected, no camera opened."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

mujoco = pytest.importorskip("mujoco")

from fastapi.testclient import TestClient  # noqa: E402

from cleanproof.robot.commissioning import ArmingToken  # noqa: E402
from cleanproof.robot.server import SkillRequest, SkillService, build_service, create_app  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture()
def svc(tmp_path):
    s = build_service("mujoco", scenario="S2_dried", log_dir=tmp_path / "ss")
    yield s
    s.close()


def test_request_validation(svc):
    c = TestClient(create_app(svc))
    assert c.post("/skill", json={"skill": "moveJ"}).status_code == 422                       # not in the whitelist
    assert c.post("/skill", json={"skill": "survey", "speed": 1}).status_code == 422          # extra field
    bad = {"skill": "spot_scrub", "zone_ids": ["C3"], "force_n": 5, "bbox_norm": [0.5, 0.1, 0.2, 0.9]}
    assert c.post("/skill", json=bad).status_code == 422                                      # bbox is inverted
    r = c.post("/skill", json={"skill": "wipe_zone", "zone_ids": ["Z9"], "force_n": 5})
    assert r.status_code == 200 and r.json()["status"] == "rejected"
    r = c.post("/skill", json={"skill": "wipe_zone", "zone_ids": ["C3"]})
    assert r.json()["status"] == "rejected" and "target force" in r.json()["reason"]
    for f in (4.0, 15.5, 20.0):                                                               # outside force_limits 5-15 N
        r = c.post("/skill", json={"skill": "wipe_zone", "zone_ids": ["C3"], "force_n": f, "pattern": "zigzag"}).json()
        assert r["status"] == "rejected" and "allowed range" in r["reason"]


def test_real_backend_only_runs_validated_force():
    """ur3e backend: anything above force_limits.real_validated_max_n (only 5 N has been validated) is always rejected; the simulation backend is not subject to this limit."""
    from types import SimpleNamespace

    from cleanproof.robot.skills import SkillExecutor

    ex = SkillExecutor.__new__(SkillExecutor)
    ex.cfg = {"force_limits": {"min_n": 5.0, "max_n": 15.0, "real_validated_max_n": 5.0}}
    ex.robot = SimpleNamespace(backend_name="ur3e")
    assert ex.force_problem(5.0) == ""
    assert "validated" in ex.force_problem(10.0)
    assert "invalid" in ex.force_problem("abc")
    ex.robot = SimpleNamespace(backend_name="rtde_fake")
    assert ex.force_problem(10.0) == "" and ex.force_problem(15.0) == ""


def test_survey_wipe_stop_reset(svc):
    c = TestClient(create_app(svc))
    h = c.get("/health").json()
    assert h["backend"] == "mujoco" and h["mode"] == "full" and h["startup_problems"] == []
    assert c.post("/skill", json={"skill": "survey"}).json()["status"] == "ok"
    r = c.post("/skill", json={"skill": "wipe_zone", "zone_ids": ["C3"], "force_n": 5, "pattern": "zigzag"}).json()
    assert r["status"] == "ok" and r["trace"] and "C3" in r["force"]
    # the client may only add forbidden zones
    r = c.post("/skill", json={"skill": "wipe_zone", "zone_ids": ["B2"], "force_n": 5, "forbidden_zones": ["B2"]}).json()
    assert r["status"] == "rejected"
    assert c.post("/stop", json={"reason": "test"}).json()["stopped"] == "test"
    assert c.post("/skill", json={"skill": "survey"}).status_code == 423
    assert c.post("/reset", json={"by": "tester"}).status_code == 200
    assert c.post("/skill", json={"skill": "survey"}).json()["status"] == "ok"
    log = [json.loads(ln) for ln in open(svc.log_dir / "skill_server.jsonl", encoding="utf-8")]
    assert {"request", "outcome", "stop", "reset"} <= {r["kind"] for r in log}


def test_mujoco_health_is_ready_with_platform(svc):
    """Simulation backend: safety mode is normal (=1, the connection check reports "· ready · software adapter" from it), reports the platform; multiple app builds on the same service share one showcase view."""
    h = TestClient(create_app(svc)).get("/health").json()
    assert h["safety_mode"] == 1 and h["platform"] == "ur3e" and h["stopped"] is None and h["fault"] is None
    v = svc.view
    TestClient(create_app(svc))
    assert v is not None and svc.view is v


@pytest.fixture()
def uncalibrated_ur3e(tmp_path, monkeypatch):
    """ur3e backend + uncalibrated site: the real-robot factory is replaced with a fake RTDE (MuJoCo), no test constructs any real RTDE interface.
    The site name points to a nonexistent directory (= no zones.yaml calibration keys), and no files are created."""
    from cleanproof.robot.backends import rtde_backend
    from cleanproof.robot.backends.fake_rtde import FakeRTDE

    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "robot_token").write_text("ur3e-token\n", encoding="utf-8")
    os.chmod(cfg / "robot_token", 0o600)
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("CLEANPROOF_SITE", "zz_uncalibrated_test_site")
    orig, made = rtde_backend.make_rtde_robot, []

    def factory(backend, mode="full", rcfg=None, host=None, z_surface=None, arming=None, **kw):
        fake = FakeRTDE()
        made.append((backend, mode, fake))
        return orig("rtde_fake", mode, rcfg=rcfg, z_surface=z_surface, fake=fake)

    monkeypatch.setattr(rtde_backend, "make_rtde_robot", factory)
    return made


def test_uncalibrated_ur3e_starts_only_in_shadow_and_refuses_skills(uncalibrated_ur3e, tmp_path):
    from cleanproof.robot.server import NOT_CALIBRATED

    for mode in ("dry_run", "full"):                                  # motion modes: without calibration, startup is still refused
        with pytest.raises(RuntimeError, match="site calibration"):
            build_service("ur3e", mode=mode, log_dir=tmp_path / mode)
    assert uncalibrated_ur3e == []                                    # refusal happens before the robot interface is constructed
    s = build_service("ur3e", mode="shadow", log_dir=tmp_path / "ss")
    try:
        (backend, mode, fake), = uncalibrated_ur3e
        assert mode == "shadow"
        c = TestClient(create_app(s))
        hd = {"X-CleanProof-Token": "ur3e-token"}
        h = c.get("/health", headers=hd).json()
        assert h["backend"] == "ur3e" and h["mode"] == "shadow" and h["calibrated"] is False
        assert h["startup_problems"][0] == NOT_CALIBRATED
        st = c.get("/state", headers=hd).json()                       # the in-app site calibration only reads these
        assert len(st["q"]) == 6 and len(st["tcp_pose"]) == 6
        assert c.get("/config", headers=hd).json()["calibrated"] is False
        for body in ({"skill": "survey"}, {"skill": "inspect_closeup", "zone_ids": ["B2"]},
                     {"skill": "wipe_zone", "zone_ids": ["B2"], "force_n": 5}, {"skill": "survey", "forbidden_zones": ["Q9"]}):
            r = c.post("/skill", json=body, headers=hd)
            assert r.status_code == 423 and r.json()["error"] == NOT_CALIBRATED, body
        assert c.post("/stop", json={"reason": "t"}, headers=hd).status_code == 200
        assert c.post("/reset", json={"by": "tester"}, headers=hd).status_code == 200
        assert c.get("/health", headers=hd).json()["calibrated"] is False      # a reset does not make an uncalibrated site movable
        assert c.post("/skill", json={"skill": "survey"}, headers=hd).status_code == 423
        assert not s.robot.shadow_log and all(fake.count(k) == 0 for k in ("speedL", "moveL", "moveJ", "forceMode"))
        kinds = [json.loads(ln)["kind"] for ln in open(s.log_dir / "skill_server.jsonl", encoding="utf-8")]
        assert kinds.count("refused") == 5 and "request" not in kinds and "outcome" not in kinds
    finally:
        s.close()


def test_busy_returns_409(svc):
    svc.lock.acquire()
    try:
        code, body = svc.run(SkillRequest(skill="survey"))
        assert code == 409
    finally:
        svc.lock.release()


def test_ur3e_requires_live_arming(tmp_path):
    class Dummy:      # the executor should not be called
        robot = None
        grid = None
        forbidden: set = set()
        fault = None

        def survey(self):
            raise AssertionError("skill must not run when not armed")

    tok = ArmingToken(tmp_path / "arming.json", ttl_s=1.0)
    s = SkillService(Dummy(), "ur3e", "full", {}, arming=tok)
    code, body = s.run(SkillRequest(skill="survey"))
    assert code == 423 and "authorization" in body["error"]
    tok.arm("tester", now=time.time() - 5.0)          # already expired
    code, body = s.run(SkillRequest(skill="survey"))
    assert code == 423 and "expired" in body["error"]
    shadow = SkillService(Dummy(), "ur3e", "shadow", {}, arming=tok)
    assert shadow.arming_problem() is None


def test_rtde_fake_shadow_rejects_contact(tmp_path):
    s = build_service("rtde_fake", mode="shadow", scenario="S1_light", log_dir=tmp_path)
    try:
        c = TestClient(create_app(s))
        assert c.get("/health").json()["contact_allowed"] is False
        r = c.post("/skill", json={"skill": "wipe_zone", "zone_ids": ["B2"], "force_n": 5}).json()
        assert r["status"] == "rejected" and "shadow" in r["reason"]
        assert c.post("/skill", json={"skill": "survey"}).json()["status"] == "ok"
        log = [json.loads(ln) for ln in open(tmp_path / "skill_server.jsonl", encoding="utf-8")]
        cmds = [c["cmd"] for r in log if r["kind"] == "shadow_commands" for c in r["commands"]]
        assert "moveJ" in cmds and not s.robot.shadow_log   # shadow: commands are only recorded (written to the log as gate evidence, memory cleared)
    finally:
        s.close()


def test_rtde_software_stop(tmp_path):
    s = build_service("rtde_fake", mode="full", scenario="S1_light", log_dir=tmp_path)
    try:
        r = s.robot
        s.stop("unit test")
        assert r.get_safety_mode() == 99 and not r.speed_l([0, 0, 0.001, 0, 0, 0], 0.1, 0.02)
        assert not r.move_l(r.get_tcp_pose(), 0.01, 0.01)
        code, _ = s.reset("tester")
        assert code == 200 and r.get_safety_mode() == 1
    finally:
        s.close()


def test_end_to_end_over_http(tmp_path, monkeypatch):
    """agent (run_real.py --sim, separate process) → HTTP → skill server (simulation): same loop as running in-process.
    Both sides use the token in a temporary config directory (token auth as in deployment; the real ~/.config/cleanproof is not read)."""
    import uvicorn

    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "robot_token").write_text("e2e-token\n", encoding="utf-8")
    os.chmod(cfg / "robot_token", 0o600)
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(cfg))
    s = build_service("mujoco", scenario="S2_dried", log_dir=tmp_path / "ss")
    assert s.token == "e2e-token"
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(s), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    try:
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "CLEANPROOF_SITE": "", "CLEANPROOF_CONFIG_DIR": str(cfg)}
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_real.py"), "--sim", "--skill-url", url, "--camera-url", url,
                            "--policy", "rules", "--verifier", "cv", "--approve", "auto", "--runs-dir", str(tmp_path / "runs"),
                            "--trials-csv", str(tmp_path / "trials.csv")], capture_output=True, text=True, timeout=600, env=env)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        import csv

        rows = list(csv.DictReader(open(tmp_path / "trials.csv", encoding="utf-8")))
        assert rows[0]["status"] == "done" and rows[0]["human_verdict"] == "sim_truth_clean"
        run = Path(rows[0]["run_dir"])
        kinds = [json.loads(ln)["kind"] for ln in open(run / "events.jsonl", encoding="utf-8")]
        assert kinds[0] == "task_start" and "skill" in kinds and kinds[-1] == "task_end"
    finally:
        server.should_exit = True
        th.join(timeout=10)
        s.close()


# ---------------------------------------------------------------- calibration and camera
def test_calibration_math_and_markers():
    from calibrate_site import detect_markers, fit_plane_z, inscribed_rect, verify_drift

    from cleanproof.perception.camera import PlaneHomography

    xy = np.array([[0.15, -0.22], [0.15, 0.13], [0.47, -0.21], [0.46, 0.12]])
    o, sz = inscribed_rect(xy, 0.035)
    assert np.allclose(o, [0.15 + 0.035, -0.21 + 0.035]) and np.allclose(sz, [0.46 - 0.15 - 0.07, 0.12 + 0.21 - 0.07])
    P = np.c_[xy, [0.061, 0.0612, 0.0608, 0.061]]
    pl = fit_plane_z(P)
    assert abs(pl["z_median"] - 0.061) < 1e-3 and pl["z_span_mm"] < 1.0
    # synthetic frame: 4 markers placed on the tabletop by a known homography; detect → solve homography → agrees with ground truth
    H_true = np.array([[0.0, 1800.0, 640.0], [-1800.0, 0.0, 900.0], [0.0, 0.0, 1.0]])
    img = np.full((720, 1280, 3), 200, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    cam_true = PlaneHomography(H_true, 1280, 720)
    centers = cam_true.project(np.c_[xy, np.zeros(4)])[0]
    for i, (u, v) in enumerate(centers):
        m = cv2.cvtColor(cv2.aruco.generateImageMarker(d, i, 60), cv2.COLOR_GRAY2RGB)
        pad = np.full((80, 80, 3), 255, np.uint8)
        pad[10:70, 10:70] = m
        img[int(v) - 40:int(v) + 40, int(u) - 40:int(u) + 40] = pad
    found = detect_markers(img)
    assert sorted(found) == [0, 1, 2, 3]
    cam = PlaneHomography.from_points(xy, np.array([found[i] for i in range(4)]), 1280, 720)
    test_pt = np.array([[0.30, -0.05, 0.0]])
    assert np.linalg.norm(cam.project(test_pt)[0] - cam_true.project(test_pt)[0]) < 2.0
    calib = {"marker_px": {str(i): found[i].tolist() for i in range(4)}}
    assert verify_drift(img, calib)["max_px"] < 0.5
    assert verify_drift(np.roll(img, 12, axis=1), calib)["max_px"] > 10


def test_reach_report_flags_far_zones():
    from calibrate_site import reach_report

    from cleanproof.common.config import load_yaml

    z, r = load_yaml("zones.yaml"), load_yaml("robot.yaml")
    assert reach_report(z, r) == []
    far = {**z, "work_area": {**z["work_area"], "origin_xy": [0.40, 0.10]}}
    assert any("horizontal radius" in p for p in reach_report(far, r))


def test_camera_server_replay_and_record(tmp_path):
    from cleanproof.perception.camera_server import CameraHub, ReplayCamera, create_app as cam_app

    for i in range(3):
        cv2.imwrite(str(tmp_path / f"f{i}.png"), np.full((120, 160, 3), 40 * i, np.uint8))
    hub = CameraHub({"fixed": ReplayCamera("fixed", tmp_path, fps=20)}, tmp_path / "rec")
    c = TestClient(cam_app(hub))
    r = c.get("/frame/fixed")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert c.get("/frame/wrist").status_code == 404
    assert c.post("/record/start", json={"name": "t1"}).status_code == 200
    assert c.post("/record/start", json={"name": "t2"}).status_code == 409
    time.sleep(0.5)
    out = c.post("/record/stop").json()
    assert out["frames"] >= 3 and Path(out["path"]).stat().st_size > 0


def test_site_overlay(tmp_path, monkeypatch):
    import shutil

    from cleanproof.common import config as cfg

    shutil.copy(cfg.CONFIG_DIR / "zones.yaml", tmp_path / "zones.yaml")
    (tmp_path / "site" / "lab").mkdir(parents=True)
    (tmp_path / "site" / "lab" / "zones.yaml").write_text("table: {z_surface: 0.0612}\nwork_area: {rows: 3}\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("CLEANPROOF_SITE", raising=False)
    assert cfg.load_yaml("zones.yaml")["table"]["z_surface"] == 0.0
    monkeypatch.setenv("CLEANPROOF_SITE", "lab")
    z = cfg.load_yaml("zones.yaml")
    assert z["table"]["z_surface"] == 0.0612 and z["work_area"]["rows"] == 3 and z["work_area"]["cols"] == 5


def test_token_fault_forbidden_guards(svc):
    svc.token = "s3cret"
    c = TestClient(create_app(svc))
    assert c.get("/health").status_code == 401
    h = {"X-CleanProof-Token": "s3cret"}
    assert c.get("/health", headers=h).status_code == 200
    # site forbidden zone: close look not allowed (the tool would drop to 8 cm above the tabletop)
    svc.site_forbidden = {"A1"}
    r = c.post("/skill", json={"skill": "inspect_closeup", "zone_ids": ["A1"]}, headers=h).json()
    assert r["status"] == "rejected" and "site forbidden zone" in r["reason"]
    # unknown forbidden zone id → 422
    assert c.post("/skill", json={"skill": "survey", "forbidden_zones": ["Q9"]}, headers=h).status_code == 422
    # latched fault: every moving skill is rejected, recovers after reset
    svc.ex.fault = "unit-test fault"
    assert c.post("/skill", json={"skill": "survey"}, headers=h).status_code == 423
    assert c.post("/reset", json={"by": "tester"}, headers=h).status_code == 200
    assert c.post("/skill", json={"skill": "survey"}, headers=h).json()["status"] == "ok"


def test_site_forbidden_must_exist_in_grid(svc):
    with pytest.raises(ValueError):
        SkillService(svc.ex, "mujoco", "full", svc.rcfg, site_forbidden=("Z9",))


def test_failed_survey_stops_inspection(tmp_path):
    """When returning to the survey pose fails, do not check using an image that the arm may be blocking; hand off to a human."""
    from cleanproof.agent.loop import TaskRunner
    from cleanproof.agent.policy import RulePolicy
    from cleanproof.agent.runlog import RunLogger
    from cleanproof.perception.residue_cv import ResidueDetector
    from cleanproof.perception.sources import SimCameraSource
    from cleanproof.perception.verifier import CVVerifier
    from cleanproof.robot.skills import SkillOutcome
    from cleanproof.sim.scenarios import build

    sc = build("S1_light")
    try:
        src = SimCameraSource(sc.robot, sc.grid, mode="topdown")
        sc.executor.survey = lambda: SkillOutcome("survey", "aborted", "skill server HTTP 423: motion authorization expired")
        runner = TaskRunner(sc.task, sc.executor, src, CVVerifier(sc.grid, ResidueDetector(sc.grid)), RulePolicy(), RunLogger(tmp_path))
        summary = runner.run()
        assert summary["status"] == "needs_human" and "survey" in summary["reason"]
        kinds = [json.loads(ln)["kind"] for ln in open(runner.log.events_path, encoding="utf-8")]
        assert "inspection" not in kinds and "inspection_aborted" in kinds
    finally:
        sc.close()


def test_health_hides_absolute_paths_and_hostname_without_token(tmp_path, monkeypatch):
    """: A simulation backend without a token: /health has no absolute path or host name; the host name is given only with a token; site is just the site name."""
    monkeypatch.setenv("CLEANPROOF_SITE", "home_dining")
    s = build_service("mujoco", scenario="S2_dried", log_dir=tmp_path / "ss")
    try:
        h = TestClient(create_app(s)).get("/health").json()
        assert "host" not in h and h["site"] == "home_dining" and "/" not in h["site"]
        s.token = "t" * 16
        assert s.health()["host"] == socket.gethostname()
    finally:
        s.close()
