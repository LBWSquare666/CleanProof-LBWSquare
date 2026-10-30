"""In-app guided on-site calibration (cleanproof/ui/calibration.py, /api/calibrate/*): the same procedure as on the real robot computer
(markers → touch x4 → survey → capture → zones → reference → sync). The robot readings (skill server GET /state) and the camera image
(camera service GET /frame/fixed) are fakes: the image draws 4 ArUco markers under a known homography, and the touch points are the base coordinates of the marker centers.
No robot is connected and no RTDE runs; the site and the user config are in temporary directories."""

from __future__ import annotations

import json
import os
import shutil
import time

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from cleanproof.common.config import CONFIG_DIR, use_site
from cleanproof.ui import calibration as cal
from cleanproof.ui import operator as op

MARKER_XY = np.array([[0.12, -0.17], [0.12, 0.13], [0.40, -0.16], [0.39, 0.12]])       # a patch of tabletop the UR3e can reach
Z = np.array([0.0610, 0.0612, 0.0608, 0.0611])
H_TRUE = np.array([[0.0, 1800.0, 640.0], [-1800.0, 0.0, 860.0], [0.0, 0.0, 1.0]])     # base xy → pixels
SURVEY_Q = [-1.15, -1.18, -1.79, -1.74, 1.56, -0.2]


def scene_image(hide: tuple[int, ...] = (), shift_px: int = 0) -> np.ndarray:
    """The tabletop seen by the fixed camera: gradient background + 4 markers (80 px white border)."""
    h, w = 720, 1280
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([120 + 60 * xx / w, 110 + 50 * yy / h, np.full((h, w), 100.0)], -1).astype(np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    pts = (H_TRUE @ np.c_[MARKER_XY, np.ones(4)].T).T
    for i, (u, v, s) in enumerate(pts):
        if i in hide:
            continue
        u, v = int(u / s) + shift_px, int(v / s)
        m = cv2.cvtColor(cv2.aruco.generateImageMarker(d, i, 60), cv2.COLOR_GRAY2RGB)
        pad = np.full((80, 80, 3), 255, np.uint8)
        pad[10:70, 10:70] = m
        img[v - 40:v + 40, u - 40:u + 40] = pad
    return img


class FakeRobot:
    """Stand-in for skill server /state and camera service /frame/fixed; records how many times they were called."""

    def __init__(self):
        self.pose = [0.3, 0.0, 0.2, 3.14, 0.0, 0.0]
        self.q = [0.0] * 6
        self.image = scene_image()
        self.down = False
        self.calls = {"state": 0, "frame": 0}

    def state(self, url):
        if self.down:
            import httpx
            raise httpx.ConnectError("connection refused")
        self.calls["state"] += 1
        return {"now": time.time(), "q": list(self.q), "tcp_pose": list(self.pose), "tcp_force": [0.0, 0.0, -2.0, 0, 0, 0],
                "safety_mode": 1, "busy": None}

    def frame(self, url):
        self.calls["frame"] += 1
        return self.image.copy()


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "site"
    shutil.copytree(CONFIG_DIR / "site", d)
    monkeypatch.setattr(op, "SITE_DIR", d)
    import cleanproof.common.config as cfg

    def site_dir(site=None):
        s = site if site is not None else os.environ.get(cfg.ENV_SITE, "")
        return (d / s) if s else None

    monkeypatch.setattr(cfg, "site_dir", site_dir)
    monkeypatch.setattr(cfg, "load_secrets", lambda: ())
    u = tmp_path / "usercfg"
    u.mkdir()
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(u))
    for name, f in (("SETUP_FILE", "setup.json"), ("SECRETS_FILE", "secrets.env"), ("ROBOT_HOST_FILE", "robot_host"), ("APP_FILE", "app.json")):
        monkeypatch.setattr(op, name, u / f)
    fake = FakeRobot()
    monkeypatch.setattr(cal, "fetch_state", fake.state)
    monkeypatch.setattr(cal, "fetch_frame", fake.frame)
    monkeypatch.setattr(cal, "fetch_camera_health",
                        lambda url: {"cameras": {"fixed": {"exposure": {"locked": True, "exposure": 120, "gain": 2.0, "white_balance": 4600}}}})
    monkeypatch.setattr(cal, "REF_INTERVAL_S", 0.0)
    monkeypatch.setattr(cal, "STATE_POLL_HZ", 200.0)
    dead = "http://127.0.0.1:9"                               # not a single real HTTP request should go out (readings and images both go through the stand-ins)
    st = op.OperatorState("home_dining", runs_dir=tmp_path / "runs", skill_url=dead, camera_url=dead)
    real = {"ok": True, "simulated": False, "backend": "ur3e", "mode": "shadow", "label": "UR3e", "detail": "UR3e · mode shadow · connected"}
    monkeypatch.setattr(st, "robot_link", lambda max_age_s=3.0: real)
    c = TestClient(op.create_operator_app(st), headers={"X-CleanProof": "1"})
    yield c, st, fake, d, u
    use_site(None)


def _step(c, step, **kw):
    return c.post("/api/calibrate/step", json={"site": "test_bench", "step": step, **kw})


def test_guided_calibration_full_flow(env, tmp_path, monkeypatch):
    c, st, fake, sites, ucfg = env
    s = c.post("/api/sites", json={"name": "Test bench", "type": "Factory", "robot": "ur3e"}).json()["site"]
    assert s["id"] == "test_bench" and s["calibrated"] is False and s["job_scenario"] is None
    info = c.get("/api/setup/site", params={"site": "test_bench"}).json()
    assert info["how"] == "guided" and info["calibrated"] is False
    assert c.post("/api/setup/calibrate", json={"site": "test_bench"}).status_code == 409

    stt = c.get("/api/calibrate/state", params={"site": "test_bench"}).json()
    assert [x["id"] for x in stt["steps"]] == ["markers", "touch", "survey", "capture", "zones", "reference", "sync"]
    assert stt["next"] == "markers" and stt["touched"] == [] and not any(x["done"] for x in stt["steps"])
    assert all(x["title"] and x["detail"] for x in stt["steps"])

    # print page: all 4 markers can be detected
    r = c.get("/api/calibrate/markers.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    sheet = cv2.cvtColor(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    assert sorted(cal.calib_cli().detect_markers(sheet)) == [0, 1, 2, 3]

    assert "Touch all four" in _step(c, "capture").json()["detail"]
    assert _step(c, "bogus").status_code == 400
    assert _step(c, "markers").json()["steps"][0]["done"] is True

    # touch x4: read the TCP pose from skill server /state (average of several readings)
    assert _step(c, "touch", marker=7).status_code == 400 and _step(c, "touch").status_code == 400
    fake.down = True
    r = _step(c, "touch", marker=0)
    assert r.status_code == 400 and "robot computer" in r.json()["detail"]
    fake.down = False
    for i in range(4):
        fake.pose = [*MARKER_XY[i], Z[i], 3.14, 0.0, 0.0]
        n0 = fake.calls["state"]
        out = _step(c, "touch", marker=i).json()
        assert fake.calls["state"] - n0 >= 5                   # several readings within about 0.5 s
        assert out["touched"] == list(range(i + 1))
    touch = next(x for x in out["steps"] if x["id"] == "touch")
    assert touch["done"] and touch["detail"].startswith("4 of 4 markers touched · surface flat within")
    assert out["next"] == "survey"

    assert "survey pose first" in _step(c, "zones", rows=2, cols=3).json()["detail"]
    fake.q = SURVEY_Q
    assert _step(c, "survey").json()["next"] == "capture"
    z0 = yaml.safe_load((sites / "test_bench" / "zones.yaml").read_text())
    assert set(z0) == {"survey_q"} and c.get("/api/setup/site", params={"site": "test_bench"}).json()["calibrated"] is False
    pts = json.loads((sites / "test_bench" / "calib_points.json").read_text())      # the same intermediate file as the command line, with the same keys
    assert np.allclose(pts["touch"]["2"]["tcp_pose"][:3], [*MARKER_XY[2], Z[2]]) and np.allclose(pts["survey_q"], SURVEY_Q)

    # capture: a marker is blocked → 400 and the reason, save the failed image; normal → homography
    fake.image = scene_image(hide=(3,))
    r = _step(c, "capture")
    assert r.status_code == 400 and "Marker 3 not found" in r.json()["detail"]
    assert (sites / "test_bench" / "capture_failed.jpg").is_file()
    fake.image = scene_image()
    out = _step(c, "capture").json()
    assert out["next"] == "zones" and next(x for x in out["steps"] if x["id"] == "capture")["detail"].startswith("4 markers found")
    calib = json.loads((sites / "test_bench" / "calib.json").read_text())
    from cleanproof.perception.camera import PlaneHomography
    cam = PlaneHomography.from_dict(calib["fixed"])
    p = np.array([[0.30, -0.05, 0.0]])
    ref_px = (H_TRUE @ np.array([0.30, -0.05, 1.0]))
    assert np.linalg.norm(cam.project(p)[0][0] - ref_px[:2] / ref_px[2]) < 2.0
    assert abs(calib["mm_per_px_mean"] - 1000 / 1800) < 0.02

    # zones: plane fit + inscribed rectangle (inset 35 mm) + reachability; write zones.yaml (with the survey pose) and robot.yaml (pad diameter)
    assert _step(c, "zones", rows=0, cols=4).status_code == 400
    assert _step(c, "zones", rows="x").status_code == 400
    r = _step(c, "zones", rows=3, cols=4, pad_diameter_mm=100)           # cells smaller than the pad
    assert r.status_code == 400 and "smaller than the polishing pad" in r.json()["detail"]
    assert "work_area" not in yaml.safe_load((sites / "test_bench" / "zones.yaml").read_text())       # there is a warning: not written
    out = _step(c, "zones", rows=2, cols=3, inset_mm=35, pad_diameter_mm=60)
    assert out.status_code == 200, out.text
    zinfo = next(x for x in out.json()["steps"] if x["id"] == "zones")
    assert zinfo["done"] and zinfo["detail"].startswith("2 × 3 zones · 200 × 210 mm") and "reach OK" in zinfo["detail"]
    z = yaml.safe_load((sites / "test_bench" / "zones.yaml").read_text())
    assert z["work_area"] == {"origin_xy": [0.155, -0.125], "size_xy": [0.2, 0.21], "rows": 2, "cols": 3}
    assert abs(z["table"]["z_surface"] - 0.061) < 1e-3 and np.allclose(z["survey_q"], SURVEY_Q)
    assert yaml.safe_load((sites / "test_bench" / "robot.yaml").read_text())["tool"]["pad_radius_m"] == 0.03
    info = c.get("/api/setup/site", params={"site": "test_bench"}).json()
    assert info["zones"] == 6 and info["calibrated"] is False and info["reference"] is None      # clean reference not captured yet: not counted as calibrated
    assert c.get("/api/sites/test_bench/reference.png").status_code == 404                       # and a rendered image is not passed off as the reference

    # reference: the arm must be at the survey pose; multi-frame median → top-down clean reference (same size as the zone grid); the locked exposure is written to cameras.yaml
    fake.q = [v + 0.3 for v in SURVEY_Q]
    assert "survey pose" in _step(c, "reference").json()["detail"]
    fake.q = SURVEY_Q
    n0 = fake.calls["frame"]
    out = _step(c, "reference").json()
    assert fake.calls["frame"] - n0 == cal.REF_FRAMES and out["next"] == "sync"
    ref = cv2.imread(str(sites / "test_bench" / "reference.png"))
    assert ref.shape[:2] == (400, 420)
    assert yaml.safe_load((sites / "test_bench" / "cameras.yaml").read_text())["cameras"]["fixed"] == {"exposure": 120, "gain": 2.0,
                                                                                                         "white_balance": 4600}
    rec = json.loads((sites / "test_bench" / "calibration.json").read_text())
    assert rec["source"] == "guided" and rec["rows"] == 2 and rec["cols"] == 3
    for f in ("calib_overlay.jpg", "reference_grid.jpg", "capture_raw.png"):
        assert (sites / "test_bench" / f).is_file(), f
    # reference captured but not yet synced to the robot computer: not counted as calibrated, the guide panel goes back to the Sync step
    info = c.get("/api/setup/site", params={"site": "test_bench"}).json()
    assert info["calibrated"] is False and info["needs_sync"] is True and info["how"] == "guided" and info["zones"] == 6
    assert c.get("/api/calibrate/state", params={"site": "test_bench"}).json()["next"] == "sync"
    # the robot computer reports this very site and it is calibrated (synced elsewhere): that also counts as synced
    real = st.robot_link()
    real.update(site="test_bench", calibrated=True)
    assert c.get("/api/setup/site", params={"site": "test_bench"}).json()["calibrated"] is True
    real.update(site=None, calibrated=None)

    # sync: not paired → 400; script missing → 400; script failed → 400 with the reason; success → with --restart, ROBOT_PC_HOST = paired host
    assert "No paired robot computer" in _step(c, "sync").json()["detail"]
    # the robot is still executing a skill / a job is running: do not sync (syncing restarts the skill server)
    st.robot_link()["busy"] = "wipe_zone"
    r = _step(c, "sync")
    assert r.status_code == 409 and r.json()["detail"] == "The robot is still running a skill; wait for it to finish or press E-stop"
    st.robot_link()["busy"] = None
    from types import SimpleNamespace
    st.live = SimpleNamespace(running=True)
    assert _step(c, "sync").status_code == 409
    st.live = None
    # the built-in simulated deployment does not prompt for sync (needs_sync only applies in paired live mode)
    st.sim_robot = True
    info = c.get("/api/setup/site", params={"site": "test_bench"}).json()
    assert info["needs_sync"] is False and info["calibrated"] is True
    st.sim_robot = False
    op.ROBOT_HOST_FILE.write_text("robot-pc\n")
    monkeypatch.setattr(cal, "PUSH_SITE", tmp_path / "missing.sh")
    assert "not installed" in _step(c, "sync").json()["detail"]
    script = tmp_path / "push_site.sh"
    log = tmp_path / "push.log"
    script.write_text(f'#!/bin/bash\necho "$ROBOT_PC_HOST $*" >> {log}\n[ -f {tmp_path}/fail ] && {{ echo "ssh: connect refused" >&2; exit 1; }}\nexit 0\n')
    monkeypatch.setattr(cal, "PUSH_SITE", script)
    (tmp_path / "fail").write_text("1")
    r = _step(c, "sync")
    assert r.status_code == 400 and "ssh: connect refused" in r.json()["detail"]
    (tmp_path / "fail").unlink()
    st._link = (time.time(), {"ok": False})
    out = _step(c, "sync").json()
    assert st._link == (0.0, None)
    assert out["next"] is None and all(x["done"] for x in out["steps"])
    assert log.read_text().splitlines()[-1] == "robot-pc test_bench --restart"
    assert next(x for x in out["steps"] if x["id"] == "sync")["detail"] == "Sent to robot-pc · robot services restarted"
    info = c.get("/api/setup/site", params={"site": "test_bench"}).json()
    assert info["calibrated"] and info["needs_sync"] is False and info["zones"] == 6
    assert info["calibrated_utc"] == time.strftime("%Y-%m-%d", time.gmtime())
    assert c.get(info["reference"]).content == (sites / "test_bench" / "reference.png").read_bytes()

    # touch one marker again: the later steps that depend on it are invalidated and must be redone
    fake.pose = [*MARKER_XY[0], Z[0], 3.14, 0.0, 0.0]
    out = _step(c, "touch", marker=0).json()
    done = {x["id"]: x["done"] for x in out["steps"]}
    assert done["touch"] and done["survey"] and not done["capture"] and not done["zones"] and not done["reference"] and not done["sync"]
    assert out["next"] == "capture"
    assert c.get("/api/setup/site", params={"site": "test_bench"}).json()["calibrated"] is False     # re-calibration in progress
    # wrong touch points: the side ratios of the quadrilateral do not match → 400, the existing calib.json is not overwritten, and this step is not counted as done
    good = (sites / "test_bench" / "calib.json").read_bytes()
    fake.pose = [MARKER_XY[3][0] + 0.06, MARKER_XY[3][1], Z[3], 3.14, 0.0, 0.0]
    _step(c, "touch", marker=3)
    r = _step(c, "capture")
    assert r.status_code == 400 and "disagree" in r.json()["detail"]
    assert (sites / "test_bench" / "calib.json").read_bytes() == good and (sites / "test_bench" / "calib_rejected.json").is_file()
    assert not next(x for x in c.get("/api/calibrate/state", params={"site": "test_bench"}).json()["steps"] if x["id"] == "capture")["done"]
    fake.pose = [*MARKER_XY[3], Z[3], 3.14, 0.0, 0.0]
    _step(c, "touch", marker=3)
    # the camera was bumped: the reference step is refused (marker drift)
    assert _step(c, "capture").status_code == 200 and _step(c, "zones", rows=2, cols=3).status_code == 200
    fake.image = scene_image(shift_px=15)
    r = _step(c, "reference")
    assert r.status_code == 400 and "camera seems to have moved" in r.json()["detail"]


def test_guided_zones_warn_until_forced(env):
    c, st, fake, sites, _ = env
    c.post("/api/sites", json={"name": "Test bench", "type": "Factory", "robot": "ur3e"})
    for i in range(4):
        fake.pose = [*MARKER_XY[i], 0.061 + (0.01 if i == 3 else 0.0), 3.14, 0.0, 0.0]     # one corner 10 mm high: the table is not flat
        _step(c, "touch", marker=i)
    _step(c, "survey")
    r = _step(c, "zones", rows=2, cols=3)
    assert r.status_code == 400 and r.json()["detail"].startswith("Not saved:") and "not level" in r.json()["detail"]
    assert "work_area" not in yaml.safe_load((sites / "test_bench" / "zones.yaml").read_text())
    r = _step(c, "zones", rows=2, cols=3, force=True)
    assert r.status_code == 200 and "work_area" in yaml.safe_load((sites / "test_bench" / "zones.yaml").read_text())


def test_update_yaml_file_only_touches_given_keys(tmp_path):
    f = tmp_path / "z.yaml"
    src = ("# head\ntable: { z_surface: 0.0, mug: { xy: [1, 2] } }   # flow\nwork_area:\n  # inner comment\n  origin_xy: [0.1, 0.2]  # o\n"
           "  rows: 3\nsurvey_q: [1.0,\n           2.0]   # two lines\nother: keep   # c\n")
    f.write_text(src)
    assert cal.update_yaml_file(f, {("table", "z_surface"): 0.0, ("work_area", "rows"): 3}) is False      # unchanged: not written
    assert f.read_text() == src
    assert cal.update_yaml_file(f, {("table", "z_surface"): 0.012, ("work_area", "cols"): 4, ("survey_q",): [3.0, 4.0],
                                    ("cameras", "fixed", "gain"): 2.0})
    out = f.read_text()
    y = yaml.safe_load(out)
    assert y["table"] == {"z_surface": 0.012, "mug": {"xy": [1, 2]}} and y["work_area"] == {"origin_xy": [0.1, 0.2], "rows": 3, "cols": 4}
    assert y["survey_q"] == [3.0, 4.0] and y["other"] == "keep" and y["cameras"] == {"fixed": {"gain": 2.0}}
    for keep in ("# head", "  # inner comment", "  origin_xy: [0.1, 0.2]  # o", "other: keep   # c", "   # flow", "   # two lines"):
        assert keep in out, keep
    g = tmp_path / "new.yaml"
    assert cal.update_yaml_file(g, cal.grid_updates(0.0, [0.18, -0.18], [0.24, 0.3], 4, 5), cal.ZONES_HEADER)
    assert g.read_text().startswith("# ") and yaml.safe_load(g.read_text())["work_area"]["cols"] == 5
