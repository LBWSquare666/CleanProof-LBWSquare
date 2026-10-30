"""Operator software (cleanproof/ui/operator.py): the two kinds of robot computer in live mode - the built-in local simulation (--sim-robot) and a paired robot computer
(played here by an in-process mujoco skill server, connected over HTTP, the same path as a real-robot deployment). Settings page validation, site calibration and creating sites, starting jobs, the picture, reports.
All offline: no model calls, no real robot; the user config directory and the site config are in temporary directories."""

from __future__ import annotations

import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from cleanproof.common.config import CONFIG_DIR, use_site
from cleanproof.ui import calibration as cal
from cleanproof.ui import operator as op

KEY = "tp-test-key-not-real-0123456789"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(pred, timeout=90.0, dt=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture(scope="module")
def cfgdir(tmp_path_factory):
    """The user config directory shared by this module (it holds the shared token, which the skill server and the software both authenticate with)."""
    d = tmp_path_factory.mktemp("usercfg")
    tok = d / "robot_token"
    tok.write_text("test-token-" + os.urandom(8).hex() + "\n", encoding="utf-8")
    os.chmod(tok, 0o600)
    mp = pytest.MonkeyPatch()
    mp.setenv("CLEANPROOF_CONFIG_DIR", str(d))
    import cleanproof.robot.server as srv
    if hasattr(srv, "DEFAULT_TOKEN_PATH"):
        mp.setattr(srv, "DEFAULT_TOKEN_PATH", str(tok))
    yield d
    mp.undo()


@pytest.fixture(autouse=True)
def user_cfg(cfgdir, monkeypatch):
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(cfgdir))
    for name, f in (("SETUP_FILE", "setup.json"), ("SECRETS_FILE", "secrets.env"), ("ROBOT_HOST_FILE", "robot_host"), ("APP_FILE", "app.json")):
        (cfgdir / f).unlink(missing_ok=True)
        monkeypatch.setattr(op, name, cfgdir / f)
    import cleanproof.common.config as cfg
    monkeypatch.setattr(cfg, "load_secrets", lambda: ())
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    yield cfgdir
    os.environ.pop("NEBIUS_API_KEY", None)


@pytest.fixture
def site_tmp(tmp_path, monkeypatch):
    """Copy the site directory to a temporary directory: calibration, creating sites and settings-page writes never touch configs/site in the repo."""
    d = tmp_path / "site"
    shutil.copytree(CONFIG_DIR / "site", d)
    monkeypatch.setattr(op, "SITE_DIR", d)
    import cleanproof.common.config as cfg

    def site_dir(site=None):
        s = site if site is not None else os.environ.get(cfg.ENV_SITE, "")
        return (d / s) if s else None

    monkeypatch.setattr(cfg, "site_dir", site_dir)
    yield d
    use_site(None)
    from cleanproof.agent.force_table import reload_force_table
    reload_force_table()


@pytest.fixture(scope="module")
def robot_pc(cfgdir, tmp_path_factory):
    """A paired robot computer: an in-process mujoco skill server (home dining table scenario, authenticated by the shared token), accessed over HTTP."""
    pytest.importorskip("mujoco")
    import uvicorn

    from cleanproof.robot.server import build_service, create_app
    os.environ.setdefault("MUJOCO_GL", "egl")
    s = build_service("mujoco", scenario="H5_dining_permanent", log_dir=tmp_path_factory.mktemp("ss"))
    assert s.token, "skill server should pick up the shared token"
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(s), host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    assert _wait(lambda: server.started, timeout=20)
    yield SimpleNamespace(url=f"http://127.0.0.1:{port}", svc=s)
    server.should_exit = True
    th.join(timeout=10)
    s.close()
    use_site(None)


@pytest.fixture(autouse=True)
def key_probe_ok(monkeypatch):
    """: A key is verified online before it is saved: in tests this passes by default and touches no network; the rejected / cannot-connect paths are set up separately in dedicated cases."""
    from cleanproof.ui import deploy_checks as dc
    monkeypatch.setattr(dc, "probe_api_key", lambda key, timeout=8.0: (True, ""))


def _client(tmp_path, **kw):
    st = op.OperatorState("home_dining", runs_dir=tmp_path / "runs", **kw)
    return TestClient(op.create_operator_app(st), headers={"X-CleanProof": "1"}), st


def _dead_url() -> str:
    return f"http://127.0.0.1:{_free_port()}"                 # nobody is listening: cannot connect


# ---------------------------------------------------------------- settings page
def test_settings_put_validates_everything_before_writing(site_tmp, tmp_path, user_cfg):
    c, st = _client(tmp_path, sim_robot=True)
    g = c.get("/api/settings").json()
    assert g["runtime"] == {"policy": "rules", "verifier": "cv", "api_key_set": False}
    assert g["robot"]["mode"] == "sim" and g["robot"]["sim_robot"] is True and g["about"]["release"] == op.APP_VERSION
    bad = [{"runtime": {"policy": "gpt"}}, {"runtime": {"verifier": "eyes"}}, {"robot_link": {"skill_url": "ftp://robot:8100"}},
           {"robot_link": {"camera_url": "not a url"}}, {"api_key": "has space"}, {"runtime": {"policy": "nemotron"}},
           {"robot_link": {"skill": "http://x"}},
           # site settings valid, local settings invalid: the whole save is rejected, and the site file is not written either
           {"jobs": {"max_passes_per_zone": 4}, "api_key": "bad key"},
           {"jobs": {"max_passes_per_zone": 4}, "runtime": {"verifier": "x"}},
           # local settings valid, site settings invalid: neither the key nor app.json is written
           {"api_key": KEY, "runtime": {"verifier": "vlm"}, "jobs": {"max_passes_per_zone": 99}},
           {"max_n": {"sealed_wood": 13}, "jobs": {"speed": 3}}]
    for b in bad:
        r = c.put("/api/settings", json=b)
        assert r.status_code == 400, b
        assert KEY not in r.text
    # the robot computer address can only be the local tunnel (loopback address) or https://<paired robot computer>: the shared token is never sent to an arbitrary host
    for u in ("http://10.0.0.5:8100", "https://evil.example:8100", "http://robot.local:18101", "http://user:pw@127.0.0.1:8100",
              "http://127.0.0.1:8100/x?y=1"):
        r = c.put("/api/settings", json={"robot_link": {"skill_url": u}})
        assert r.status_code == 400, u
    assert "encrypted tunnel on this computer (127.0.0.1)" in c.put("/api/settings", json={"robot_link": {"skill_url": "http://10.0.0.5:8100"}}).json()["detail"]
    for f in ("app.json", "secrets.env"):
        assert not (user_cfg / f).exists(), f
    (user_cfg / "robot_host").write_text("robot.local\n")                                  # paired: its https address is allowed
    for f in ("operator.yaml", "force_table.yaml", "models.yaml"):
        assert not (site_tmp / "home_dining" / f).exists(), f
    assert "API key" in c.put("/api/settings", json={"runtime": {"policy": "nemotron"}}).json()["detail"]
    # valid: key + planning / checking mode + robot computer address + site settings, saved in one go; returns the full settings (same structure as GET), without echoing the key
    r = c.put("/api/settings", json={"api_key": f"  {KEY} ", "runtime": {"policy": "nemotron", "verifier": "vlm"},
                                     "robot_link": {"skill_url": "http://127.0.0.1:18100/", "camera_url": "https://robot.local:18101"},
                                     "jobs": {"max_passes_per_zone": 4}})
    assert r.status_code == 200 and KEY not in r.text
    out = r.json()
    assert set(g) <= set(out) and out["runtime"] == {"policy": "nemotron", "verifier": "vlm", "api_key_set": True}
    assert out["robot"]["skill_url"] == "http://127.0.0.1:18100" and out["robot"]["camera_url"] == "https://robot.local:18101"
    assert out["jobs"]["max_passes_per_zone"] == 4
    sec = user_cfg / "secrets.env"
    assert sec.read_text().strip() == f"NEBIUS_API_KEY={KEY}" and (sec.stat().st_mode & 0o777) == 0o600
    app = json.loads((user_cfg / "app.json").read_text())
    assert app == {"policy": "nemotron", "verifier": "vlm", "skill_url": "http://127.0.0.1:18100", "camera_url": "https://robot.local:18101"}
    assert st.policy == "nemotron" and st.verifier == "vlm"
    for path in ("/api/settings", "/api/meta"):
        assert KEY not in c.get(path).text
    # restart the software: the local options saved on the settings page take precedence over the startup arguments
    st2 = op.OperatorState("home_dining", runs_dir=tmp_path / "runs", sim_robot=True, policy="rules", verifier="cv")
    assert (st2.policy, st2.verifier, st2.skill_url) == ("nemotron", "vlm", "http://127.0.0.1:18100")
    # a key already exists: switching to Nemotron later needs no re-entry; an empty key field does not overwrite
    assert c.put("/api/settings", json={"runtime": {"policy": "rules"}, "api_key": ""}).status_code == 200
    assert c.put("/api/settings", json={"runtime": {"policy": "nemotron"}}).status_code == 200
    assert sec.read_text().strip() == f"NEBIUS_API_KEY={KEY}"
    # empty address = back to the default tunnel port; an address given explicitly on the command line takes precedence over the one saved on the settings page
    st3 = op.OperatorState("home_dining", runs_dir=tmp_path / "runs", sim_robot=True, skill_url="http://127.0.0.2:9100")
    assert st3.skill_url == "http://127.0.0.2:9100" and st3.camera_url == "https://robot.local:18101"
    # an invalid address is stored in app.json (e.g. the paired host was changed): not used, back to the default
    (user_cfg / "robot_host").unlink()
    st4 = op.OperatorState("home_dining", runs_dir=tmp_path / "runs", sim_robot=True)
    assert st4.camera_url == op.DEFAULT_CAMERA_URL and st4.skill_url == "http://127.0.0.1:18100"
    r = c.put("/api/settings", json={"robot_link": {"skill_url": "", "camera_url": " "}}).json()
    assert r["robot"]["skill_url"] == op.DEFAULT_SKILL_URL and r["robot"]["camera_url"] == op.DEFAULT_CAMERA_URL
    assert "skill_url" not in json.loads((user_cfg / "app.json").read_text())
    assert c.put("/api/settings", json={"robot_link": {"skill_url": "http://robot:99999"}}).status_code == 400


def test_config_dir_env_is_honoured(tmp_path, monkeypatch):
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(tmp_path / "cfgx"))
    assert op.user_config_dir() == tmp_path / "cfgx"
    monkeypatch.delenv("CLEANPROOF_CONFIG_DIR")
    assert op.user_config_dir() == Path("~/.config/cleanproof").expanduser()
    # save_api_key writes by default to the user config directory (as recorded by the module)
    monkeypatch.setattr(op, "SECRETS_FILE", tmp_path / "cfgy" / "secrets.env")
    op.save_api_key(KEY)
    assert (tmp_path / "cfgy" / "secrets.env").read_text().strip() == f"NEBIUS_API_KEY={KEY}"
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)


def test_zones_named_in_task():
    assert op.zones_in_task("Wipe B2 and C3 only", 3, 5) == ["B2", "C3"]
    assert op.zones_in_task("clean b2, b2 and zone C 5", 3, 5) == ["B2", "C5"]
    assert op.zones_in_task("Clean the table. Then D1 and A9.", 3, 5) == []          # not in this site's grid
    assert op.zones_in_task("Wipe a 2 cm spot near the mug", 3, 5) == []              # the article "a" + a number, not zone A2
    assert op.zones_in_task("A1-A3 are wet; skip E1", 3, 5) == ["A1", "A2", "A3"]     # range; E1 is not in the 3 × 5 grid
    assert op.zones_in_task("", 3, 5) == [] and op.zones_in_task(None, 3, 5) == []


def test_meta_and_robot_link_in_sim_mode(site_tmp, tmp_path):
    c, _ = _client(tmp_path, sim_robot=True)
    m = c.get("/api/meta").json()
    assert m["robot_mode"] == "sim" and m["app_version"] == "1.0" and m["tunnel_ports"] == [8100, 8101] and m["robot_link"] is None
    tpl = {t["id"]: t for t in m["sim_templates"]}
    assert tpl["home_dining"] == {"id": "home_dining", "name": "Home · dining table", "robot": "spot_arm"}
    assert set(tpl) == {s["id"] for s in m["sites"] if s.get("job_scenario")}
    r = c.get("/api/robot/link").json()
    assert r["mode"] == "sim" and r["link"] is None and r["skill_server"]["ok"] and r["skill_server"]["detail"].startswith("built-in simulation")
    # paired mode, robot computer not running: honestly show that it cannot connect (no made-up status)
    c2, _ = _client(tmp_path, skill_url=_dead_url(), camera_url="http://127.0.0.1:18101")
    m2 = c2.get("/api/meta").json()
    assert m2["robot_mode"] == "paired" and m2["sim_templates"] == [] and m2["tunnel_ports"][1] == 18101
    lk = c2.get("/api/robot/link").json()
    assert lk["mode"] == "paired" and lk["link"]["ok"] is False and lk["skill_server"] == {"ok": False, "detail": "skill server not reachable"}
    assert set(lk["link"]) >= {"ok", "detail", "backend", "mode", "label", "host", "simulated", "camera", "skill_url", "camera_url"}


# ---------------------------------------------------------------- calibration (local built-in simulation)
def _png(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def test_builtin_sim_calibration_keeps_look_and_files(site_tmp, tmp_path):
    c, st = _client(tmp_path, sim_robot=True)
    zf = site_tmp / "home_dining" / "zones.yaml"
    before_txt, before_mtime = zf.read_text(), zf.stat().st_mtime_ns
    s0 = c.get("/api/setup/site", params={"site": "home_dining"}).json()
    assert s0["calibrated"] and s0["zones"] == 15 and s0["how"] == "auto" and "?v=" in s0["reference"]
    r0 = c.get(s0["reference"])
    img0 = _png(r0.content)                                       # before calibration: the tabletop render (this is the one in the video)
    r = c.post("/api/setup/calibrate", json={"site": "home_dining"})
    assert r.status_code == 200, r.text
    s1 = r.json()
    assert s1["calibrated"] and s1["zones"] == 15 and s1["grid"]["rows"] == 3 and s1["grid"]["cols"] == 5
    assert s1["calibrated_utc"] == time.strftime("%Y-%m-%d", time.gmtime()) and s1["reference"] != s0["reference"]
    assert zf.read_text() == before_txt and zf.stat().st_mtime_ns == before_mtime      # grid unchanged: zones.yaml is not rewritten
    rec = json.loads((site_tmp / "home_dining" / "calibration.json").read_text())
    assert rec["source"] == "built-in simulation" and rec["rows"] == 3 and rec["cols"] == 5 and rec["size_m"] == [0.34, 0.6]
    assert (site_tmp / "home_dining" / "reference.png").is_file()
    r1 = c.get(s1["reference"])
    img1 = _png(r1.content)
    assert c.get("/api/sites/home_dining/reference.png", headers={"If-None-Match": r1.headers["etag"]}).status_code == 304
    assert r1.headers["etag"] != r0.headers["etag"]                # the version changed after calibration: the address without ?v= also gets the new image
    assert img1.shape == img0.shape and int(np.abs(img1.astype(int) - img0.astype(int)).max()) == 0   # the captured clean reference is identical pixel by pixel to the one shown before
    # a site never calibrated (only site.yaml): after calibration there is zones.yaml (only the zone keys) and a reference image.
    # the demo site in the repo was already calibrated in the software (019808e), so the temporary copy is first reverted to having only site.yaml
    for f in ("zones.yaml", "reference.png", "calibration.json"):
        (site_tmp / "carwash_van" / f).unlink(missing_ok=True)
    s = c.get("/api/setup/site", params={"site": "carwash_van"}).json()
    assert s["calibrated"] is False and s["reference"] is None and s["calibrated_utc"] is None
    s = c.post("/api/setup/calibrate", json={"site": "carwash_van"}).json()
    assert s["calibrated"] and s["zones"] == 20 and s["reference"]
    import yaml
    z = yaml.safe_load((site_tmp / "carwash_van" / "zones.yaml").read_text())
    assert set(z) == {"table", "work_area"} and z["work_area"]["rows"] == 4
    assert c.get(s["reference"]).headers["content-type"] == "image/png"
    # the grid was changed (row count): calibration only changes back this one value; comments, end-of-line comments, set-dressing keys (mug, survey_arm_q) and other styles are kept as is
    zc = site_tmp / "carwash_van" / "zones.yaml"
    edited = ("# hand-written\ntable:\n  z_surface: 0.0   # measured\n  mug: { xy: [0.1, 0.2], yaw_deg: 30 }\n"
              "work_area:\n  origin_xy: [0.18, -0.18]\n  size_xy: [0.24, 0.30]\n  rows: 3   # was edited\n  cols: 5\n"
              "# scene\nsurvey_arm_q: { ur3e: [0, 1] }\n")
    zc.write_text(edited)
    assert c.get("/api/setup/site", params={"site": "carwash_van"}).json()["zones"] == 15
    assert c.post("/api/setup/calibrate", json={"site": "carwash_van"}).json()["zones"] == 20
    assert zc.read_text() == edited.replace("  rows: 3   #", "  rows: 4   #")
    assert c.post("/api/setup/calibrate", json={"site": "nope"}).status_code == 404


def test_new_site_then_calibrate(site_tmp, tmp_path):
    c, _ = _client(tmp_path, sim_robot=True)
    for bad in ({"name": "", "type": "Home", "robot": "spot_arm"}, {"name": "X", "type": "Moon", "robot": "spot_arm"},
                {"name": "X", "type": "Home", "robot": "r2d2"}, {"name": "X", "type": "Home", "robot": "spot_arm", "template": "nope"}):
        assert c.post("/api/sites", json=bad).status_code == 400, bad
    r = c.post("/api/sites", json={"name": "Kitchen Island #2", "type": "Home", "robot": "spot_arm"})
    assert r.status_code == 200, r.text
    s = r.json()["site"]
    assert s["id"] == "kitchen_island_2" and s["name"] == "Kitchen Island #2" and s["type"] == "Homes" and s["robot"] == "spot_arm"
    assert s["job_scenario"] == "H5_dining_permanent" and s["calibrated"] is False and s["surface"] == "sealed_wood"
    assert (site_tmp / "kitchen_island_2" / "site.yaml").read_text().startswith("#")
    assert c.post("/api/sites", json={"name": "kitchen island #2", "type": "Home", "robot": "spot_arm"}).status_code == 400   # duplicate name
    assert c.post("/api/sites", json={"name": "Kitchen island 2", "type": "Homes", "robot": "spot_arm"}).json()["site"]["id"] == "kitchen_island_2_2"
    w = c.post("/api/sites", json={"name": "Wash bay", "type": "Car wash", "robot": "rizon4"}).json()["site"]
    assert w["job_scenario"] == "G1_window_glass" and w["type"] == "Car washes"          # default template: a site for the same robot
    assert c.get("/api/meta").json()["sites"][-1]["id"] in {"wash_bay", "kitchen_island_2_2", "kitchen_island_2"}
    cal_ = c.post("/api/setup/calibrate", json={"site": "kitchen_island_2"}).json()
    assert cal_["calibrated"] and cal_["zones"] == 15 and cal_["reference"]
    # create in paired mode: no simulated scenario
    c2, _ = _client(tmp_path, skill_url=_dead_url())
    p = c2.post("/api/sites", json={"name": "Test bench", "type": "Factory", "robot": "ur3e"}).json()["site"]
    assert p["id"] == "test_bench" and p["job_scenario"] is None and p["type"] == "Factories"


def test_calibrate_real_robot_computer_is_guided(site_tmp, tmp_path, monkeypatch):
    c, st = _client(tmp_path, skill_url=_dead_url())
    assert c.post("/api/setup/calibrate", json={"site": "home_dining"}).status_code == 400        # cannot connect: a readable 400
    monkeypatch.setattr(st, "robot_link", lambda max_age_s=3.0: {"ok": True, "simulated": False, "backend": "ur3e", "detail": "UR3e"})
    assert c.get("/api/setup/site", params={"site": "home_dining"}).json()["how"] == "guided"
    r = c.post("/api/setup/calibrate", json={"site": "home_dining"})
    assert r.status_code == 409 and r.json() == {"detail": "guided"}


def test_apply_auto_mismatched_capture_keeps_surface_look(site_tmp):
    """The reference image of the robot computer simulation looks different from the tabletop render (e.g. --source render): reference.png stores the tabletop render, and the captured one is saved separately and recorded."""
    d = site_tmp / "home_dining"
    zones = cal.site_zones(d)
    look = cal.surface_render(d, "sealed_wood")
    other = np.full_like(look, 90)
    rec = cal.apply_auto(d, {"table": zones["table"], "work_area": zones["work_area"]}, other, "robot computer simulation · test",
                         "sealed_wood", compare_look=True)
    assert rec["reference"]["kind"] == "surface_render" and rec["reference"]["capture"] == "reference_capture.png"
    assert int(np.abs(cv2.cvtColor(cv2.imread(str(d / "reference.png")), cv2.COLOR_BGR2RGB).astype(int) - look.astype(int)).max()) == 0
    assert (d / "reference_capture.png").is_file()
    jpg = cv2.imdecode(cv2.imencode(".jpg", look, [cv2.IMWRITE_JPEG_QUALITY, 98])[1], cv2.IMREAD_COLOR)
    assert cal.visually_same(jpg, look)[0]


# ---------------------------------------------------------------- paired robot computer (simulated backend)
def test_paired_connect_checks_frame_link_and_calibration(robot_pc, site_tmp, tmp_path):
    c, st = _client(tmp_path, skill_url=robot_pc.url, camera_url=robot_pc.url)
    r = c.post("/api/setup/connect", json={"robot": "spot_arm"}).json()
    assert [x["id"] for x in r["checks"]] == ["skill_server", "camera", "adapter"]
    from cleanproof.robot.platforms import load_platforms
    label = load_platforms()["spot_arm"].label
    assert [x["detail"] for x in r["checks"]] == ["connected · token OK", "1 camera stream live", f"{label} · ready · software adapter"]
    assert r["ok"] is True
    lk = c.get("/api/robot/link").json()
    assert lk["mode"] == "paired" and lk["link"]["ok"] and lk["link"]["simulated"] is True and lk["link"]["backend"] == "mujoco"
    assert {"busy", "stopped", "fault"} <= set(lk["link"]) and lk["link"]["stopped"] is None and lk["link"]["fault"] is None
    assert lk["skill_server"]["ok"] and "token OK" in lk["skill_server"]["detail"]
    m = c.get("/api/meta").json()
    assert m["robot_mode"] == "paired" and m["robot_link"]["camera"] is True
    # camera image before a job starts: the robot computer's current image (the local simulation is not used)
    f = c.get("/api/jobs/frame", params={"t": 0, "site": "home_dining"})
    assert f.status_code == 200 and f.headers["content-type"] == "image/jpeg" and f.content[:2] == b"\xff\xd8"
    assert st.previews == {}
    # one-click calibration: the zone grid comes from skill server /config, the clean reference from /sim/reference (looks the same as the tabletop render → store the captured one)
    zf = site_tmp / "home_dining" / "zones.yaml"
    before = zf.read_text()
    zf.write_text(before.replace("  rows: 3\n", "  rows: 4   # was edited\n"))        # this local copy differs from what the robot computer actually uses
    s = c.get("/api/setup/site", params={"site": "home_dining"}).json()
    assert s["how"] == "auto" and s["zones"] == 20
    r = c.post("/api/setup/calibrate", json={"site": "home_dining"})
    assert r.status_code == 200, r.text
    assert r.json()["zones"] == 15 and zf.read_text() == before.replace("  rows: 3\n", "  rows: 3   # was edited\n")
    mt = zf.stat().st_mtime_ns
    assert c.post("/api/setup/calibrate", json={"site": "home_dining"}).status_code == 200 and zf.stat().st_mtime_ns == mt   # unchanged: not rewritten
    rec = json.loads((site_tmp / "home_dining" / "calibration.json").read_text())
    assert rec["source"].startswith("robot computer simulation") and rec["reference"]["kind"] == "captured"
    assert rec["reference"]["mean_diff_vs_surface_render"] < 2.0


def test_paired_job_start_and_report(robot_pc, site_tmp, tmp_path):
    # in paired mode a job does not need a simulated template: a job can start even if the site has no job_scenario
    f = site_tmp / "home_dining" / "site.yaml"
    f.write_text("\n".join(ln for ln in f.read_text().splitlines() if not ln.startswith("job_scenario")) + "\n")
    c, st = _client(tmp_path, skill_url=robot_pc.url, camera_url=robot_pc.url)
    assert c.put("/api/settings", json={"jobs": {"no_wipe_zones": ["B3"]}}).status_code == 200
    assert c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Wipe B3 only"}).status_code == 400   # only the no-wipe zone was named
    r = c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Wipe C2 and B3 please"})
    assert r.status_code == 200, r.text
    assert type(st.live).__name__ == "RemoteSession"
    import inspect

    from cleanproof.ui.session import RemoteSession
    if "robot" in inspect.signature(RemoteSession).parameters:     # the robot registered for the site is passed to the paired job (written to reference_meta)
        assert st.live.robot_key == "spot_arm"
    assert st.live.task["zones"] == ["C2"] and st.live.task["forbidden_zones"] == ["B3"]
    assert _wait(lambda: st.live.finished, timeout=90), "remote job did not finish"
    assert st.live.error is None, st.live.error
    s = c.get("/api/jobs/state").json()
    rc = s["live"]["robot_computer"]
    assert rc["backend"] == "mujoco" and rc["simulated"] is True and rc["skill_url"] == robot_pc.url
    rid = s["run_id"]
    assert rid.startswith("live-home_dining-mujoco-")
    meta = json.loads((tmp_path / "runs" / rid / "reference_meta.json").read_text())
    assert meta["site"] == "home_dining"
    rows = c.get("/api/reports").json()["reports"]
    row = next(x for x in rows if x["run_id"] == rid)
    assert row["site"] == "Home · dining table" and row["site_id"] == "home_dining"
    from cleanproof.common.config import current_site
    use_site("carwash_van")
    assert "Cleaning report" in c.get(f"/api/reports/{rid}").text
    assert current_site() == "carwash_van"                                 # viewing a report does not switch the global site (the running job reads its config by it)
    real_live = st.live
    st.live = SimpleNamespace(running=True, run_dir=None, finished=False)
    assert c.post("/api/setup/finish", json={"robot": "spot_arm", "site": "carwash_van"}).status_code == 409
    assert current_site() == "carwash_van"
    st.live = real_live
    use_site(None)
    reqs = c.get("/api/requests").json()["requests"]
    if reqs:                                                   # request to hand over to a person: the note may be left empty
        assert c.post("/api/requests/resolve", json={"id": reqs[0]["id"], "note": ""}).json()["ok"]
    assert c.get("/api/jobs/frame", params={"t": 0}).status_code == 200


def test_reports_read_site_and_robot_from_reference_meta(site_tmp, tmp_path):
    runs = tmp_path / "runs"
    d = runs / "live-carwash_van-ur3e-shadow-20261029-120000-000001"
    d.mkdir(parents=True)
    (d / "reference_meta.json").write_text(json.dumps({"site": "factory_line", "robot": {"platform": "panda"}}))
    assert op.run_site(d) == "factory_line" and op.run_platform(d) == "panda"          # reference_meta takes precedence over the directory name
    (d / "reference_meta.json").write_text(json.dumps({"source": "x"}))
    assert op.run_site(d) == "carwash_van" and op.run_platform(d) == "spot_arm"
    e = runs / "sim-H5_dining_permanent-rules-cv-spot_arm-20261029-120000-000002"
    e.mkdir()
    (e / "reference_meta.json").write_text(json.dumps({"scenario": {"site": "home_dining", "operator_site": "office_stairs"},
                                                       "robot": {"platform": "spot_arm"}}))
    assert op.run_site(e) == "office_stairs"                    # local live job: by the site the job was started with (several sites can share one simulated scenario)


def _serve(app):
    import uvicorn
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    assert _wait(lambda: server.started, timeout=20)
    return f"http://127.0.0.1:{port}", server, th


def test_wizard_connect_lets_uncalibrated_real_robot_through(robot_pc, site_tmp, tmp_path, monkeypatch):
    """A new real-robot computer: the site is not calibrated yet, the skill server starts only in shadow and rejects all skills (/health calibrated=false).
    The wizard Connect step lets it through as "connected, awaiting calibration" (fallback), the Site step says not calibrated and goes to on-site guidance; until calibrated and synced, every job start is refused.
    Here the in-process simulated skill server reports a real-robot backend + not calibrated (no RTDE interface is created)."""
    from cleanproof.perception.camera_server import CameraHub, ReplayCamera, create_app as cam_app
    from cleanproof.robot.platforms import load_platforms
    from cleanproof.robot.server import NOT_CALIBRATED
    from cleanproof.ui.deploy_checks import check_skill_server
    orig = robot_pc.svc.health

    def health():
        return {**orig(), "backend": "ur3e", "mode": "shadow", "label": "UR3e", "calibrated": False, "site": "", "platform": "ur3e",
                "startup_problems": [NOT_CALIBRATED]}
    monkeypatch.setattr(robot_pc.svc, "health", health)
    frames = tmp_path / "frames"
    frames.mkdir()
    cv2.imwrite(str(frames / "f0.png"), np.full((120, 160, 3), 128, np.uint8))
    hub = CameraHub({"fixed": ReplayCamera("fixed", frames, fps=20)}, tmp_path / "rec")
    cam_url, server, th = _serve(cam_app(hub))
    try:
        time.sleep(0.1)
        c, st = _client(tmp_path, skill_url=robot_pc.url, camera_url=cam_url)
        r = c.post("/api/setup/connect", json={"robot": "ur3e"}).json()
        sk, cam, ad = r["checks"]
        assert sk["ok"] and sk["fallback"] and sk["detail"] == "connected · token OK · site not calibrated yet"
        assert cam["ok"] and cam["detail"] == "1 camera stream live" and not cam.get("fallback")
        label = load_platforms()["ur3e"].label
        assert ad["ok"] and ad["fallback"] and ad["detail"] == f"{label} · connected · calibrate the site in the next steps"
        assert r["ok"] is True                                                  # the wizard can continue to the Site step
        assert check_skill_server(robot_pc.url)["ok"] is False                  # the deployment check still honestly reports not ready
        lk = c.get("/api/robot/link").json()["link"]
        assert lk["ok"] and lk["simulated"] is False and lk["calibrated"] is False
        new = c.post("/api/sites", json={"name": "Test bench", "type": "Factory", "robot": "ur3e"}).json()["site"]
        for sid in (new["id"], "home_dining"):                                   # calibrated locally but not on the robot computer: also counts as not calibrated
            s = c.get("/api/setup/site", params={"site": sid}).json()
            assert s["calibrated"] is False and s["how"] == "guided" and s["reference"] is None, sid
        assert c.post("/api/setup/calibrate", json={"site": "home_dining"}).status_code == 409
        r = c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Clean the table"})
        assert r.status_code == 400 and "not calibrated on the robot computer" in r.json()["detail"] and st.live is None
        # the robot computer is set to another site (calibrated): starting a job gives an actionable hint
        monkeypatch.setattr(robot_pc.svc, "health", lambda: {**health(), "calibrated": True, "startup_problems": [],
                                                             "site": "/home/robot/cleanproof/configs/site/factory_line"})
        r = c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Clean the table"})
        assert r.status_code == 400 and r.json()["detail"] == ("The robot computer is set up for Factory line · workbench. "
                                                               "Sync this site to it from Setup › Site, or pick Factory line · workbench.")
        # not ready for another reason (e.g. E-stop): not let through
        monkeypatch.setattr(robot_pc.svc, "health", lambda: {**health(), "stopped": "operator E-STOP"})
        r = c.post("/api/setup/connect", json={"robot": "ur3e"}).json()
        assert r["ok"] is False and r["checks"][0]["ok"] is False and not r["checks"][0].get("fallback")
        # robot computer in E-stop / has an unreset fault: refuse before starting a job, and say clearly what to do (do not start a job and then end with a software error)
        r = c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Clean the table"})
        assert r.status_code == 400 and r.json()["detail"] == "The robot computer is stopped (E-stop). Reset it on site before the next job."
        monkeypatch.setattr(robot_pc.svc, "health", lambda: {**health(), "fault": "joint 3 overcurrent\ntrace"})
        r = c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Clean the table"})
        assert r.status_code == 400 and r.json()["detail"] == ("The robot computer reports a fault (joint 3 overcurrent). "
                                                               "Reset it on site before the next job.")
        assert st.live is None
    finally:
        server.should_exit = True
        th.join(timeout=10)
        hub.close()


# ---------------------------------------------------------------- (E-stop, keys, platform check, picture, concurrency, request validation, site ownership)
def test_paired_estop_without_job_reaches_robot_computer(robot_pc, site_tmp, tmp_path):
    c, st = _client(tmp_path, skill_url=robot_pc.url, camera_url=robot_pc.url)
    try:
        r = c.post("/api/jobs/estop")
        assert r.status_code == 200 and r.json() == {"ok": True, "estop_delivered": True}
        assert robot_pc.svc.stopped                                           # the skill server enters software stop
        assert c.get("/api/robot/link").json()["link"]["stopped"]           # the title bar can show the stopped state
        assert st.live is None                                           # no job was recorded as E-stop
    finally:
        import httpx

        from cleanproof.robot.server import auth_headers
        assert httpx.post(robot_pc.url + "/reset", json={"by": "test"}, headers=auth_headers(), timeout=10).status_code == 200
    c2, _ = _client(tmp_path, skill_url=_dead_url())
    r = c2.post("/api/jobs/estop")
    assert r.status_code == 502 and r.json()["estop_delivered"] is False and "physical E-stop" in r.json()["detail"]
    c3, _ = _client(tmp_path, sim_robot=True)
    assert c3.post("/api/jobs/estop").status_code == 409                   # the built-in simulation is not running: nothing to stop


def test_connect_rejects_other_platform_and_frame_keeps_last_on_timeout(robot_pc, site_tmp, tmp_path, monkeypatch):
    c, st = _client(tmp_path, skill_url=robot_pc.url, camera_url=robot_pc.url)
    from cleanproof.robot.platforms import load_platforms
    specs = load_platforms()
    ad = c.post("/api/setup/connect", json={"robot": "ur3e"}).json()["checks"][2]       # the robot computer has a Spot
    assert ad["ok"] is False and ad["detail"] == f"{specs['spot_arm'].label} on the robot computer, not {specs['ur3e'].label}"
    first = c.get("/api/jobs/frame", params={"t": 0, "site": "home_dining"}).content
    import httpx
    st._remote_frame = (0.0, st._remote_frame[1])                          # cache expired

    def timeout(*a, **k):
        raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(httpx, "get", timeout)
    st._link = (time.time(), st._link[1])
    r = c.get("/api/jobs/frame", params={"t": 0, "site": "home_dining"})
    assert r.status_code == 200 and r.content == first                     # timeout: keep the previous frame, do not fall back to the top-down image


def test_new_api_key_resets_model_client(site_tmp, tmp_path, monkeypatch):
    import cleanproof.ui.deploy_checks as dc
    for name in ("check_tokenfactory", "check_cosmos", "check_orchestrator"):        # no network (and certainly no Token Factory calls)
        monkeypatch.setattr(dc, name, lambda *a, **k: {"ok": True, "detail": "stub"})
    c, st = _client(tmp_path, sim_robot=True)
    st.client = object()
    assert c.put("/api/settings", json={"api_key": KEY}).status_code == 200 and st.client is None
    st.client = old = object()
    st.live = SimpleNamespace(running=True)                                # a job is running: use up the old client before swapping
    assert c.post("/api/setup/models", json={"api_key": KEY + "x"}).status_code == 200
    assert st.client is old and st._client_stale
    st.live.running = False
    assert st.get_client() is None and not st._client_stale                 # rule planning + classical vision: no model client needed


def test_calibration_how_when_robot_computer_unreachable(site_tmp, tmp_path):
    c, st = _client(tmp_path, skill_url=_dead_url())
    assert c.get("/api/setup/site", params={"site": "home_dining"}).json()["how"] == "unknown"     # never connected: no guessing
    st._last_backend = "mujoco"
    assert c.get("/api/setup/site", params={"site": "home_dining"}).json()["how"] == "auto"
    st._last_backend = "ur3e"
    assert c.get("/api/setup/site", params={"site": "home_dining"}).json()["how"] == "guided"


def test_malformed_requests_do_not_echo_body(site_tmp, tmp_path):
    c, _ = _client(tmp_path, sim_robot=True)
    for kw in ({"content": ('{"api_key": "' + KEY + '",').encode(), "headers": {"content-type": "application/json"}},
               {"json": [KEY]}):
        r = c.put("/api/settings", **kw)
        assert r.status_code == 422 and KEY not in r.text and r.json()["detail"].startswith("Invalid request")


def test_concurrent_duplicate_site_creation(site_tmp):
    out = []

    def make():
        try:
            out.append(op.create_site("Twin table", "Home", "spot_arm", True)["id"])
        except ValueError as e:
            out.append(f"400:{e}")
    th = [threading.Thread(target=make) for _ in range(4)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert sorted(x.startswith("400:") for x in out) == [False, True, True, True] and "twin_table" in out


def test_new_site_does_not_take_over_template_runs_and_finish_sets_site(site_tmp, tmp_path):
    c, st = _client(tmp_path, sim_robot=True)
    s = c.post("/api/sites", json={"name": "Kitchen island", "type": "Home", "robot": "spot_arm"}).json()["site"]
    assert s["job_scenario"] == "H5_dining_permanent" and s["scenarios"] == []           # only borrows the simulated scenario, does not copy the template's recording list
    assert op.site_of_scenario("H5_dining_permanent") == "home_dining" and op.site_of_scenario("H1_dining_dried") == "home_dining"
    runs = tmp_path / "runs"
    old = runs / "sim-H5_dining_permanent-rules-cv-spot_arm-20261028-120000-000001"
    old.mkdir(parents=True)
    assert op.run_site(old) == "home_dining"                                              # earlier jobs of the template site are not renamed
    # wizard Finish (live mode): the chosen site becomes this computer's current site, and after a restart it follows setup.json
    r = c.post("/api/setup/finish", json={"robot": "spot_arm", "site": s["id"], "site_type": "Home", "site_name": "Kitchen island"})
    assert r.status_code == 200 and st.site == s["id"] and op.setup_site_choice() == s["id"]
    assert c.get("/api/meta").json()["site"] == s["id"]
    assert c.post("/api/setup/finish", json={"robot": "spot_arm", "site": "nope"}).status_code == 400 and st.site == s["id"]
    shutil.rmtree(site_tmp / s["id"])
    assert op.setup_site_choice() is None                                                 # the site is gone: not used


def test_wizard_models_order_and_fallbacks(site_tmp, tmp_path, monkeypatch):
    """Wizard Models: the order is the same as the deployment checks; Token Factory only falls back to rule planning when no key is filled in, and a rejection after a key is filled in = failure with the reason (the key is not echoed);
    if Cosmos is wanted but cannot be reached: the wizard is not blocked, and it honestly says the zone check falls back to classical vision (when a VLMVerifier call fails, per-zone cv_fallback)."""
    import cleanproof.ui.deploy_checks as dc

    def boom(msg):
        def f(*a, **k):
            raise RuntimeError(msg)
        return f
    monkeypatch.setattr(dc, "check_orchestrator", lambda *a, **k: {"ok": True, "detail": "stub"})
    monkeypatch.setattr(dc, "check_cosmos", boom("ConnectError: [Errno 111] Connection refused"))
    monkeypatch.setattr(dc, "check_tokenfactory", lambda *a, **k: {"ok": False, "detail": "API key MISSING"})
    c, st = _client(tmp_path, sim_robot=True)
    r = c.post("/api/setup/models", json={}).json()
    assert [x["id"] for x in r["checks"]] == ["orchestrator", "cosmos", "tokenfactory"] and r["ok"]
    assert r["checks"][2]["fallback"] and r["checks"][2]["detail"] == "Not configured · rules planner is used"
    assert r["checks"][1]["fallback"] and r["checks"][1]["detail"] == "Not running here · classical vision check is used"
    st.verifier = "vlm"
    r = c.post("/api/setup/models", json={}).json()
    assert r["ok"] and r["checks"][1] == {**r["checks"][1], "ok": True, "fallback": True,
                                          "detail": "Not reachable · zone checks fall back to classical vision"}
    # key filled in but rejected by Token Factory: the reason is readable and does not contain the key; when planning uses rules the wizard is not blocked (amber), when using Nemotron it counts as failure
    monkeypatch.setattr(dc, "check_tokenfactory", boom(f"Client error '401 Unauthorized' for url 'https://tf/models' key={KEY}"))
    r = c.post("/api/setup/models", json={"api_key": KEY})
    assert r.status_code == 200 and KEY not in r.text
    tf = r.json()["checks"][2]
    assert tf["ok"] and tf["fallback"] and tf["detail"] == "API key rejected by Nebius Token Factory · check the key · rules planner is used"
    st.policy = "nemotron"
    r = c.post("/api/setup/models", json={})
    tf = r.json()["checks"][2]
    assert tf["ok"] is False and not tf.get("fallback") and tf["detail"] == "API key rejected by Nebius Token Factory · check the key"
    assert r.json()["ok"] is False and KEY not in r.text
    monkeypatch.setattr(dc, "check_tokenfactory", boom("ConnectTimeout: timed out"))
    assert c.post("/api/setup/models", json={}).json()["checks"][2]["detail"].startswith("Nebius Token Factory not reachable")
    bad = c.post("/api/setup/models", json={"api_key": "has space"})
    assert bad.status_code == 400 and bad.json()["detail"] == "Invalid API key"                # the message for the operator is capitalized
