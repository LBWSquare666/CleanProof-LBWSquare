"""Operator software (cleanproof/ui/operator.py): the API behind every button on the page really takes effect (all offline, no model calls)."""

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cleanproof.common.config import CONFIG_DIR, use_site
from cleanproof.ui import operator as op


def _wait(pred, timeout=90.0, dt=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture
def site_tmp(tmp_path, monkeypatch):
    """Copy the site directory to a temporary directory, so settings-page writes never touch the config in the repo."""
    d = tmp_path / "site"
    shutil.copytree(CONFIG_DIR / "site", d)
    monkeypatch.setattr(op, "SITE_DIR", d)
    import cleanproof.common.config as cfg
    orig = cfg.site_dir

    def site_dir(site=None):
        import os
        s = site if site is not None else os.environ.get(cfg.ENV_SITE, "")
        return (d / s) if s else None

    monkeypatch.setattr(cfg, "site_dir", site_dir)
    yield d
    monkeypatch.setattr(cfg, "site_dir", orig)
    use_site(None)
    from cleanproof.agent.force_table import reload_force_table
    reload_force_table()


@pytest.fixture(autouse=True)
def user_cfg_tmp(tmp_path, monkeypatch):
    """The user config directory (keys, paired host, local options of the wizard and settings page) always points to a temporary directory: tests never read or write the real ~/.config/cleanproof."""
    d = tmp_path / "usercfg"
    d.mkdir()
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(d))
    for name, f in (("SETUP_FILE", "setup.json"), ("SECRETS_FILE", "secrets.env"), ("ROBOT_HOST_FILE", "robot_host"), ("APP_FILE", "app.json")):
        monkeypatch.setattr(op, name, d / f)
    import cleanproof.common.config as cfg
    monkeypatch.setattr(cfg, "load_secrets", lambda: ())
    yield d


@pytest.fixture(autouse=True)
def key_probe_ok(monkeypatch):
    """: A key is verified online before it is saved: in tests this passes by default and touches no network; the rejected / cannot-connect paths are set up separately in dedicated cases."""
    from cleanproof.ui import deploy_checks as dc
    monkeypatch.setattr(dc, "probe_api_key", lambda key, timeout=8.0: (True, ""))


def _client(tmp_path, **kw):
    kw.setdefault("sim_robot", True)                     # by default the desktop simulated deployment (paired-mode tests are in test_operator_live.py)
    st = op.OperatorState("home_dining", runs_dir=tmp_path / "runs", **kw)
    return TestClient(op.create_operator_app(st), headers={"X-CleanProof": "1"}), st


def test_meta_sites_and_page(site_tmp, tmp_path):
    c, _ = _client(tmp_path)
    assert "setView" in c.get("/").text
    m = c.get("/api/meta").json()
    ids = {s["id"] for s in m["sites"]}
    assert {"home_dining", "carwash_van", "factory_line", "office_stairs"} <= ids
    home = next(s for s in m["sites"] if s["id"] == "home_dining")
    assert home["grid"]["rows"] * home["grid"]["cols"] == 15 and home["robot"] == "spot_arm"
    assert len(m["supported"]) == 12 and m["mode"] == "live"
    assert c.get("/api/sites/home_dining/reference.png").headers["content-type"] == "image/png"


def test_settings_save_changes_guardrail_cap(site_tmp, tmp_path):
    c, _ = _client(tmp_path)
    s = c.get("/api/settings").json()
    wood = next(x for x in s["surfaces"] if x["surface"] == "sealed_wood")
    assert wood["max_n"] == 15.0 and wood["min_allowed_n"] == 12.0
    assert c.put("/api/settings", json={"max_n": {"sealed_wood": 10}}).status_code == 400      # below the starting force: rejected, not clamped
    r = c.put("/api/settings", json={"max_n": {"sealed_wood": 12}, "require_plan_approval": True})
    assert r.status_code == 200
    assert next(x for x in r.json()["surfaces"] if x["surface"] == "sealed_wood")["max_n"] == 12.0
    assert r.json()["require_plan_approval"] is True
    # the guardrail follows the new limit: in this site's force table the wood limit becomes 12 N
    use_site("home_dining")
    from cleanproof.agent.force_table import load_force_table, reload_force_table
    reload_force_table()
    assert load_force_table().cap("sealed_wood") == 12.0
    assert load_force_table().next(12.0, "sealed_wood") is None          # limit reached: the next step can only hand over to a person
    # reset to default: the site override file is deleted
    r = c.put("/api/settings", json={"max_n": {"sealed_wood": 15}})
    assert r.status_code == 200 and not (site_tmp / "home_dining" / "force_table.yaml").exists()


def test_force_lookup_screens_without_table_values(site_tmp, tmp_path):
    c, _ = _client(tmp_path)
    L = c.get("/api/forcedb/lookup", params={"stain": "dried_stain", "surface": "sealed_wood"}).json()
    assert len(L["entries"]) == 30 and all("force_n" not in e for e in L["entries"])
    assert len(L["on_surface"]) == 5
    assert L["hit"]["force_n"] == 12.0 and L["hit"]["label"] == "Dried stain · Wood"
    assert c.get("/api/forcedb/lookup", params={"stain": "unknown", "surface": "sealed_wood"}).json()["unknown_fallback"] is True


def test_live_job_start_pause_stop_and_staff_request(site_tmp, tmp_path, monkeypatch):
    # local live job: use the generic scenario of the default site (rule policy + classical vision, no model calls), and go through the buttons one by one
    (site_tmp / "home_dining" / "site.yaml").write_text(
        "name: Test site\nsurface: sealed_wood\nrobot: ur3e\nscenarios: [S3_permanent]\njob_scenario: S3_permanent\n", encoding="utf-8")
    for f in ("zones.yaml", "sim.yaml", "robot.yaml", "platforms.yaml"):
        (site_tmp / "home_dining" / f).unlink(missing_ok=True)
    c, st = _client(tmp_path)
    assert c.put("/api/settings", json={"jobs": {"max_passes_per_zone": 2, "no_wipe_zones": ["a1"]}}).status_code == 200
    assert c.post("/api/jobs/pause").status_code == 409                  # no job yet
    r = c.post("/api/jobs/start", json={"site": "home_dining", "instruction": "Clean the table by the window"})
    assert r.status_code == 200 and r.json()["status"] in ("running", "idle", "needs_human")
    assert c.post("/api/jobs/start", json={}).status_code in (409, 200)
    assert _wait(lambda: st.live.finished, timeout=180)
    assert st.live.error is None, st.live.error
    s = c.get("/api/jobs/state").json()
    assert s["status"] == "needs_human"
    assert s["wipes"] >= 1 and s["checks"] >= 2 and s["log"] and s["log"][-1]["kind"] == "end"      # a live job has the same structure as a recording (activity log, statistics)
    assert any(z["status"] for z in s["zones"].values())
    from cleanproof.common.jsonl import read_jsonl
    start = next(e for e in read_jsonl(Path(st.live.run_dir) / "events.jsonl") if e["kind"] == "task_start")
    assert start["task"]["instruction"] == "Clean the table by the window"      # the instruction typed by the operator went into this job
    assert start["task"]["max_passes_per_zone"] == 2 and "A1" in start["task"]["forbidden_zones"]   # the Jobs group on the settings page takes effect from this job on
    reqs = c.get("/api/requests").json()["requests"]
    assert reqs and reqs[0]["resolved"] is None
    r = c.post("/api/requests/resolve", json={"id": reqs[0]["id"], "note": "Pen mark, sent to the cleaning crew"}).json()
    assert r["ok"] and r["requests"][0]["resolved"]["note"].startswith("Pen mark")
    rid = s["run_id"]
    rows = c.get("/api/reports").json()["reports"]
    assert any(x["run_id"] == rid for x in rows)
    html = c.get(f"/api/reports/{rid}").text
    assert "Cleaning report" in html and "Resolved by operator" in html
    assert "imulation" not in html and "MuJoCo" not in html and str(tmp_path) not in html


def test_recorded_deploy_checks(site_tmp, tmp_path):
    rec = tmp_path / "checks.json"
    rec.write_text(json.dumps({"site": "home_dining", "ok": True, "checks": [{"id": "cosmos", "ok": True}]}), encoding="utf-8")
    c, _ = _client(tmp_path, deploy_record=rec)
    assert c.get("/api/deploy/checks", params={"recorded": 1}).json()["ok"] is True


def test_settings_groups_validate_and_apply(site_tmp, tmp_path):
    c, _ = _client(tmp_path)
    s = c.get("/api/settings").json()
    assert s["jobs"] == op.JOB_DEFAULTS and s["notifications"]["staff_check_webhook"] == ""
    assert s["status"]["source"] == "local" and "NEBIUS" not in json.dumps(s["status"]).replace("NEBIUS_API_KEY", "")
    for bad in ({"jobs": {"max_passes_per_zone": 9}}, {"jobs": {"no_wipe_zones": ["Z9"]}}, {"jobs": {"speed": 1}},
                {"escalate_step_n": 9}, {"budget_usd": 500}, {"staff_check_webhook": "ftp://x"}):
        assert c.put("/api/settings", json=bad).status_code == 400, bad
    r = c.put("/api/settings", json={"jobs": {"max_passes_per_zone": 4, "recheck_all": False}, "escalate_step_n": 3, "budget_usd": 20,
                                     "staff_check_webhook": "https://example.invalid/hook"}).json()
    assert r["jobs"]["max_passes_per_zone"] == 4 and r["jobs"]["recheck_all"] is False and r["escalate"]["value"] == 3.0
    assert r["budget"]["limit_usd"] == 20.0 and r["notifications"]["staff_check_webhook"].startswith("https://")
    use_site("home_dining")
    from cleanproof.agent.force_table import load_force_table, reload_force_table
    reload_force_table()
    assert load_force_table().next(5.0, "sealed_wood") == 8.0              # re-wipe force step is 3 N, and the guardrail lets it through by that
    r = c.put("/api/settings", json={"jobs": op.JOB_DEFAULTS, "escalate_step_n": r["escalate"]["default"], "budget_usd": r["budget"]["max_usd"],
                                     "staff_check_webhook": ""})
    assert r.status_code == 200
    for f in ("operator.yaml", "force_table.yaml", "models.yaml"):
        assert not (site_tmp / "home_dining" / f).exists(), f              # reset to default: all override files are deleted


def test_setup_wizard_endpoints(site_tmp, tmp_path, monkeypatch):
    rec = tmp_path / "checks.json"
    rec.write_text(json.dumps({"site": "home_dining", "ok": True, "checks": [
        {"id": i, "title": i, "ok": True, "detail": "ok", "latency_ms": 1.0} for i in
        ("orchestrator", "cosmos", "tokenfactory", "skill_server", "camera", "adapter")]}), encoding="utf-8")
    monkeypatch.setattr(op, "SETUP_FILE", tmp_path / "cfg" / "setup.json")
    monkeypatch.setattr(op, "SECRETS_FILE", tmp_path / "cfg" / "secrets.env")
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    c, _ = _client(tmp_path, deploy_record=rec)
    assert c.get("/api/meta").json()["setup_done"] is False
    r = c.post("/api/setup/connect", json={"robot": "spot_arm"}).json()
    assert r["ok"] and [x["id"] for x in r["checks"]] == ["skill_server", "camera", "adapter"]
    r = c.post("/api/setup/models", json={"api_key": "test-key-not-real"})
    assert r.status_code == 200 and "test-key-not-real" not in r.text
    f = tmp_path / "cfg" / "secrets.env"
    assert f.read_text().strip() == "NEBIUS_API_KEY=test-key-not-real" and (f.stat().st_mode & 0o777) == 0o600
    assert c.post("/api/setup/models", json={"api_key": "has space"}).status_code == 400
    r = c.get("/api/setup/site", params={"site": "home_dining"}).json()
    assert r["calibrated"] and r["zones"] == 15 and len(r["calibrated_utc"]) == 10
    assert c.post("/api/setup/finish", json={"robot": "nope", "site": "home_dining"}).status_code == 400
    assert c.post("/api/setup/finish", json={"robot": "spot_arm", "site": "home_dining", "site_type": "Home", "site_name": "Kitchen table"}).json()["ok"]
    m = c.get("/api/meta").json()
    assert m["setup_done"] is True and next(s for s in m["sites"] if s["id"] == "home_dining")["name"] == "Kitchen table"
    y = (site_tmp / "home_dining" / "site.yaml").read_text()
    assert y.startswith("#")                                                             # the comment at the start of the file is kept
    assert "\ntype: Homes\n" in y and "\nname: Kitchen table\n" in y                    # the wizard's Home button → site category Homes (the word used on the Sites page)
    assert "scenarios: [H1_dining_dried," in y and "job_scenario: H5_dining_permanent        # " in y   # the other lines are unchanged (list style, end-of-line comments)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)


def test_robot_photo_prefers_licensed_cutout(site_tmp, tmp_path):
    c, _ = _client(tmp_path)
    m = c.get("/api/meta").json()
    photo = {r["key"] for r in m["supported"] if r["photo"]}
    assert {"spot_arm", "sawyer"} <= photo
    assert {r["group"] for r in m["supported"]} == {"Arms", "Mobile manipulators"}
    for k in ("spot_arm", "ur5e"):
        assert c.get(f"/robots/photo/{k}.png").headers["content-type"] == "image/png"
    assert c.get("/robots/photo/..%2Fsecrets.png").status_code == 404


def test_brand_app_icon_svg(site_tmp, tmp_path):
    """The title bar app icon uses the vector version /brand/cleanproof_app.svg (lion badge); the page's .logo references it."""
    c, _ = _client(tmp_path)
    r = c.get("/brand/cleanproof_app.svg")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/svg+xml") and r.text.startswith("<svg")
    assert c.get("/brand/..%2F..%2Fsecret.svg").status_code == 404
    assert "url(/brand/cleanproof_app.svg)" in c.get("/").text


def test_robot_host_from_pairing(site_tmp, tmp_path, monkeypatch):
    monkeypatch.setattr(op, "ROBOT_HOST_FILE", tmp_path / "robot_host")
    c, _ = _client(tmp_path)
    assert c.get("/api/meta").json()["robot_host"] is None                # not paired: no made-up host name
    (tmp_path / "robot_host").write_text("robot-pc\n")
    assert c.get("/api/meta").json()["robot_host"] == "robot-pc"
    assert c.get("/api/settings").json()["robot"]["host"] == "robot-pc"
    assert c.post("/api/settings/test-webhook").status_code == 409        # no webhook configured


def test_sim_robot_wizard_is_this_computer(site_tmp, tmp_path, monkeypatch):
    """Desktop version (--sim-robot): the wizard's robot computer is the built-in local simulation; the connection checks really run, and on failure give a readable English hint without exposing the raw internal exception."""
    monkeypatch.setattr(op, "ROBOT_HOST_FILE", tmp_path / "robot_host")
    c, _ = _client(tmp_path, sim_robot=True)
    m = c.get("/api/meta").json()
    assert m["sim_robot"] is True and m["robot_host"] is None
    import cleanproof.robot.platforms as pf

    def missing(*_a, **_k):
        raise FileNotFoundError("MuJoCo Menagerie not found")
    monkeypatch.setattr(pf, "build_platform_scene", missing)
    r = c.post("/api/setup/connect", json={"robot": "spot_arm"}).json()
    assert [x["id"] for x in r["checks"]] == ["skill_server", "camera", "adapter"]
    ad = r["checks"][2]
    assert r["ok"] is False and ad["ok"] is False and "CLEANPROOF_MENAGERIE" in ad["detail"] and "FileNotFoundError" not in ad["detail"]
    assert r["checks"][0]["ok"] and "MuJoCo" in r["checks"][0]["detail"]
    page = c.get("/").text
    assert "This computer · built-in simulation" in page and "MuJoCo simulation · no pairing needed" in page


def test_live_camera_before_first_job(site_tmp, tmp_path, monkeypatch):
    """Live mode with no job started yet: the camera image is a simulated tabletop built from this site's task template (stains already there), not a 404 / broken image."""
    monkeypatch.setenv("MUJOCO_GL", "egl")
    c, st = _client(tmp_path)
    called = []

    def fake(st_, site):
        called.append(site)
        return b"\xff\xd8fakejpeg"
    monkeypatch.setattr(op, "preview_frame", fake)
    r = c.get("/api/jobs/frame", params={"t": 0, "site": "home_dining"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and called == ["home_dining"]
    assert c.get("/api/jobs/frame", params={"t": 0}).status_code == 200 and called[-1] == "home_dining"


def test_update_site_refuses_block_values(site_tmp, tmp_path):
    """When a key in site.yaml uses multi-line style it is not forcibly changed: wizard Finish returns 400 and the reason, and the file is not corrupted."""
    f = site_tmp / "home_dining" / "site.yaml"
    f.write_text(f.read_text().replace("name: Home · dining table", "name: >-\n  Home · dining table"))
    before = f.read_text()
    with pytest.raises(ValueError):
        op.update_site("home_dining", {"name": "Kitchen"})
    assert f.read_text() == before


def test_finished_live_jobs_stay_in_reports(site_tmp, tmp_path):
    """Every job started in live mode stays in Reports / Today: after starting a new job or restarting the software the earlier jobs do not disappear."""
    runs = tmp_path / "runs"
    for n in ("sim-H5_dining_permanent-rules-cv-spot_arm-20261029-010000-000001", "sim-H5_dining_permanent-rules-cv-spot_arm-20261029-010100-000002"):
        (runs / n).mkdir(parents=True)
    st = op.OperatorState("home_dining", runs_dir=runs)
    for d in sorted(runs.iterdir()):
        st.remember(d)
    st.remember(sorted(runs.iterdir())[0])                       # remembering the same one twice does not duplicate
    st2 = op.OperatorState("home_dining", runs_dir=runs)          # restart the software
    assert [d.name for d in st2.live_history] == sorted(d.name for d in runs.iterdir() if d.is_dir())


def test_recorded_video_mode_ignores_software_job_history(site_tmp, tmp_path):
    """When recording the video (--job) the live job list is neither read nor written: Today / Reports only contain the recorded jobs, and the picture matches the video."""
    runs = tmp_path / "runs"
    d = runs / "sim-H5_dining_permanent-rules-cv-spot_arm-20261029-010000-000001"
    d.mkdir(parents=True)
    (runs / "software_jobs.txt").write_text(d.name + "\n")
    st = op.OperatorState("home_dining", object(), runs_dir=runs)
    assert st.live_history == []
    st.remember(d)
    assert st.live_history == [] and (runs / "software_jobs.txt").read_text() == d.name + "\n"


# regressions for the second round of trial-run findings
def test_job_start_refuses_negation_without_zones(site_tmp, tmp_path):
    """: A negation without a recognised zone: confirmation is required before the job starts; the whole surface is wiped only after confirming."""
    c, st = _client(tmp_path)
    r = c.post("/api/jobs/start", json={"instruction": "Don't touch the glass"})
    assert r.status_code == 409 and r.json().get("needs_confirm") is True
    p = c.get("/api/task/parse", params={"text": "avoid B3"}).json()
    assert p["mode"] == "except" and p["label"] == "All zones except B3"


def test_cross_site_guard(site_tmp, tmp_path):
    """: State-changing requests without JSON / the custom header, an unknown Host and a cross-origin Origin are all blocked."""
    c, st = _client(tmp_path)
    bare = TestClient(op.create_operator_app(st))
    assert bare.post("/api/jobs/estop", headers={"Content-Type": "text/plain"}).status_code == 403
    assert bare.post("/api/jobs/estop").status_code == 403
    assert c.get("/api/meta", headers={"Host": "evil.example:8092"}).status_code == 403
    assert c.get("/api/meta", headers={"Host": "127.0.0.1:8092"}).status_code == 200
    assert c.get("/api/meta", headers={"Host": "[::1]:8092"}).status_code == 200
    r = c.post("/api/jobs/estop", headers={"Host": "127.0.0.1:8092", "Origin": "http://evil.example"})
    assert r.status_code == 403
    r = c.post("/api/jobs/estop", headers={"Host": "127.0.0.1:8092", "Origin": "http://127.0.0.1:8092"})
    assert r.status_code != 403                              # same-origin passes (no job is running, so the business layer answers 409)
    ok = TestClient(op.create_operator_app(st, ("robot-pc.lan",)))
    assert ok.get("/api/meta", headers={"Host": "robot-pc.lan:8092"}).status_code == 200


def test_save_api_key_is_atomic_and_keeps_other_secrets(tmp_path):
    """: Atomic write: other keys in the same file are kept, permissions are 0600, no temporary file is left behind."""
    (tmp_path / "d").mkdir()
    f = tmp_path / "d" / "secrets.env"
    f.write_text("OTHER=keepme\nNEBIUS_API_KEY=old\n")
    op.save_api_key("new-key-123", f)
    txt = f.read_text()
    assert "OTHER=keepme" in txt and "NEBIUS_API_KEY=new-key-123" in txt and "old" not in txt
    assert oct(f.stat().st_mode & 0o777) == "0o600"
    assert [p.name for p in f.parent.iterdir()] == ["secrets.env"]


def test_rejected_key_does_not_overwrite_saved_key(site_tmp, tmp_path, monkeypatch, user_cfg_tmp):
    """: A mistyped key is rejected at verification and the old key stays as it was; it is written only if it passes."""
    f = op.secrets_file()
    f.write_text("NEBIUS_API_KEY=old-good\n")
    from cleanproof.ui import deploy_checks as dc
    monkeypatch.setattr(dc, "probe_api_key", lambda key, timeout=8.0: (False, "rejected"))
    c, st = _client(tmp_path)
    r = c.put("/api/settings", json={"api_key": "bad-key"})
    assert r.status_code == 400 and "rejected" in r.json()["detail"]
    assert f.read_text() == "NEBIUS_API_KEY=old-good\n"
    monkeypatch.setattr(dc, "probe_api_key", lambda key, timeout=8.0: (True, ""))
    assert c.put("/api/settings", json={"api_key": "good-key"}).status_code == 200
    assert f.read_text() == "NEBIUS_API_KEY=good-key\n"


def test_broken_site_config_gives_readable_error(site_tmp, tmp_path):
    """: Broken site config: the API gives the file and line, meta carries config_errors, a job is blocked before it starts, and nothing returns 500."""
    z = site_tmp / "home_dining" / "zones.yaml"
    z.write_text(z.read_text().replace("rows: 3", "rows: 0"))
    c, st = _client(tmp_path)
    m = c.get("/api/meta")
    assert m.status_code == 200 and m.json()["mode"] == "config_error"
    e = m.json()["config_errors"][0]
    assert e["file"].endswith("home_dining/zones.yaml") and e["line"] and "rows" in e["message"]
    r = c.get("/api/settings")
    assert r.status_code == 422 and "zones.yaml" in r.json()["detail"]
    assert c.post("/api/jobs/start", json={"instruction": "x"}).status_code == 422
    z.write_text("work_area: [unclosed\n")
    r = c.get("/api/settings")
    assert r.status_code == 422 and "line" in r.json()["detail"]
    (site_tmp / "home_dining" / "zones.yaml").write_text("")


def test_bad_job_scenario_and_negative_force_caught_before_start(site_tmp, tmp_path):
    s = site_tmp / "office_stairs" / "site.yaml"
    s.write_text(s.read_text().replace("job_scenario: ", "job_scenario: no_such_scenario # "))
    c, st = _client(tmp_path)
    r = c.post("/api/jobs/start", json={"site": "office_stairs", "instruction": "x"})
    assert r.status_code == 422 and "job_scenario" in r.json()["detail"]
    from cleanproof.ui.siteconfig import check_site
    ft = site_tmp / "factory_line" / "force_table.yaml"
    ft.write_text("surfaces:\n  steel:\n    max_n: -4\n")
    probs = check_site("factory_line", site_tmp / "factory_line")
    assert probs and probs[0]["line"] == 3 and "max_n" in probs[0]["message"]


def test_resolve_records_name_and_defaults_to_operator(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "summary.json").write_text(json.dumps({"human_requests": [{"zones": ["B3"], "reason": "x"}], "status": "needs_human"}))
    (rd / "events.jsonl").write_text(json.dumps({"kind": "task_end", "status": "needs_human"}) + "\n")
    assert op.resolve_request(rd, "0", "ok", "Operator A")["by"] == "Operator A"
    assert op.resolve_request(rd, "0", "ok")["by"] == "operator"


def test_interrupted_runs_are_found_and_prompted_once(site_tmp, tmp_path):
    """: A job started by the software without a summary = Interrupted; a notice is shown once when the next job starts."""
    runs = tmp_path / "runs"
    d = runs / "live-H5-20261030-104901-024312"
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text(json.dumps({"kind": "task_start", "ts": "2026-10-30T10:49:01+00:00", "targets": ["B3"]}) + "\n")
    (d / "reference_meta.json").write_text(json.dumps({"site": "home_dining"}))
    other = runs / "sim-S1-old"                      # run by a developer, not started by the software: not counted
    other.mkdir()
    (other / "events.jsonl").write_text("{}\n")
    done = runs / "live-done"
    done.mkdir()
    (done / "events.jsonl").write_text(json.dumps({"kind": "task_start"}) + "\n")
    (done / "summary.json").write_text("{}")
    (done / "reference_meta.json").write_text(json.dumps({"site": "home_dining"}))
    assert op.scan_interrupted(runs) == [d]
    c, st = _client(tmp_path)
    rows = c.get("/api/reports").json()["reports"]
    assert [r["status"] for r in rows if r["run_id"] == d.name] == ["interrupted"]
    first = c.post("/api/jobs/start", json={"instruction": "Clean the dining table", "site": "home_dining"})
    assert first.status_code == 200 and "interrupted" in first.json()["notice"]       # notice once, does not block the job
    assert not st.interrupt_pending
    st.live.stop()
    st.live.join(60)
    c2, st2 = _client(tmp_path)                       # start again: the notice is not repeated
    assert st2.interrupt_pending == []


def test_call_failure_labels():
    assert op.call_failure({"error": "AuthenticationError: Error code: 401", "http_status": 401}) == "failed · 401"
    assert op.call_failure({"error": "APIConnectionError: Connection error."}) == "failed · no connection"
    assert op.call_failure({"error": "BudgetExceeded: x", "error_kind": "budget"}) == "failed · over budget"
    assert op.call_failure({"error": "MissingApiKey: missing environment variable"}) == "failed · no key"
    assert op.call_failure({"latency_s": 1.0}) is None


def test_settings_follow_requested_site_and_real_limit(site_tmp, tmp_path):
    c, st = _client(tmp_path)
    s = c.get("/api/settings", params={"site": "office_stairs"}).json()
    assert s["site"] == "office_stairs" and s["real_limit"]["max_n"] == 5.0 and s["real_limit"]["active"] is False
    r = c.put("/api/settings", params={"site": "office_stairs"}, json={"jobs": {"max_passes_per_zone": 2}})
    assert r.status_code == 200 and r.json()["jobs"]["max_passes_per_zone"] == 2
    assert (site_tmp / "office_stairs" / "operator.yaml").is_file() and not (site_tmp / "home_dining" / "operator.yaml").is_file()


def test_settings_show_clamped_force_limits(site_tmp, tmp_path):
    """A force table edited by hand out of range (100 N): the settings page shows the clamped value in effect when a job starts, and says what the file contains."""
    ft = site_tmp / "home_dining" / "force_table.yaml"
    ft.write_text("surfaces:\n  sealed_wood:\n    max_n: 100\nescalate_step_n: 9\n")
    c, st = _client(tmp_path)
    s = c.get("/api/settings").json()
    w = next(x for x in s["surfaces"] if x["surface"] == "sealed_wood")
    assert w["max_n"] == 15.0 and w["clamped_from"] == 100.0 and w["clamp_note"] == "Clamped to 15 N (file had 100 N)"
    assert s["escalate"]["value"] == 5.0 and "file had 9 N" in s["escalate"]["clamp_note"]
    assert any("100 N" in n for n in s["limit_notes"])
