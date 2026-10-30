"""User data directory: with CLEANPROOF_DATA_DIR set, site configs and run records go to the user directory and the resource root (install directory) stays read-only."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from cleanproof.common import config


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_DATA_DIR, str(tmp_path / "data"))
    monkeypatch.delenv(config.ENV_SITE, raising=False)
    return tmp_path / "data"


def test_default_keeps_old_layout(monkeypatch):
    monkeypatch.delenv(config.ENV_DATA_DIR, raising=False)
    assert config.data_dir() is None
    assert config.site_root() == config.CONFIG_DIR / "site"
    assert config.runs_root() == config.REPO_ROOT / "runs"


def test_runs_and_sites_move_to_the_data_dir(data_root):
    assert config.runs_root() == data_root / "runs"
    root = config.site_root()
    assert root == data_root / "site"
    bundled = sorted(p.name for p in (config.CONFIG_DIR / "site").iterdir() if p.is_dir())
    assert bundled and sorted(p.name for p in root.iterdir()) == bundled
    assert config.site_dir("home_dining") == root / "home_dining"


def test_seeding_never_overwrites_user_edits(data_root):
    config.site_root()
    edited = data_root / "site" / "home_dining" / "site.yaml"
    edited.write_text("edited: true\n", encoding="utf-8")
    config._seed_sites.cache_clear()
    config.site_root()
    assert edited.read_text(encoding="utf-8") == "edited: true\n"


def test_site_overlay_is_read_from_the_user_copy_and_the_install_dir_is_untouched(data_root):
    bundled = config.CONFIG_DIR / "site" / "home_dining" / "force_table.yaml"
    before = bundled.read_bytes() if bundled.is_file() else None
    config.site_root()
    mine = data_root / "site" / "home_dining" / "zones.yaml"
    base = config.load_yaml("zones.yaml", site="")
    mine.write_text("z_surface: 0.123\n", encoding="utf-8")
    over = config.load_yaml("zones.yaml", site="home_dining")
    assert over["z_surface"] == 0.123 and over != base
    assert (config.CONFIG_DIR / "site" / "home_dining" / "zones.yaml").read_text(encoding="utf-8") != "z_surface: 0.123\n"
    assert before is None or bundled.read_bytes() == before


def test_use_site_checks_the_user_site_root(data_root):
    with pytest.raises(FileNotFoundError):
        config.use_site("no_such_site")
    config.use_site("home_dining")
    try:
        assert config.current_site() == "home_dining"
    finally:
        config.use_site("")


def test_resource_root_can_be_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ENV_HOME, str(tmp_path))
    assert config._resolve_root() == tmp_path.resolve()
    monkeypatch.delenv(config.ENV_HOME)
    assert (config._resolve_root() / "configs").is_dir()


def test_run_log_and_llm_log_default_to_the_runs_root(data_root):
    from cleanproof.agent.runlog import RunLogger
    from cleanproof.common.llm_client import LLMClient

    rl = RunLogger(run_id="r1")
    assert Path(rl.dir) == data_root / "runs" / "r1"
    assert LLMClient(config={"roles": {}, "budget": {"ledger_path": str(data_root / "ledger.json")}}).log_path == data_root / "runs" / "llm_calls.jsonl"


def test_launcher_prepares_the_user_data_dir(monkeypatch, tmp_path):
    from cleanproof import launcher

    monkeypatch.setenv(config.ENV_DATA_DIR, "x")           # set, then delete: only then does teardown clear the value prepare_env writes
    monkeypatch.delenv(config.ENV_DATA_DIR)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("MUJOCO_GL", "x")
    monkeypatch.delenv("MUJOCO_GL")
    monkeypatch.chdir(tmp_path)
    data = launcher.prepare_env()
    try:
        assert data == tmp_path / "xdg" / "cleanproof" and data.is_dir()
        assert os.environ[config.ENV_DATA_DIR] == str(data)
        assert Path.cwd() == data
    finally:
        os.chdir(tmp_path)


def test_launcher_routes_commands(monkeypatch, tmp_path):
    from cleanproof import launcher

    monkeypatch.setenv(config.ENV_DATA_DIR, str(tmp_path / "d"))
    monkeypatch.setenv("MUJOCO_GL", "x")
    monkeypatch.delenv("MUJOCO_GL")                        # prepare_env sets a default; it has to be cleared at teardown as well
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(launcher, "_run", lambda module, argv: seen.append((module, argv)) or 0)
    assert launcher.main([]) == 0 and launcher.main(["web"]) == 0 and launcher.main(["web", "--port", "9"]) == 0
    assert launcher.main(["--site", "x"]) == 0 and launcher.main(["bogus"]) == 2
    assert seen == [("cleanproof.ui.desktop", []), ("cleanproof.ui.operator", ["--sim-robot", "--pace", "3"]),
                    ("cleanproof.ui.operator", ["--port", "9"]), ("cleanproof.ui.desktop", ["--site", "x"])]
