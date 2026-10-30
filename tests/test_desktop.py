"""Desktop launcher: robot mode selection and the service command line (no window opened, no service started)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cleanproof.ui import desktop


def test_robot_mode_explicit(tmp_path: Path) -> None:
    (tmp_path / "robot_host").write_text("robot-pc\n")
    assert desktop.robot_mode("sim", tmp_path) == "sim"
    assert desktop.robot_mode("paired", tmp_path / "none") == "paired"


def test_robot_mode_auto(tmp_path: Path) -> None:
    assert desktop.robot_mode("auto", tmp_path) == "sim"                 # no robot_host
    (tmp_path / "robot_host").write_text("  \n")
    assert desktop.robot_mode("auto", tmp_path) == "sim"                 # empty file
    (tmp_path / "robot_host").write_text("robot-pc\n")
    assert desktop.robot_mode("auto", tmp_path) == "paired"
    assert desktop.robot_mode("auto", tmp_path / "missing_dir") == "sim"


def test_config_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLEANPROOF_CONFIG_DIR", str(tmp_path))
    assert desktop.config_dir() == tmp_path
    monkeypatch.delenv("CLEANPROOF_CONFIG_DIR")
    assert desktop.config_dir() == Path("~/.config/cleanproof").expanduser()


def test_server_cmd_sim() -> None:
    cmd = desktop.server_cmd("py", "home_dining", 8097, "rules", "cv", 3.0, "sim")
    assert cmd[:3] == ["py", "-m", "cleanproof.ui.operator"]
    assert "--sim-robot" in cmd and cmd[cmd.index("--pace") + 1] == "3.0"
    assert "--skill-url" not in cmd


def test_server_cmd_paired() -> None:
    cmd = desktop.server_cmd("py", "ur3e_site", 8097, "nemotron", "vlm", 3.0, "paired")
    assert "--sim-robot" not in cmd and "--pace" not in cmd
    assert "--skill-url" not in cmd and "--camera-url" not in cmd         # defaults to 127.0.0.1:8100 / 8101
    cmd = desktop.server_cmd("py", "ur3e_site", 8097, "rules", "cv", 3.0, "paired", "http://127.0.0.1:9100", "http://127.0.0.1:9101")
    assert cmd[cmd.index("--skill-url") + 1] == "http://127.0.0.1:9100"
    assert cmd[cmd.index("--camera-url") + 1] == "http://127.0.0.1:9101"


def test_server_cmd_site_only_when_given() -> None:
    assert "--site" not in desktop.server_cmd("py", None, 8097, "rules", "cv", 3.0, "sim")
    cmd = desktop.server_cmd("py", "ur3e_site", 8097, "rules", "cv", 3.0, "sim")
    assert cmd[cmd.index("--site") + 1] == "ur3e_site"
