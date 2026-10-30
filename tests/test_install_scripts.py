"""Install scripts (scripts/install/): --dry-run only prints and writes no files; the key steps are all there; the skill server is only installed in shadow mode."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = {"edge": REPO / "scripts/install/install_edge.sh", "robot": REPO / "scripts/install/install_robot.sh",
           "desktop": REPO / "scripts/install/install_desktop.sh", "tunnel": REPO / "scripts/robot_pc/tunnel.sh",
           "push_site": REPO / "scripts/robot_pc/push_site.sh"}


def _run(script: Path, *args: str, home: Path, **extra_env: str) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "XDG_CONFIG_HOME": str(home / ".config"), **extra_env}
    return subprocess.run(["bash", str(script), *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=60)


def _files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


@pytest.mark.parametrize("name", sorted(SCRIPTS))
def test_syntax(name: str) -> None:
    assert subprocess.run(["bash", "-n", str(SCRIPTS[name])]).returncode == 0
    if shutil.which("shellcheck"):
        r = subprocess.run(["shellcheck", "-S", "warning", str(SCRIPTS[name])], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout


def test_edge_dry_run_writes_nothing(tmp_path: Path) -> None:
    r = _run(SCRIPTS["edge"], "--dry-run", "--skip-model", home=tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for s in ("[1/6]", "[6/6]", "pip install -q -e .[ui]", "NEBIUS_API_KEY: MISSING", "cleanproof-console.service",
              "cleanproof.ui.operator", "serve_cosmos.sh"):
        assert s in out, s
    assert _files(tmp_path) == []
    assert "token_urlsafe" not in out or "generate 32-byte random token" in out


def test_edge_pair_dry_run(tmp_path: Path) -> None:
    r = _run(SCRIPTS["edge"], "--pair", "robot-pc", "--dry-run", home=tmp_path)
    assert r.returncode == 0, r.stderr
    for s in ("rsync -a --chmod=F600", "robot-pc:.config/cleanproof/robot_token", "cleanproof-tunnel.service",
              "ROBOT_PC_HOST=robot-pc", "scripts/robot_pc/tunnel.sh", "cleanproof.ui.deploy_checks"):
        assert s in r.stdout, s
    assert _files(tmp_path) == []


def test_edge_pair_env_and_no_start(tmp_path: Path) -> None:
    """Pairing: CP_SSH_OPTS is used for both ssh and rsync, the remote config directory can be changed, --no-start does not enable the tunnel, and the ports are written into the service file and passed to the deployment check."""
    r = _run(SCRIPTS["edge"], "--pair", "robot-pc", "--no-start", "--dry-run", home=tmp_path, CP_SSH_OPTS="-F /x/ssh_config",
             CP_REMOTE_CONFIG_DIR="/r/cfg", SKILL_PORT="9100", CAMERA_PORT="9101", REMOTE_SKILL_PORT="18100", REMOTE_CAMERA_PORT="18101")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for s in ("+ ssh -F /x/ssh_config robot-pc", 'rsync -a --chmod=F600 -e ssh -F /x/ssh_config', "robot-pc:/r/cfg/robot_token",
              "Environment=SKILL_PORT=9100", "Environment=CAMERA_PORT=9101", "Environment=REMOTE_SKILL_PORT=18100",
              "Environment=REMOTE_CAMERA_PORT=18101", "-F /x/ssh_config", "tunnel not started (--no-start)",
              "--skill-url http://127.0.0.1:9100", "--camera-url http://127.0.0.1:9101"):
        assert s in out, s
    assert "enable --now cleanproof-tunnel" in out.split("tunnel not started")[1]          # only appears in the "To start it:" hint
    assert "+ systemctl" not in out
    assert _files(tmp_path) == []


def test_edge_pair_default_ports(tmp_path: Path) -> None:
    out = _run(SCRIPTS["edge"], "--pair", "robot-pc", "--dry-run", home=tmp_path).stdout
    for s in ("Environment=SKILL_PORT=8100", "Environment=REMOTE_CAMERA_PORT=8101", "robot-pc:.config/cleanproof/robot_token",
              "--skill-url http://127.0.0.1:8100", "+ systemctl --user enable --now cleanproof-tunnel.service"):
        assert s in out, s
    assert "edge computer" not in out


def test_push_site_dry_run(tmp_path: Path) -> None:
    r = _run(SCRIPTS["push_site"], "home_dining", "--restart", "--dry-run", home=tmp_path, ROBOT_PC_HOST="robot-pc", CP_SSH_OPTS="-F /x/cfg")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for s in ("rsync -az", "-F /x/cfg", "robot-pc:cleanproof/configs/site/home_dining/", "site.conf", "CLEANPROOF_SITE=%s",
              "cleanproof-skill", "cleanproof-camera", "daemon-reload", "restart"):
        assert s in out, s
    assert "arm" not in out.replace("farm", "")                                  # not armed
    assert _run(SCRIPTS["push_site"], "no_such_site", "--dry-run", home=tmp_path).returncode == 1
    assert _run(SCRIPTS["push_site"], "../x", "--dry-run", home=tmp_path).returncode == 2
    assert "--restart" not in _run(SCRIPTS["push_site"], "home_dining", "--dry-run", home=tmp_path).stdout


def test_tunnel_env_options(tmp_path: Path) -> None:
    """tunnel.sh turns the ports and ssh options into environment variables: record the arguments with a fake ssh."""
    fake = tmp_path / "bin"
    fake.mkdir()
    log = tmp_path / "args.txt"
    (fake / "ssh").write_text(f'#!/bin/bash\necho "$@" > {log}\nkill -TERM $PPID\n')
    (fake / "ssh").chmod(0o755)
    env = {"HOME": str(tmp_path), "PATH": f"{fake}:{os.environ['PATH']}", "ROBOT_PC_HOST": "robot-pc", "CP_SSH_OPTS": "-F /x/cfg",
           "SKILL_PORT": "9100", "CAMERA_PORT": "9101", "REMOTE_SKILL_PORT": "18100"}
    subprocess.run(["bash", str(SCRIPTS["tunnel"])], env=env, capture_output=True, timeout=20)
    args = log.read_text()
    assert "-F /x/cfg" in args and "-L 127.0.0.1:9100:127.0.0.1:18100" in args and "-L 127.0.0.1:9101:127.0.0.1:9101" in args and args.strip().endswith("robot-pc")


@pytest.mark.parametrize("backend", ["ur3e", "mujoco"])
def test_robot_dry_run_shadow_only(tmp_path: Path, backend: str) -> None:
    r = _run(SCRIPTS["robot"], "--backend", backend, "--dry-run", home=tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for s in ("[1/5]", "[5/5]", "cleanproof-skill.service", "--bind 127.0.0.1", f"--backend {backend} --mode shadow",
              "robot_token: MISSING"):
        assert s in out, s
    assert ("cleanproof-camera.service" in out) == (backend == "ur3e")
    assert ("--pace 3" in out and "--scenario S1_light" in out) == (backend == "mujoco")
    assert "edge computer" not in out
    assert "--mode full" not in out and "--mode dry_run" not in out and "commissioning arm" not in out
    assert _files(tmp_path) == []


def test_robot_rejects_unknown_backend(tmp_path: Path) -> None:
    assert _run(SCRIPTS["robot"], "--backend", "spot", "--dry-run", home=tmp_path).returncode == 2


def test_desktop_dry_run_writes_nothing(tmp_path: Path) -> None:
    """Desktop install: --dry-run only prints the three steps and the launcher content, creates no environment and writes no files; the launcher points to cleanproof.ui.desktop and the app icon."""
    r = _run(SCRIPTS["desktop"], "--dry-run", "--site", "home_dining", home=tmp_path)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    for s in ("[1/3]", "[2/3]", "[3/3]", "GTK WebKit2:", "pip install -q pywebview", "cleanproof.desktop", "-m cleanproof.ui.desktop",
              "assets/brand_icons/cleanproof_app.png"):
        assert s in out, s
    assert "CLEANPROOF_SITE" not in out.split("Exec=")[1].splitlines()[0]     # the launcher does not pin a site (--site is ignored with a note)
    assert _files(tmp_path) == []
    assert (REPO / "assets/brand_icons/cleanproof_app.png").is_file()


def test_config_dir_in_units_and_ssh_opts(tmp_path: Path) -> None:
    """Custom config directory: the console / tunnel / robot-side services all carry CLEANPROOF_CONFIG_DIR; the ssh options used for pairing are stored in the config directory for push_site / tunnel to read."""
    cfg = tmp_path / "cfg"
    out = _run(SCRIPTS["edge"], "--pair", "robot-pc", "--dry-run", home=tmp_path, CLEANPROOF_CONFIG_DIR=str(cfg), CP_SSH_OPTS="-F /x/c").stdout
    assert f'Environment="CLEANPROOF_CONFIG_DIR={cfg}"' in out and f"{cfg}/ssh_opts" in out and "-F /x/c" in out
    out = _run(SCRIPTS["edge"], "--dry-run", "--skip-model", home=tmp_path, CLEANPROOF_CONFIG_DIR=str(cfg)).stdout
    assert f'Environment="CLEANPROOF_CONFIG_DIR={cfg}"' in out
    out = _run(SCRIPTS["robot"], "--backend", "mujoco", "--dry-run", home=tmp_path, CLEANPROOF_CONFIG_DIR=str(cfg)).stdout
    assert f'Environment="CLEANPROOF_CONFIG_DIR={cfg}"' in out
    assert "CLEANPROOF_CONFIG_DIR" not in _run(SCRIPTS["robot"], "--dry-run", home=tmp_path).stdout
    # push_site reads ssh_opts from the config directory
    cfg.mkdir()
    (cfg / "ssh_opts").write_text("-F /from/file\n")
    out = _run(SCRIPTS["push_site"], "home_dining", "--dry-run", home=tmp_path, CLEANPROOF_CONFIG_DIR=str(cfg), ROBOT_PC_HOST="rp").stdout
    assert "-F /from/file" in out and ".site_backups/home_dining-" in out


def test_help_and_missing_values(tmp_path: Path) -> None:
    for name in ("edge", "robot", "push_site"):
        r = _run(SCRIPTS[name], "-h", home=tmp_path)
        assert r.returncode == 0 and r.stdout.startswith(("GPU workstation", "Robot computer", "GPU")) and "set -e" not in r.stdout
    for name, opt in (("edge", "--pair"), ("edge", "--site"), ("robot", "--backend")):
        r = _run(SCRIPTS[name], opt, home=tmp_path)
        assert r.returncode == 2 and "needs a value" in r.stderr and "unbound" not in r.stderr


def test_desktop_launcher_never_pins_site(tmp_path: Path) -> None:
    """The launcher never writes CLEANPROOF_SITE (a site changed in the wizard is kept after a restart via setup.json); --site is ignored too."""
    for extra in ((), ("--site", "home_dining")):
        out = _run(SCRIPTS["desktop"], "--dry-run", *extra, home=tmp_path).stdout
        assert "Exec=" + str(REPO) in out and "CLEANPROOF_SITE" not in out


def test_edge_console_unit_site_only_when_given(tmp_path: Path) -> None:
    """Console service: without --site the site is not pinned (the choice in the wizard still holds after a restart)."""
    out = _run(SCRIPTS["edge"], "--dry-run", "--skip-model", home=tmp_path).stdout
    assert "cleanproof.ui.operator --policy nemotron" in out and "CLEANPROOF_SITE" not in out and "--site" not in out
    out = _run(SCRIPTS["edge"], "--dry-run", "--skip-model", "--site", "ur3e_site", home=tmp_path).stdout
    assert "Environment=CLEANPROOF_SITE=ur3e_site" in out and "cleanproof.ui.operator --site ur3e_site --policy" in out
