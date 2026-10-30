"""Entry point `cleanproof` of an installed copy: puts user data in the home directory, then starts the desktop app or the browser version.

  cleanproof [desktop] [desktop options]      desktop window (default)
  cleanproof web [operator options]         browser version (built-in simulated robot at 3x by default; options given are passed to the operator as they are)
  cleanproof fetch-menagerie             download MuJoCo Menagerie (the models of the non-UR3e robots) into the user data directory
  cleanproof --version

The install directory (/opt/cleanproof) is read-only. Site configs and run records go to $CLEANPROOF_DATA_DIR (default ~/.local/share/cleanproof),
keys and settings to $CLEANPROOF_CONFIG_DIR (default ~/.config/cleanproof), the spending ledger and other state to $CLEANPROOF_STATE_DIR (default ~/.local/state/cleanproof).
"""
from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie"
USAGE = __doc__


def default_data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or "~/.local/share").expanduser() / "cleanproof"


def prepare_env() -> Path:
    """Set the environment variables for the user directories, create the data directory and change into it (MuJoCo and others write logs to the current directory). Returns the data directory."""
    data = Path(os.environ.setdefault("CLEANPROOF_DATA_DIR", str(default_data_dir()))).expanduser()
    data.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.chdir(data)
    return data


def version() -> str:
    try:
        from importlib.metadata import version as _v
        return _v("cleanproof")
    except Exception:  # noqa: BLE001
        return "unknown"


def fetch_menagerie(data: Path) -> int:
    dest = data / "mujoco_menagerie"
    if dest.is_dir() and any(dest.iterdir()):
        print(f"already there: {dest}")
        return 0
    print(f"cloning MuJoCo Menagerie (latest snapshot) into {dest}", flush=True)
    try:
        return subprocess.call(["git", "clone", "--depth", "1", MENAGERIE_URL, str(dest)])
    except FileNotFoundError:
        print("git is not installed: sudo apt install git", file=sys.stderr)
        return 1


def _run(module: str, argv: list[str]) -> int:
    sys.argv = [module] + argv
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if args and args[0] in ("--version", "version"):
        print(f"cleanproof {version()}")
        return 0
    data = prepare_env()
    cmd = args[0] if args and not args[0].startswith("-") else "desktop"
    rest = args[1:] if args and not args[0].startswith("-") else args
    if cmd == "fetch-menagerie":
        return fetch_menagerie(data)
    if cmd == "desktop":
        return _run("cleanproof.ui.desktop", rest)
    if cmd == "web":
        return _run("cleanproof.ui.operator", rest or ["--sim-robot", "--pace", "3"])
    print(f"unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
