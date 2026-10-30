"""CleanProof desktop edition (Linux; "implement the software UI, supporting the Linux desktop is enough, the UI must look exactly like the video demo").

Starts the CleanProof service locally, as cleanproof.ui.operator in a simulated deployment where the robot computer is the built-in MuJoCo simulation, and opens the same software UI in a frameless native window using pywebview,
GTK WebKit2, with ?app=1 showing only the software window itself, a draggable title bar, and three dots = close / minimize / maximize.
Closing the window also stops the service. The UI is the same page and the same rendering as the software segment in the demo video (cleanproof/ui/static/operator.html).

Uses only the standard library + pywebview: runs in a separate environment of the system Python, where scripts/install/install_desktop.sh creates .venv-desktop that can see the system
python3-gi / WebKit2, while the service itself uses the project .venv.

  .venv-desktop/bin/python -m cleanproof.ui.desktop [--site home_dining] [--port 8097] [--policy auto] [--verifier auto] [--robot auto|sim|paired]

--policy auto: use Nemotron if this computer has a Nebius Token Factory key (planning costs about $0.003 per job; the Nemotron 3 Ultra audit, on by default, adds about $0.006 per job (59 recorded calls, at most $0.009); both count toward the $40 limit in the ledger), otherwise the rule planner;
--verifier auto: use Cosmos if the local Cosmos 3 inference service (local vLLM in configs/models.yaml) is up, otherwise use classical vision checks (shown honestly in the wizard).
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _repo_root() -> Path:
    """Resource root (same rule as common/config.py; only the standard library is used here, because the environment of the desktop window may not have the service dependencies)."""
    if os.environ.get("CLEANPROOF_HOME"):
        return Path(os.environ["CLEANPROOF_HOME"]).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    return checkout if (checkout / "pyproject.toml").is_file() and (checkout / "configs").is_dir() else Path(__file__).resolve().parents[1] / "_bundle"


def data_dir() -> "Path | None":
    v = os.environ.get("CLEANPROOF_DATA_DIR")
    return Path(v).expanduser() if v else None


REPO = _repo_root()
SECRETS = Path(os.environ.get("CLEANPROOF_CONFIG_DIR", "~/.config/cleanproof")).expanduser() / "secrets.env"
COSMOS_MODELS = "http://127.0.0.1:8001/v1/models"
WIN_W, WIN_H = 1568, 718                                  # design size of the software window (same as the window in the video), the UI scales proportionally with the window size


def config_dir() -> Path:
    """User config directory: $CLEANPROOF_CONFIG_DIR, otherwise ~/.config/cleanproof."""
    return Path(os.environ.get("CLEANPROOF_CONFIG_DIR") or "~/.config/cleanproof").expanduser()


def robot_mode(arg: str, cfg_dir: Path) -> str:
    """How the robot computer is attached: sim = built-in local simulation; paired = paired robot computer reached through an SSH tunnel.
    auto: paired if the config directory has a non-empty robot_host (written by install_edge.sh --pair), otherwise sim."""
    if arg in ("sim", "paired"):
        return arg
    try:
        return "paired" if (cfg_dir / "robot_host").read_text().strip() else "sim"
    except OSError:
        return "sim"


def server_cmd(python: str, site: str | None, port: int, policy: str, verifier: str, pace: float, mode: str,
               skill_url: str | None = None, camera_url: str | None = None) -> list[str]:
    """Command line of the operator service. sim adds --sim-robot (+ --pace); paired does not, the robot computer is reached through the tunnel (default 127.0.0.1:8100 / 8101)."""
    cmd = [python, "-m", "cleanproof.ui.operator", "--port", str(port), "--policy", policy, "--verifier", verifier]
    if site:
        cmd += ["--site", site]
    if mode == "sim":
        cmd += ["--sim-robot", "--pace", str(pace)]
    if skill_url:
        cmd += ["--skill-url", skill_url]
    if camera_url:
        cmd += ["--camera-url", camera_url]
    return cmd


def have_key() -> bool:
    """Only check whether a key is set; never read out or print the key value."""
    if os.environ.get("NEBIUS_API_KEY"):
        return True
    try:
        return any(ln.startswith("NEBIUS_API_KEY=") and ln.strip() != "NEBIUS_API_KEY=" for ln in SECRETS.read_text().splitlines())
    except OSError:
        return False


def cosmos_up(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(COSMOS_MODELS, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_up(port: int, proc: subprocess.Popen, timeout: float = 90.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/meta", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.4)
    return False


class WindowApi:
    """Window buttons for the page (wireApp in operator.html: the three dots in the title bar)."""

    def __init__(self) -> None:
        self.window = None
        self._max = False

    def close(self) -> None:
        self.window.destroy()

    def minimize(self) -> None:
        self.window.minimize()

    def toggle_maximize(self) -> None:
        if self._max:
            self.window.restore()
        else:
            self.window.maximize()
        self._max = not self._max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=os.environ.get("CLEANPROOF_SITE") or None, help="if omitted, the service uses the site chosen in the wizard (setup.json)")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--policy", default="auto", choices=("auto", "rules", "nemotron"))
    ap.add_argument("--verifier", default="auto", choices=("auto", "cv", "vlm"))
    ap.add_argument("--pace", type=float, default=3.0, help="simulation pace (times real time): 3 = about half a minute per job with smooth motion in the camera (the software segment in the video is compressed the same way)")
    ap.add_argument("--robot", default="auto", choices=("auto", "sim", "paired"),
                    help="auto: connect to the paired robot computer (paired) if the config directory has robot_host, otherwise use the built-in local simulation (sim)")
    ap.add_argument("--skill-url", default=None, help="paired: skill server address (default http://127.0.0.1:8100, through the SSH tunnel)")
    ap.add_argument("--camera-url", default=None, help="paired: camera service address (default http://127.0.0.1:8101)")
    ap.add_argument("--server-python", default=str(REPO / ".venv" / "bin" / "python") if (REPO / ".venv" / "bin" / "python").is_file() else sys.executable,
                    help="Python that runs the service (default: .venv in the repository, otherwise the current one)")
    ap.add_argument("--no-window", action="store_true", help="start only the service and print the address (for debugging / screenshots)")
    args = ap.parse_args()
    policy = args.policy if args.policy != "auto" else ("nemotron" if have_key() else "rules")
    verifier = args.verifier if args.verifier != "auto" else ("vlm" if cosmos_up() else "cv")
    port = args.port
    while not port_free(port):
        port += 1
    log_dir = Path(os.environ.get("CLEANPROOF_STATE_DIR") or "~/.local/state/cleanproof").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    mode = robot_mode(args.robot, config_dir())
    print(f"CleanProof robot computer: {mode}" + (" (SSH tunnel)" if mode == "paired" else " (built-in simulation)"), flush=True)
    if mode == "paired":
        from urllib.parse import urlparse
        sk = urlparse(args.skill_url or "http://127.0.0.1:8100")
        if port_free(sk.port or 80) and sk.hostname in ("127.0.0.1", "localhost"):
            print(f"Note: tunnel port {sk.port} is not responding and the robot computer cannot be reached yet; start the tunnel first: systemctl --user start cleanproof-tunnel"
                  "(or scripts/robot_pc/tunnel.sh)", flush=True)
    cmd = server_cmd(args.server_python, args.site, port, policy, verifier, args.pace, mode, args.skill_url, args.camera_url)
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")                                 # rendering happens in the simulation / request threads: EGL does not require the main thread (GLFW does)
    store = data_dir() / "mujoco_menagerie" if data_dir() else REPO / "store" / "mujoco_menagerie"   # robot model cache location (one of platforms.MENAGERIE_DEFAULTS)
    if "CLEANPROOF_MENAGERIE" not in env and store.is_dir():
        env["CLEANPROOF_MENAGERIE"] = str(store)
    with open(log_dir / "desktop_server.log", "ab") as log:
        srv = subprocess.Popen(cmd, cwd=str(data_dir() or REPO), stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_up(port, srv):
            print(f"CleanProof service did not start, see {log_dir / 'desktop_server.log'}", file=sys.stderr)
            return 1
        url = f"http://127.0.0.1:{port}/?app=1"
        print("server:", " ".join(cmd), flush=True)
        print(f"CleanProof · {url} · planner {policy} · verifier {verifier} · robot {mode}", flush=True)
        if args.no_window:
            srv.wait()
            return 0
        import webview                                        # pywebview (GTK WebKit2): only needed in a desktop environment
        api = WindowApi()
        win = webview.create_window("CleanProof", url, width=WIN_W, height=WIN_H, min_size=(980, 449), frameless=True, easy_drag=False,
                                    js_api=api, background_color="#FBFAF8")
        api.window = win
        webview.start(gui="gtk", icon=str(REPO / "assets" / "brand_icons" / "cleanproof_app.png"))   # window / taskbar icon = app icon (lion badge)
        return 0
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.kill()


if __name__ == "__main__":
    raise SystemExit(main())
