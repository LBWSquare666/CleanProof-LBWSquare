#!/usr/bin/env python3
"""End-to-end self-test of paired deployment for the desktop edition (everything in temp dirs, never touching the real ~/.config, ~/.ssh or systemd).

Flow: temporary sshd (127.0.0.1, random port) → two temporary config dirs W (GPU workstation) / R (robot computer) → start the skill server on the robot side (mujoco backend,
pace 3) → install_edge.sh --pair really copies the token over SSH and writes robot_host → tunnel.sh maps 8100 / 8101 across → desktop edition --robot auto (must pick paired)
→ click through the whole flow in the UI with desktop_parity_shots.py --paired → collect evidence into runs/desktop_paired/ → tear everything down (also on failure).

  .venv/bin/python scripts/desktop_paired_selftest.py [--plumbing-only] [--memory 3G]

By default wraps itself in flock + systemd-run --scope (memory limit --memory); --no-wrap turns this off (when already wrapped outside).
--plumbing-only: verify only a-f (sshd, pairing, tunnel, desktop picks paired, the three wizard checks), skip the UI clicks (g).
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCK = os.environ.get("CP_HEAVY_LOCK", "/tmp/cleanproof_heavy.lock")
sys.path.insert(0, str(REPO / "scripts"))
import desktop_parity_shots as _parity  # noqa: E402  snapshot / restore configs/site

PY = str(REPO / ".venv" / "bin" / "python")
SIM_RUNS = REPO / "runs"


def free_port(prefer: int | None = None) -> int:
    for p in ([prefer] if prefer else []) + [0]:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
            except OSError:
                continue
            return s.getsockname()[1]
    raise RuntimeError("no free port")


def port_used(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def http(url: str, token: str | None = None, method: str = "GET", body: dict | None = None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if token:
        req.add_header("X-CleanProof-Token", token)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, None


class Selftest:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.ev = Path(a.evidence).resolve()
        self.scratch = Path(tempfile.mkdtemp(prefix="cp_selftest_"))
        self.procs: list[subprocess.Popen] = []
        self.checks: dict[str, dict] = {}
        self.t0 = time.time()
        self.new_runs: list[str] = []

    # ---- recording ----
    def check(self, name: str, ok: bool | None, detail: str = "") -> bool:
        self.checks[name] = {"ok": ok, "detail": detail}
        print(f"[{'PASS' if ok else ('SKIP' if ok is None else 'FAIL')}] {name}  {detail}", flush=True)
        return bool(ok)

    def spawn(self, name: str, cmd: list[str], env: dict[str, str] | None = None, log: Path | None = None, cwd: Path = REPO) -> subprocess.Popen:
        f = open(log or (self.ev / f"{name}.log"), "ab")
        p = subprocess.Popen(cmd, cwd=str(cwd), env={**os.environ, **(env or {})}, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        self.procs.append(p)
        return p

    def sh(self, cmd: list[str], env: dict[str, str] | None = None, timeout: float = 120.0) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=str(REPO), env={**os.environ, **(env or {})}, capture_output=True, text=True, timeout=timeout)

    def wait_http(self, url: str, p: subprocess.Popen | None, timeout: float, token: str | None = None) -> bool:
        t = time.time()
        while time.time() - t < timeout:
            if p is not None and p.poll() is not None:
                return False
            try:
                if http(url, token, timeout=2)[0] == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        return False

    def teardown(self) -> None:
        for p in reversed(self.procs):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        t = time.time()
        while time.time() - t < 8 and any(p.poll() is None for p in self.procs):
            time.sleep(0.2)
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        shutil.rmtree(self.scratch, ignore_errors=True)

    # ---- main flow ----
    def run(self) -> int:
        a, S = self.a, self.scratch
        if self.ev.exists():                                   # move the previous evidence into prev_<time>/, do not delete it
            old = [x for x in self.ev.iterdir() if not x.name.startswith("prev_")]
            if old:
                prev = self.ev / time.strftime("prev_%Y%m%d-%H%M%S")
                prev.mkdir()
                for x in old:
                    shutil.move(str(x), str(prev / x.name))
        self.ev.mkdir(parents=True, exist_ok=True)
        for pt in (8100, 8101):
            if port_used(pt):
                print(f"port {pt} is already in use: aborting (the tunnel needs 8100 / 8101)", file=sys.stderr)
                return 3
        before_runs = {p.name for p in SIM_RUNS.glob("live-*")}
        W, R = S / "W", S / "R"
        for d in (W, R):
            d.mkdir(mode=0o700)
        # (a) temporary sshd
        sport = free_port()
        for n in ("id", "hostkey"):
            self.sh(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(S / n)])
        shutil.copy(S / "id.pub", S / "authorized_keys")
        os.chmod(S / "authorized_keys", 0o600)
        (S / "sshd_config").write_text(f"Port {sport}\nListenAddress 127.0.0.1\nHostKey {S / 'hostkey'}\nAuthorizedKeysFile {S / 'authorized_keys'}\n"
                                       f"UsePAM no\nStrictModes no\nPasswordAuthentication no\nPidFile {S / 'sshd.pid'}\n")
        (S / "ssh_config").write_text(f"Host robot-pc\n  HostName 127.0.0.1\n  Port {sport}\n  IdentityFile {S / 'id'}\n  IdentitiesOnly yes\n"
                                      f"  UserKnownHostsFile {S / 'known_hosts'}\n  StrictHostKeyChecking accept-new\n  BatchMode yes\n")
        ssh_opts = f"-F {S / 'ssh_config'}"
        self.spawn("sshd", ["/usr/sbin/sshd", "-D", "-e", "-f", str(S / "sshd_config")])
        t = time.time()
        ok = False
        while time.time() - t < 10 and not ok:
            ok = self.sh(["ssh", *ssh_opts.split(), "robot-pc", "echo", "ok"]).stdout.strip() == "ok"
            time.sleep(0.3)
        if not self.check("sshd_login", ok, f"temporary sshd 127.0.0.1:{sport}, login with a temporary key"):
            return 1
        # (b) token
        token = secrets.token_urlsafe(32)
        (W / "robot_token").write_text(token)
        os.chmod(W / "robot_token", 0o600)
        # (c) robot side: skill server (mujoco backend)
        rs, rc = free_port(18100), free_port(18101)
        skill_dir = self.ev / "skill_server"
        skill_dir.mkdir()
        srv = self.spawn("skill_server_stdout", [PY, "-m", "cleanproof.robot.server", "--backend", "mujoco", "--scenario", "H5_dining_permanent",
                                                 "--pace", "3", "--port", str(rs), "--log-dir", str(skill_dir)],
                         env={"CLEANPROOF_CONFIG_DIR": str(R), "MUJOCO_GL": "egl"}, log=skill_dir / "server.log")
        if not self.check("robot_skill_server_up", self.wait_http(f"http://127.0.0.1:{rs}/health", srv, 90, token), f"mujoco backend 127.0.0.1:{rs}"):
            return 1
        # (d) pairing: really copy the token over SSH
        env_pair = {"CLEANPROOF_CONFIG_DIR": str(W), "XDG_CONFIG_HOME": str(S / "xdg"), "CP_SSH_OPTS": ssh_opts, "CP_REMOTE_CONFIG_DIR": str(R),
                    "SKILL_PORT": "8100", "CAMERA_PORT": "8101", "REMOTE_SKILL_PORT": str(rs), "REMOTE_CAMERA_PORT": str(rc)}
        r = self.sh(["bash", "scripts/install/install_edge.sh", "--pair", "robot-pc", "--no-start"], env_pair)
        (self.ev / "pair.log").write_text(r.stdout + r.stderr)
        same = (R / "robot_token").is_file() and (R / "robot_token").read_text() == token
        mode_ok = same and oct((R / "robot_token").stat().st_mode & 0o777) == "0o600"
        self.check("pair_exit0", r.returncode == 0, f"install_edge.sh --pair --no-start exit code {r.returncode}")
        self.check("pair_token_copied_0600", mode_ok, "robot_token copied to the robot side over SSH, same content (not printed), permissions 0600")
        host = (W / "robot_host").read_text().strip() if (W / "robot_host").is_file() else ""
        self.check("pair_robot_host_written", host == "robot-pc", f"W/robot_host = {host!r}")
        unit = S / "xdg" / "systemd" / "user" / "cleanproof-tunnel.service"
        self.check("pair_unit_written_not_started", unit.is_file() and "REMOTE_SKILL_PORT=%d" % rs in unit.read_text(), "tunnel service file written (not started)")
        # (e) tunnel
        tun = self.spawn("tunnel", ["bash", "scripts/robot_pc/tunnel.sh"], env={"ROBOT_PC_HOST": "robot-pc", "CP_SSH_OPTS": ssh_opts, "SKILL_PORT": "8100",
                                                                          "CAMERA_PORT": "8101", "REMOTE_SKILL_PORT": str(rs), "REMOTE_CAMERA_PORT": str(rc)})
        up = self.wait_http("http://127.0.0.1:8100/health", tun, 30, token)
        code, h = http("http://127.0.0.1:8100/health", token) if up else (0, None)
        self.check("tunnel_health", up and code == 200, f"GET 127.0.0.1:8100/health through the tunnel → {code} backend={(h or {}).get('backend')} label={(h or {}).get('label')}")
        if not up:
            return 1
        # (f) desktop edition (no window): auto must pick paired
        oport = free_port(8097)
        (self.ev / "operator_state").mkdir()
        op = self.spawn("desktop_stdout", [PY, "-m", "cleanproof.ui.desktop", "--robot", "auto", "--no-window", "--port", str(oport),
                                           "--site", "home_dining", "--policy", "rules", "--verifier", "cv", "--server-python", PY],
                        env={"CLEANPROOF_CONFIG_DIR": str(W), "CLEANPROOF_STATE_DIR": str(self.ev / "operator_state"), "MUJOCO_GL": "egl"})
        base = f"http://127.0.0.1:{oport}"
        up = self.wait_http(base + "/api/meta", op, 120)
        for _ in range(60):                                    # the launcher prints the server: line only after the service is ready, which races with this polling: wait at most 15 s
            out = (self.ev / "desktop_stdout.log").read_text(errors="replace")
            line = next((ln for ln in out.splitlines() if ln.startswith("server:")), "")
            if line:
                break
            time.sleep(0.25)
        self.check("desktop_auto_picks_paired", "robot computer: paired" in out and "--sim-robot" not in line and bool(line),
                   "launcher printed paired, the service command line has no --sim-robot")
        if not up:
            return self.finish(before_runs, token, oport)
        _, meta = http(base + "/api/meta")
        rm = (meta or {}).get("robot_mode")
        self.check("meta_robot_mode_paired", (rm == "paired") if rm is not None else None, f"/api/meta robot_mode={rm!r} (None = operator does not have this field yet)")
        code, cj = http(base + "/api/setup/connect", method="POST", body={"robot": "spot_arm"}, timeout=60)
        cs = (cj or {}).get("checks") or []
        ids = [c.get("id") for c in cs]
        self.check("wizard_connect_three_checks", code == 200 and ids == ["skill_server", "camera", "adapter"], f"ids={ids}")
        for c in cs:
            self.check(f"wizard_connect_{c.get('id')}", bool(c.get("ok")), str(c.get("detail")))
        (self.ev / "connect.json").write_text(json.dumps(cj, indent=1, ensure_ascii=False))
        # (g) full UI flow
        if not a.plumbing_only:
            shots = self.ev / "shots"
            # separate process group: on timeout kill headless Chrome along with it; configs/site is also snapshotted here, and restored as is on timeout / exception
            snap = _parity.snapshot()
            pp = subprocess.Popen([PY, "scripts/desktop_parity_shots.py", "--paired", "--url", base, "--skill-url", "http://127.0.0.1:8100",
                                   "--out", str(shots)], cwd=str(REPO), env={**os.environ, "CLEANPROOF_CONFIG_DIR": str(W)},
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
            self.procs.append(pp)
            try:
                outp, _ = pp.communicate(timeout=a.shots_timeout)
                rc_shots = pp.returncode
            except subprocess.TimeoutExpired:
                os.killpg(pp.pid, signal.SIGKILL)
                outp, _ = pp.communicate()
                rc_shots = -9
            finally:
                _parity.restore(snap)
            (self.ev / "shots.log").write_text(outp or "")
            self.check("ui_shots_exit0", rc_shots == 0, f"desktop_parity_shots.py --paired exit code {rc_shots}")
            try:
                res = json.loads((shots / "paired_results.json").read_text())
            except (OSError, ValueError):
                res = {}
            for k in ("robot_connected_chip", "toast_seen", "save_toast_ok", "save_runtime_unchanged", "resolved_empty_note", "job_started", "paused", "resumed", "stopped", "estop_clicked", "estop_health_stopped"):
                self.check(f"ui_{k}", bool(res.get(k)), str(res.get(f"{k}_detail", "")))
            ci = res.get("connect_idle_texts") or {}
            self.check("ui_connect_idle_texts", bool(ci) and all(ci.values()), json.dumps(ci, ensure_ascii=False))
            wt = res.get("wizard_texts") or {}
            self.check("ui_wizard_texts", bool(wt) and all(wt.values()), json.dumps(wt, ensure_ascii=False))
            slog = (skill_dir / "server.log").read_text(errors="replace")
            n_stop = sum(1 for ln in slog.splitlines() if "POST /stop" in ln)
            n_skill = sum(1 for ln in slog.splitlines() if "POST /skill" in ln)
            self.check("skill_server_log_has_skill_calls", n_skill > 0, f"POST /skill x{n_skill} (the job really reached the robot side through the tunnel)")
            self.check("skill_server_log_has_stop", n_stop >= 1, f"POST /stop x{n_stop} (Stop / E-stop reached the robot side)")
        return self.finish(before_runs, token, oport)

    def finish(self, before_runs: set[str], token: str, oport: int) -> int:
        self.new_runs = sorted({p.name for p in SIM_RUNS.glob("live-*")} - before_runs)
        (self.ev / "run_dirs.txt").write_text("\n".join(self.new_runs) + "\n")
        # peak memory (memory.peak of the cgroup it runs in, empty if unavailable)
        peak = None
        try:
            cg = Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().strip().split("::")[-1])
            peak = int((cg / "memory.peak").read_text()) / 2**20
        except (OSError, ValueError):
            pass
        failed = [k for k, v in self.checks.items() if v["ok"] is False]
        summary = {"plumbing_only": self.a.plumbing_only, "passed": not failed, "failed": failed, "checks": self.checks,
                   "new_run_dirs": self.new_runs, "memory_peak_mib": round(peak) if peak else None, "seconds": round(time.time() - self.t0),
                   "ports": {"tunnel": [8100, 8101], "operator": oport}}
        s = json.dumps(summary, indent=1, ensure_ascii=False)
        assert not token or token not in s
        (self.ev / "selftest.json").write_text(s)
        print(f"== {'PASS' if not failed else 'FAIL ' + str(failed)}  peak memory {summary['memory_peak_mib']} MiB  elapsed {summary['seconds']} s", flush=True)
        return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plumbing-only", action="store_true", help="verify only a-f, skip the UI clicks")
    ap.add_argument("--evidence", default=str(REPO / "runs" / "desktop_paired"))
    ap.add_argument("--memory", default="3G", help="memory limit of the whole self-test (systemd-run MemoryMax)")
    ap.add_argument("--shots-timeout", type=float, default=1500.0)
    ap.add_argument("--lock", default=LOCK, help="lock file for flock (default $CP_HEAVY_LOCK or the heavy.lock of the current session)")
    ap.add_argument("--no-wrap", action="store_true", help="do not wrap in flock + systemd-run")
    a = ap.parse_args()
    if not a.no_wrap and not os.environ.get("CP_SELFTEST_INNER"):
        os.environ["CP_SELFTEST_INNER"] = "1"
        os.execvp("flock", ["flock", a.lock, "systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={a.memory}", "-p", "MemorySwapMax=0",
                            sys.executable, *sys.argv])
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))   # also go through finally for teardown when killed
    t = Selftest(a)
    rc = 1
    try:
        rc = t.run()
    except BaseException as e:  # noqa: BLE001
        print(f"self-test error: {type(e).__name__}: {e}", file=sys.stderr)
        t.checks["exception"] = {"ok": False, "detail": f"{type(e).__name__}: {e}"}
        try:
            (t.ev / "selftest.json").write_text(json.dumps({"passed": False, "checks": t.checks}, indent=1, ensure_ascii=False))
        except OSError:
            pass
    finally:
        try:
            if not (t.ev / "selftest.json").exists():
                t.finish({p.name for p in SIM_RUNS.glob("live-*")}, "", 0)
        except OSError:
            pass
        t.teardown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
