"""Staged commissioning of the real-robot controller: address policy, motion authorization (arming), staged gate and evidence file.

Stages (PLAN §5.6, decisions 2026-10-02; the idea follows the author's own LIM-VLA policy_dry_run → policy_full gate, implemented independently):

    shadow   read-only: only create RTDEReceiveInterface, skills run as usual and record the "would have sent" commands, the robot does not move;
    dry_run  low-speed free space: contact skills are forbidden, any command that would push the TCP below the tabletop clearance is rejected, speed ratio ≤ 0.3;
    full     full function (still bound by the safety envelope of skills/safety).

The gate file (JSON, default ~/.local/state/cleanproof/commissioning.json) is registered per backend:
dry_run requires a passed shadow registered for the same backend, full requires a passed dry_run. Registering needs an evidence path
(usually runs/<run_id>/summary.json) and the person who registers. The gate only protects real controllers (ursim / ur3e); it is not enforced for fake RTDE by default,
but the same code can rehearse the whole procedure.

Motion authorization (ur3e): a person on site runs `python -m cleanproof.robot.commissioning arm --by <name>` to write the authorization file,
authorized_until = now + configs/robot.yaml arming.ttl_s; the code only reads and validates, and never writes an authorization itself.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..common.config import REPO_ROOT, expand, load_yaml

MODES = ("shadow", "dry_run", "full")
MODE_PREREQ = {"shadow": None, "dry_run": "shadow", "full": "dry_run"}
REAL_BACKENDS = ("ursim", "ur3e")
RTDE_BACKENDS = ("rtde_fake",) + REAL_BACKENDS
DEFAULT_GATE_PATH = "~/.local/state/cleanproof/commissioning.json"
DEFAULT_ARMING_PATH = "~/.local/state/cleanproof/arming.json"
ENV_GATE_PATH = "CLEANPROOF_COMMISSIONING_GATE"
ENV_ARMING_PATH = "CLEANPROOF_ARMING_TOKEN"


class CommissioningRefused(RuntimeError):
    """Address, authorization, gate or startup check not satisfied: refuse to create the control interface."""


# ---------------------------------------------------------------- Utilities
def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a JSON object")
    return data


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        # The deployed copy on robot-pc has no .git: scripts/robot_pc/sync_to_robot_pc.sh writes DEPLOYED_FROM (commit hash + whether there are uncommitted changes)
        dep = REPO_ROOT / "DEPLOYED_FROM"
        if not dep.is_file():
            return "unknown"
        parts = dep.read_text(encoding="utf-8").split()
        return parts[0] + ("+dirty" if len(parts) > 1 and parts[1] == "dirty" else "")


def commissioning_cfg(rcfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    rcfg = rcfg or load_yaml("robot.yaml")
    c = dict(rcfg.get("commissioning", {}))
    c.setdefault("gate_path", DEFAULT_GATE_PATH)
    c.setdefault("arming_token_path", DEFAULT_ARMING_PATH)
    c.setdefault("ur3e_allowed_hostnames", ["robot-pc"])
    c.setdefault("dry_run", {})
    c["dry_run"] = {"max_speed_scale": 0.3, "min_clearance_m": 0.03, **c["dry_run"]}
    return c


# ---------------------------------------------------------------- Address policy
def check_address(backend: str, host: str, rcfg: dict[str, Any]) -> None:
    """ursim only allows whitelisted prefixes (this machine / docker network), never a real robot IP; ur3e only allows the real robot address in the config."""
    host = (host or "").strip()
    real_ip = str(rcfg["hosts"]["ur3e"])
    if backend == "ursim":
        prefixes = [str(p) for p in rcfg.get("ursim_allowed_prefixes", [])]
        if host == real_ip or not any(host.startswith(p) for p in prefixes):
            raise CommissioningRefused(f"ursim backend may not connect to {host!r} (whitelisted prefixes {prefixes}, real robot {real_ip} is never allowed)")
    elif backend == "ur3e":
        if host != real_ip:
            raise CommissioningRefused(f"ur3e backend may only connect to configs/robot.yaml hosts.ur3e={real_ip}, got {host!r}")
    elif backend == "rtde_fake":
        if host:
            raise CommissioningRefused("the rtde_fake backend does not accept network addresses (it only connects to MuJoCo)")
    else:
        raise CommissioningRefused(f"unknown RTDE backend {backend!r}, choose from {RTDE_BACKENDS}")


# ---------------------------------------------------------------- Motion authorization
@dataclass
class ArmingStatus:
    armed: bool
    reason: str
    by: str = ""
    remaining_s: float = 0.0


class ArmingToken:
    """Motion authorization file written by a person on site; expires automatically. Code paths only read and validate, they never arm on a person's behalf."""

    def __init__(self, path: Optional[str | Path] = None, ttl_s: float = 300.0):
        p = path or os.environ.get(ENV_ARMING_PATH) or DEFAULT_ARMING_PATH
        self.path = expand(p)
        self.ttl_s = float(ttl_s)

    def status(self, now: Optional[float] = None) -> ArmingStatus:
        now = time.time() if now is None else now
        try:
            tok = _read_json(self.path)
        except Exception as e:  # noqa: BLE001
            return ArmingStatus(False, f"authorization file {self.path} is unreadable: {e}")
        if not tok:
            return ArmingStatus(False, f"no motion authorization ({self.path} does not exist); ask on-site staff to run arm")
        if tok.get("hostname") != socket.gethostname():
            return ArmingStatus(False, f"authorization was issued on host {tok.get('hostname')!r}, current host is {socket.gethostname()!r}")
        until = float(tok.get("authorized_until", 0.0))
        if now >= until:
            return ArmingStatus(False, f"motion authorization expired at {time.strftime('%H:%M:%S', time.localtime(until))}, re-arm")
        return ArmingStatus(True, "", by=str(tok.get("by", "")), remaining_s=until - now)

    def arm(self, by: str, now: Optional[float] = None) -> dict[str, Any]:
        """Should be called only by on-site staff in a terminal on robot-pc."""
        if not by.strip():
            raise ValueError("the name of the person arming must be given")
        now = time.time() if now is None else now
        tok = {"by": by.strip(), "hostname": socket.gethostname(), "armed_at": now,
               "authorized_until": now + self.ttl_s, "ttl_s": self.ttl_s, "git": git_rev()}
        _atomic_write_json(self.path, tok)
        return tok

    def disarm(self) -> None:
        if self.path.is_file():
            self.path.unlink()


# ---------------------------------------------------------------- graded gate
class CommissioningGate:
    def __init__(self, path: Optional[str | Path] = None):
        p = path or os.environ.get(ENV_GATE_PATH) or DEFAULT_GATE_PATH
        self.path = expand(p)

    def load(self) -> dict[str, Any]:
        return _read_json(self.path)

    def passed(self, backend: str, mode: str) -> bool:
        entry = self.load().get(backend, {}).get(mode)
        return isinstance(entry, dict) and entry.get("passed") is True

    def check(self, backend: str, mode: str) -> tuple[bool, str]:
        """Whether backend may run in mode: (ok, reason). shadow is always allowed; the others require the previous stage to be registered as passed."""
        if mode not in MODES:
            return False, f"unknown commissioning mode {mode!r}, choose from {MODES}"
        need = MODE_PREREQ[mode]
        if need is None:
            return True, ""
        try:
            ok = self.passed(backend, need)
        except Exception as e:  # noqa: BLE001
            return False, f"gate file {self.path} is unreadable: {e}"
        if not ok:
            return False, (f"before {backend} runs in {mode}, a passed {need} must be registered in {self.path}"
                           f": python -m cleanproof.robot.commissioning record --backend {backend} --stage {need} "
                           f"--evidence <runs/.../summary.json> --by <name>")
        return True, ""

    def record(self, backend: str, stage: str, evidence: str | Path, by: str,
               extra: Optional[dict[str, Any]] = None, passed: bool = True) -> dict[str, Any]:
        """Register the result of a stage. The evidence file must exist; registering dry_run needs shadow passed, registering full needs dry_run passed."""
        if backend not in RTDE_BACKENDS:
            raise ValueError(f"backend must be one of {RTDE_BACKENDS}")
        if stage not in MODES:
            raise ValueError(f"stage must be one of {MODES}")
        if not by.strip():
            raise ValueError("the person registering must be given")
        ev = Path(evidence).expanduser().resolve()
        if not ev.is_file():
            raise ValueError(f"evidence file does not exist: {ev}")
        need = MODE_PREREQ[stage]
        if need is not None and not self.passed(backend, need):
            raise ValueError(f"a passed {need} must be registered before registering {stage} (backend {backend})")
        gate = self.load()
        summary: dict[str, Any] = {}
        if ev.suffix == ".json":
            try:
                raw = json.loads(ev.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    summary = {k: raw[k] for k in ("run_id", "status", "reason", "robot", "steps", "robot_time_s") if k in raw}
            except Exception:  # noqa: BLE001
                summary = {}
        entry = {"passed": bool(passed), "evidence_path": str(ev), "evidence_mtime": ev.stat().st_mtime,
                 "recorded_by": by.strip(), "recorded_at": time.time(), "hostname": socket.gethostname(),
                 "git": git_rev(), "summary": summary, **(extra or {})}
        gate.setdefault(backend, {})[stage] = entry
        _atomic_write_json(self.path, gate)
        return entry

    def status_lines(self) -> list[str]:
        gate = self.load()
        out = [f"gate file: {self.path}"]
        for backend in RTDE_BACKENDS:
            for mode in MODES:
                e = gate.get(backend, {}).get(mode)
                if isinstance(e, dict) and e.get("passed") is True:
                    out.append(f"{backend:10s} {mode:8s} passed  by={e.get('recorded_by')!r} evidence={e.get('evidence_path')}")
                else:
                    out.append(f"{backend:10s} {mode:8s} not passed")
        return out


# ---------------------------------------------------------------- ur3e server-side guard
def ur3e_construction_problems(rcfg: dict[str, Any], hostname: Optional[str] = None,
                               arming: Optional[ArmingToken] = None) -> list[str]:
    """Hard conditions before creating a real-robot control interface: must run on robot-pc, and the motion authorization must be valid. Refuse if any is unmet."""
    c = commissioning_cfg(rcfg)
    hostname = hostname or socket.gethostname()
    out = []
    if hostname not in [str(h) for h in c["ur3e_allowed_hostnames"]]:
        out.append(f"the ur3e control interface can only be created on robot host {c['ur3e_allowed_hostnames']}, current host is {hostname!r}")
    st = (arming or ArmingToken(c["arming_token_path"], float(rcfg.get("arming", {}).get("ttl_s", 300)))).status()
    if not st.armed:
        out.append(st.reason)
    return out


# ---------------------------------------------------------------- Command line
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="staged commissioning gate and motion authorization (only touches local evidence files, never connects to the robot)")
    ap.add_argument("--gate-path", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("record", help="register that a stage passed (evidence is usually runs/<run_id>/summary.json)")
    r.add_argument("--backend", required=True, choices=RTDE_BACKENDS)
    r.add_argument("--stage", required=True, choices=MODES)
    r.add_argument("--evidence", required=True)
    r.add_argument("--by", required=True)
    a = sub.add_parser("arm", help="on-site staff issue a motion authorization on the robot host (time-limited)")
    a.add_argument("--by", required=True)
    a.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    sub.add_parser("disarm")
    args = ap.parse_args(argv)
    rcfg = load_yaml("robot.yaml")
    c = commissioning_cfg(rcfg)
    gate = CommissioningGate(args.gate_path or c["gate_path"])
    if args.cmd == "status":
        print("\n".join(gate.status_lines()))
        tok = ArmingToken(c["arming_token_path"], float(rcfg["arming"]["ttl_s"])).status()
        print(f"motion authorization: {'valid, %.0f s left (%s)' % (tok.remaining_s, tok.by) if tok.armed else tok.reason}")
        return 0
    if args.cmd == "record":
        e = gate.record(args.backend, args.stage, args.evidence, args.by)
        print(f"registered {args.backend}/{args.stage}: {json.dumps(e, ensure_ascii=False)}")
        return 0
    tok = ArmingToken(c["arming_token_path"], float(rcfg["arming"]["ttl_s"]))
    if args.cmd == "disarm":
        tok.disarm()
        print("motion authorization revoked")
        return 0
    if args.cmd == "arm":
        hostname = socket.gethostname()
        if hostname not in [str(h) for h in c["ur3e_allowed_hostnames"]]:
            print(f"refused: arming is only allowed on robot host {c['ur3e_allowed_hostnames']}, current host is {hostname!r}", file=sys.stderr)
            return 2
        if not args.yes:
            ans = input(f"Confirm that a person is on site, the E-stop is within reach and the teach-pendant slider is at 100%? The authorization expires after {rcfg['arming']['ttl_s']} s. Type ARM to confirm: ")
            if ans.strip() != "ARM":
                print("not confirmed, exiting")
                return 1
        t = tok.arm(args.by)
        print(f"armed: {t['by']}, expires at {time.strftime('%H:%M:%S', time.localtime(t['authorized_until']))}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
