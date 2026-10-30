"""ur_rtde backend: maps RobotIO to RTDEControlInterface / RTDEReceiveInterface of ur_rtde.

The interface objects are passed in by dependency injection (receive instance + control factory), so the same code can drive:
- `rtde_fake`: fake_rtde.FakeRTDE (MuJoCo physics, local tests, not a real robot);
- `ursim`: URSim (whitelisted addresses only);
- `ur3e`: the real robot (created by the skill server only on robot-pc, after manual arming; the development machine of this repo can never get to this step).

Method names follow the signatures of the installed ur_rtde 1.6.5 package (checked on 2026-10-02 after `uv pip install ur_rtde` in .venv, no robot connected):
    Control: speedL(xd, acceleration, time) / speedStop(a) / moveL(pose, speed, acceleration, asynchronous)
          / moveJ(q, speed, acceleration, asynchronous) / getTCPOffset() / getInverseKinematics(x, qnear)
          / getInverseKinematicsHasSolution(x, qnear) / reuploadScript() / stopScript() / disconnect() / isConnected()
    Receive: getActualQ / getActualTCPPose / getActualTCPSpeed / getActualTCPForce / getRobotMode / getSafetyMode
          / isProtectiveStopped / isEmergencyStopped / getTargetSpeedFraction / getPayload / getPayloadCog / isConnected
Construction: `RTDEControlInterface(hostname)`, `RTDEReceiveInterface(hostname)`, no frequency passed (our earlier UR3e force-control work convention).

Staged modes (cleanproof.robot.commissioning):
    shadow   no control interface is created; speed_l/move_l/move_j only record the "would have sent" command and return True;
    dry_run  speed ratio ≤ 0.3 (acceleration × ratio²), commands that would push the TCP below the tabletop clearance are rejected, the skill layer rejects contact skills;
    full     everything enabled (still bound by the skills/safety envelope).
Single writer: at most one RTDERobot per process holds the control interface.
Fault handling : a failed control-interface call / a disconnect → mark connection_lost, get_robot_mode returns 0 (DISCONNECTED),
the skill layer stops accordingly and hands over to a human; close() goes speedStop → stopScript → disconnect (forceMode is not used, so there is no forceModeStop).
"""

from __future__ import annotations

import logging
import socket
import threading
import time as _time
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ...common.config import load_yaml
from ..commissioning import (
    MODES,
    REAL_BACKENDS,
    RTDE_BACKENDS,
    ArmingToken,
    CommissioningGate,
    CommissioningRefused,
    check_address,
    commissioning_cfg,
    ur3e_construction_problems,
)
from ..io import ROBOT_MODE_RUNNING, SAFETY_MODE_NORMAL
from ..kinematics import UR3eKinematics

log = logging.getLogger("cleanproof.robot.rtde")

ROBOT_MODE_DISCONNECTED = 0
SAFETY_MODE_UNKNOWN = -1
SAFETY_MODE_SOFTWARE_STOP = 99    # defined by this repo: software stop after the skill server receives /stop (not a UR safety mode number)

_CONTROL_OWNER: Optional[int] = None   # id() of the RTDERobot that holds the control interface (single writer)


class SingleWriterViolation(RuntimeError):
    """Another RTDERobot in this process already holds the control interface."""


class RealtimeClock:
    """Real-robot control clock: perf_counter absolute-deadline scheduling; if more than one tick behind, reset instead of catching up (our earlier UR3e force-control work §3)."""

    is_fake = False

    def __init__(self, dt: float):
        self.dt = float(dt)
        self._next = _time.perf_counter() + self.dt
        self.resets = 0

    def now(self) -> float:
        return _time.perf_counter()

    def wait_tick(self) -> None:
        t = _time.perf_counter()
        if t > self._next + self.dt:
            self._next = t + self.dt
            self.resets += 1
        else:
            remain = self._next - t
            if remain > 0:
                _time.sleep(remain)
            self._next += self.dt


class RTDERobot:
    """ur_rtde implementation of RobotIO. backend_name ∈ {rtde_fake, ursim, ur3e}."""

    def __init__(self, backend: str, mode: str, receive: Any, control_factory: Optional[Callable[[], Any]],
                 clock: Any, rcfg: Optional[dict[str, Any]] = None, floor_z: Optional[float] = None,
                 label: Optional[str] = None):
        if backend not in RTDE_BACKENDS:
            raise CommissioningRefused(f"unknown RTDE backend {backend!r}")
        if mode not in MODES:
            raise CommissioningRefused(f"unknown commissioning mode {mode!r}, choose from {MODES}")
        self.backend_name = backend
        self.mode = mode
        self.label = label or backend
        self.rcfg = rcfg or load_yaml("robot.yaml")
        self.control_dt = 1.0 / float(self.rcfg["control_hz"])
        self.receive = receive
        self.control: Any = None
        self._control_factory = control_factory
        self.clock = clock
        self.tcp_offset_cfg = np.array(self.rcfg["startup"]["expected_tcp_offset"], dtype=float)
        self.kin = UR3eKinematics(self.tcp_offset_cfg)          # local kinematics: IK for shadow and path spot checks in the skill layer
        cc = commissioning_cfg(self.rcfg)
        self.speed_scale = float(cc["dry_run"]["max_speed_scale"]) if mode == "dry_run" else 1.0
        self.floor_z = floor_z if mode == "dry_run" else None     # lowest allowed TCP height in dry_run (tabletop + clearance)
        self.contact_allowed = mode == "full"
        self.shadow_log: list[dict[str, Any]] = []
        self.connection_lost = False
        self.connection_lost_reason = ""
        self.last_refusal: Optional[str] = None
        self.protective_stop_reason: Optional[str] = None
        self.software_stop_reason: Optional[str] = None   # /stop: reject later motion commands; the skill loop judges fatal on the next tick
        self._last: dict[str, Any] = {}
        self.closed = False
        # The control interface is not thread-safe (ur_rtde releases the GIL during calls, commands share input registers): all control interface calls are serialized by this lock
        self._ctrl_lock = threading.RLock()
        self._tcp_offset: Optional[np.ndarray] = None       # read once at startup and cached (the /health thread never touches the control interface)
        rc = dict(self.rcfg.get("rtde", {}) or {})
        self.watchdog_hz = float(rc.get("watchdog_hz", 10.0))
        self.watchdog_kicks = 0
        self._kicker: Optional[threading.Thread] = None
        self._startup_and_connect()

    # ================= Startup =================
    def _startup_and_connect(self) -> None:
        global _CONTROL_OWNER
        st = self.rcfg["startup"]
        if not self._recv("isConnected", True):
            raise CommissioningRefused(f"{self.label}: RTDEReceiveInterface is not connected")
        mode_now = int(self._recv("getRobotMode", -1))
        if mode_now != int(st["require_robot_mode"]):
            raise CommissioningRefused(f"{self.label}: robot_mode={mode_now}, requires {st['require_robot_mode']} (running state)")
        frac = float(self._recv("getTargetSpeedFraction", float("nan")))
        if not abs(frac - float(st["require_speed_fraction"])) <= float(st["speed_fraction_tol"]):
            raise CommissioningRefused(f"{self.label}: teach-pendant speed slider is {frac:.2f}, must be 100% (speed scaling is done by the program)")
        if self.mode == "shadow":
            log.info("%s: shadow mode, read-only telemetry, no RTDEControlInterface is created, skill commands are only recorded and not sent", self.label)
            return
        if self._control_factory is None:
            raise CommissioningRefused(f"{self.label}: mode {self.mode} needs a control interface factory")
        if _CONTROL_OWNER is not None:
            raise SingleWriterViolation(f"{self.label}: this process already has an RTDEControlInterface holder (single writer)")
        self.control = self._control_factory()
        _CONTROL_OWNER = id(self)
        # Controller watchdog: the speed thread of the ur_rtde control script keeps moving at the last speedL; without the watchdog, if the skill server crashes,
        # is killed, or robot-pc loses its connection to the robot, the robot keeps moving until a protective stop. With it on, if kicks stop for more than 1/watchdog_hz seconds, the controller shuts down the control script.
        # Real controllers (ursim / ur3e) are kicked by a separate thread at 3× the frequency (kicks stop if the process dies); the fake backend only records setWatchdog.
        if self.watchdog_hz > 0:
            if not self._ctrl("setWatchdog", False, self.watchdog_hz):
                raise CommissioningRefused(f"{self.label}: setWatchdog({self.watchdog_hz}) failed")
            if self.backend_name in REAL_BACKENDS:
                self._kicker = threading.Thread(target=self._kick_loop, daemon=True, name="rtde-watchdog")
                self._kicker.start()
        else:
            log.warning("%s: controller watchdog is off (rtde.watchdog_hz=0)", self.label)
        off = self._ctrl("getTCPOffset", None)
        self._tcp_offset = None if off is None else np.asarray(off, dtype=float)
        log.info("%s: control interface created (mode=%s, speed_scale=%.2f, floor_z=%s, watchdog=%s Hz)", self.label, self.mode,
                 self.speed_scale, None if self.floor_z is None else f"{self.floor_z:.3f}", self.watchdog_hz)

    def _kick_loop(self) -> None:
        period = 1.0 / (3.0 * self.watchdog_hz)
        while not self.closed and self.control is not None:
            if self._ctrl("kickWatchdog", False):
                self.watchdog_kicks += 1
            _time.sleep(period)

    @property
    def describe(self) -> str:
        return f"{self.label} [mode={self.mode}]"

    # ================= Read-only =================
    def _recv(self, name: str, default: Any) -> Any:
        try:
            if name != "isConnected" and not self.receive.isConnected():
                self._lost("RTDEReceiveInterface disconnected")
                return self._last.get(name, default)
            v = getattr(self.receive, name)()
            self._last[name] = v
            return v
        except Exception as e:  # noqa: BLE001
            self._lost(f"{name} failed: {type(e).__name__}: {e}")
            return self._last.get(name, default)

    def _lost(self, reason: str) -> None:
        if not self.connection_lost:
            self.connection_lost = True
            self.connection_lost_reason = reason
            log.error("%s: RTDE connection lost: %s", self.label, reason)

    def now(self) -> float:
        return float(self.clock.now())

    def wait_tick(self) -> None:
        self.clock.wait_tick()

    def get_q(self) -> np.ndarray:
        return np.asarray(self._recv("getActualQ", [0.0] * 6), dtype=float)

    def get_tcp_pose(self) -> np.ndarray:
        return np.asarray(self._recv("getActualTCPPose", [0.0] * 6), dtype=float)

    def get_tcp_speed(self) -> np.ndarray:
        return np.asarray(self._recv("getActualTCPSpeed", [0.0] * 6), dtype=float)

    def get_tcp_force(self) -> np.ndarray:
        return np.asarray(self._recv("getActualTCPForce", [0.0] * 6), dtype=float)

    def get_robot_mode(self) -> int:
        if self.connection_lost:
            return ROBOT_MODE_DISCONNECTED
        m = int(self._recv("getRobotMode", ROBOT_MODE_DISCONNECTED))
        return ROBOT_MODE_DISCONNECTED if self.connection_lost else m

    def get_safety_mode(self) -> int:
        if self.connection_lost:
            return SAFETY_MODE_UNKNOWN
        if self.software_stop_reason is not None:
            self.protective_stop_reason = f"software stop: {self.software_stop_reason}"
            return SAFETY_MODE_SOFTWARE_STOP
        sm = int(self._recv("getSafetyMode", SAFETY_MODE_UNKNOWN))
        if sm != SAFETY_MODE_NORMAL and self.protective_stop_reason is None:
            if self._recv("isEmergencyStopped", False):
                self.protective_stop_reason = f"emergency stop (safety_mode={sm})"
            elif self._recv("isProtectiveStopped", False):
                self.protective_stop_reason = f"controller protective stop (safety_mode={sm}), must be cleared manually on the teach pendant"
            else:
                self.protective_stop_reason = f"safety_mode={sm}"
        elif sm == SAFETY_MODE_NORMAL:
            self.protective_stop_reason = None
        return sm

    def get_speed_fraction(self) -> float:
        return float(self._recv("getTargetSpeedFraction", float("nan")))

    def get_tcp_offset(self) -> np.ndarray:
        """Cached value read once from the control interface at startup. shadow has no control interface and returns the config value (the teach-pendant TCP must be checked by hand)."""
        if self._tcp_offset is None:
            return self.tcp_offset_cfg.copy()
        return self._tcp_offset.copy()

    def get_payload(self) -> float:
        return float(self._recv("getPayload", float("nan")))

    # ================= Software stop =================
    def emergency_stop(self, reason: str = "operator stop") -> None:
        """Software stop from the console / skill server (callable from other threads, only writes one flag).

        After this, speed_l / move_l / move_j are always rejected; a running speedL wipe loop is judged fatal by SafetyMonitor on the next tick (4 ms) and
        speedStop is issued; a running moveL / moveJ (dispatched asynchronously, polled per tick) gets stopL / stopJ on the next tick. A software stop does not replace the physical emergency stop.
        """
        if self.software_stop_reason is None:
            self.software_stop_reason = reason
            log.warning("%s: software stop: %s", self.label, reason)

    def clear_software_stop(self) -> None:
        self.software_stop_reason = None
        self.protective_stop_reason = None

    # ================= motion =================
    def _shadow(self, cmd: str, **kw: Any) -> None:
        rec = {"t": self.now(), "cmd": cmd, **kw}
        self.shadow_log.append(rec)
        log.info("%s shadow: not sending %s %s", self.label, cmd, {k: v for k, v in kw.items()})

    def _ctrl(self, name: str, default: Any, *args: Any) -> Any:
        """Call the control interface; on a disconnect or exception → mark lost and return default (False for speedL/moveL)."""
        if self.control is None:
            return default
        with self._ctrl_lock:
            try:
                if not self.control.isConnected():
                    self._lost("RTDEControlInterface disconnected")
                    return default
                return getattr(self.control, name)(*args)
            except Exception as e:  # noqa: BLE001
                self._lost(f"{name} failed: {type(e).__name__}: {e}")
                return default

    def _reached(self, kind: str, target: list[float]) -> bool:
        if kind == "moveL":
            return float(np.linalg.norm(self.get_tcp_pose()[:3] - np.asarray(target[:3]))) < 0.002
        return float(np.max(np.abs(self.get_q() - np.asarray(target)))) < 0.01

    def _await_move(self, kind: str, target: list[float], timeout_s: float) -> bool:
        """After an asynchronous moveL / moveJ is dispatched, poll per tick: software stop → stopL / stopJ; protective stop / disconnect / timeout → failure; after finishing, verify arrival."""
        stop_fn, stop_a = ("stopL", 0.5) if kind == "moveL" else ("stopJ", 1.0)
        t0 = self.now()
        seen_running = False
        while True:
            el = self.now() - t0
            if self.software_stop_reason is not None:
                self._ctrl(stop_fn, None, stop_a)
                return self._refuse(f"{kind} interrupted by software stop ({self.software_stop_reason})")
            if self.connection_lost:
                return False
            st = self._ctrl("getAsyncOperationProgressEx", None)
            running = bool(st.isAsyncOperationRunning()) if st is not None else False
            seen_running = seen_running or running
            if not running:
                if self._reached(kind, target):
                    return True
                if seen_running or el > 0.5:
                    return False                      # finished but did not arrive (protective stop etc.), judged by _move_failure in the skill layer
            if int(self.get_safety_mode()) != SAFETY_MODE_NORMAL:
                self._ctrl(stop_fn, None, stop_a)
                if self.software_stop_reason is not None:
                    return self._refuse(f"{kind} interrupted by software stop ({self.software_stop_reason})")
                return False
            if el > timeout_s:
                self._ctrl(stop_fn, None, stop_a)
                return self._refuse(f"{kind} timed out ({timeout_s:.0f} s)")
            self.clock.wait_tick()

    def _refuse(self, msg: str) -> bool:
        self.last_refusal = f"{self.mode} mode refused: {msg}"
        log.warning("%s %s", self.label, self.last_refusal)
        return False

    def speed_l(self, twist: Sequence[float], acceleration: float, time: float) -> bool:
        xd = [float(v) for v in twist]
        if self.software_stop_reason is not None:
            self._ctrl("speedStop", False, 0.5)
            return self._refuse(f"software stop active ({self.software_stop_reason})")
        if self.mode == "shadow":
            self._shadow("speedL", xd=xd, acceleration=float(acceleration), time=float(time))
            return True
        if self.floor_z is not None and xd[2] < 0:
            z = float(self.get_tcp_pose()[2])
            horizon = float(self.rcfg["safety"]["command_timeout_s"])
            if z + xd[2] * horizon < self.floor_z:
                return self._refuse(f"speedL would press TCP below {self.floor_z:.3f} m (current z={z:.3f}, vz={xd[2]:.4f})")
        s = self.speed_scale
        xd = [v * s for v in xd]
        self.last_refusal = None
        return bool(self._ctrl("speedL", False, xd, float(acceleration) * s * s, float(time)))

    def speed_stop(self, deceleration: float = 0.5) -> None:
        if self.mode == "shadow":
            self._shadow("speedStop", a=float(deceleration))
            return
        self._ctrl("speedStop", False, float(deceleration))

    def move_l(self, pose: Sequence[float], speed: float, acceleration: float) -> bool:
        p = [float(v) for v in pose]
        if self.software_stop_reason is not None:
            return self._refuse(f"software stop active ({self.software_stop_reason})")
        if self.mode == "shadow":
            self._shadow("moveL", pose=p, speed=float(speed), acceleration=float(acceleration))
            return True
        if self.floor_z is not None and p[2] < self.floor_z:
            return self._refuse(f"moveL target z={p[2]:.3f} m is below {self.floor_z:.3f} m")
        s = self.speed_scale
        self.last_refusal = None
        dist = float(np.linalg.norm(self.get_tcp_pose()[:3] - np.asarray(p[:3])))
        if not self._ctrl("moveL", False, p, float(speed) * s, float(acceleration) * s * s, True):
            return False
        return self._await_move("moveL", p, 2.0 * dist / max(float(speed) * s, 1e-3) + 10.0)

    def move_j(self, q: Sequence[float], speed: float, acceleration: float) -> bool:
        qq = [float(v) for v in q]
        if self.software_stop_reason is not None:
            return self._refuse(f"software stop active ({self.software_stop_reason})")
        if self.mode == "shadow":
            self._shadow("moveJ", q=qq, speed=float(speed), acceleration=float(acceleration))
            return True
        if self.floor_z is not None:
            z = float(self.kin.fk(qq)[2, 3])
            if z < self.floor_z:
                return self._refuse(f"moveJ target configuration TCP z={z:.3f} is below {self.floor_z:.3f} m")
        s = self.speed_scale
        self.last_refusal = None
        dq = float(np.max(np.abs(self.get_q() - np.asarray(qq))))
        if not self._ctrl("moveJ", False, qq, float(speed) * s, float(acceleration) * s * s, True):
            return False
        return self._await_move("moveJ", qq, 2.0 * dq / max(float(speed) * s, 1e-3) + 10.0)

    def get_ik(self, pose: Sequence[float], q_near: Sequence[float]) -> Optional[np.ndarray]:
        if self.control is None or self.connection_lost:
            return self.kin.ik(np.asarray(pose, dtype=float), q_near)
        x, qn = [float(v) for v in pose], [float(v) for v in q_near]
        if not self._ctrl("getInverseKinematicsHasSolution", False, x, qn):
            return None
        q = self._ctrl("getInverseKinematics", None, x, qn)
        return None if q is None else np.asarray(q, dtype=float)

    def stop_script(self) -> None:
        if self.mode == "shadow":
            self._shadow("stopScript")
            return
        self._ctrl("stopScript", None)

    def close(self) -> None:
        """fatal stop order: speedStop → stopScript → disconnect; release the single writer. Can be called repeatedly."""
        global _CONTROL_OWNER
        if self.closed:
            return
        self.closed = True                       # the watchdog thread exits when it sees this
        if self.control is not None:
            with self._ctrl_lock:
                for name in ("speedStop", "stopScript", "disconnect"):
                    try:
                        getattr(self.control, name)()
                    except Exception as e:  # noqa: BLE001
                        log.warning("%s close: %s failed: %s", self.label, name, e)
                if _CONTROL_OWNER == id(self):
                    _CONTROL_OWNER = None
                self.control = None
        if self._kicker is not None:
            self._kicker.join(timeout=1.0)
        try:
            self.receive.disconnect()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:  # safety net: release the writer slot when the object is garbage collected
        global _CONTROL_OWNER
        if _CONTROL_OWNER == id(self):
            _CONTROL_OWNER = None


def _reset_single_writer_for_tests() -> None:
    global _CONTROL_OWNER
    _CONTROL_OWNER = None


# ================= Factory =================
def make_rtde_robot(backend: str, mode: str = "full", rcfg: Optional[dict[str, Any]] = None,
                    host: Optional[str] = None, z_surface: Optional[float] = None, fake: Any = None,
                    gate: Optional[CommissioningGate] = None, enforce_gate: Optional[bool] = None,
                    rtde_modules: Optional[dict[str, Any]] = None, hostname: Optional[str] = None,
                    arming: Optional[ArmingToken] = None) -> RTDERobot:
    """Create an RTDERobot by policy. Real-robot interfaces (ursim / ur3e) are constructed only after all checks pass, and only here.

    - rtde_fake: fake must be fake_rtde.FakeRTDE; the gate is not enforced by default (enforce_gate=True to rehearse).
    - ursim: host must match ursim_allowed_prefixes; the gate is enforced.
    - ur3e: must run on robot-pc; dry_run/full also require a valid motion authorization; the gate is enforced. On a development machine this step is always refused.
    rtde_modules: tests inject {"rtde_control": module, "rtde_receive": module}; by default the real ur_rtde is imported.
    """
    rcfg = rcfg or load_yaml("robot.yaml")
    cc = commissioning_cfg(rcfg)
    if mode not in MODES:
        raise CommissioningRefused(f"unknown commissioning mode {mode!r}, choose from {MODES}")
    if backend not in RTDE_BACKENDS:
        raise CommissioningRefused(f"unknown RTDE backend {backend!r}, choose from {RTDE_BACKENDS}")
    if z_surface is None:
        z_surface = float(load_yaml("zones.yaml")["table"]["z_surface"])
    floor_z = float(z_surface) + float(cc["dry_run"]["min_clearance_m"])

    if backend == "rtde_fake":
        if fake is None:
            raise CommissioningRefused("rtde_fake needs a fake_rtde.FakeRTDE instance")
        check_address(backend, host or "", rcfg)
        enforce = bool(enforce_gate)
        receive, control_factory, clock, label = fake.receive, fake.control_factory, fake.clock, fake.label
    else:
        host = host or str(rcfg["hosts"][backend])
        check_address(backend, host, rcfg)
        if backend == "ur3e":
            problems = ur3e_construction_problems(rcfg, hostname=hostname, arming=arming) if mode != "shadow" else \
                ur3e_construction_problems(rcfg, hostname=hostname, arming=arming)[:1]  # shadow only requires being on robot-pc
            if problems:
                raise CommissioningRefused("refusing to create the ur3e interface: " + "; ".join(problems))
        enforce = True if enforce_gate is None else bool(enforce_gate)
    if enforce:                                   # the gate is checked before any interface is constructed
        ok, reason = (gate or CommissioningGate(cc["gate_path"])).check(backend, mode)
        if not ok:
            raise CommissioningRefused(reason)
    if backend in REAL_BACKENDS:
        mods = rtde_modules or _import_ur_rtde()
        log.warning("creating %s read-only interface RTDEReceiveInterface(%s) (mode=%s, host %s)", backend, host, mode,
                    hostname or socket.gethostname())
        receive = mods["rtde_receive"].RTDEReceiveInterface(host)       # read-only telemetry
        control_factory = (lambda: mods["rtde_control"].RTDEControlInterface(host))   # only called inside RTDERobot according to the mode
        clock = RealtimeClock(1.0 / float(rcfg["control_hz"]))
        label = f"{backend}@{host}"
    return RTDERobot(backend, mode, receive, control_factory, clock, rcfg=rcfg, floor_z=floor_z, label=label)


def _import_ur_rtde() -> dict[str, Any]:
    try:
        import rtde_control  # noqa: F401
        import rtde_receive  # noqa: F401
    except ImportError as e:
        raise CommissioningRefused(f"ur_rtde is not installed (pip install ur_rtde): {e}") from e
    return {"rtde_control": rtde_control, "rtde_receive": rtde_receive}
