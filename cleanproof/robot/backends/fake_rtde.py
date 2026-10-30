"""**Fake** ur_rtde interface (fake RTDE, MuJoCo backend): for tests and simulation only; not a real robot, not URSim.

The method names and parameter order of FakeRTDEControl / FakeRTDEReceive are identical to ur_rtde 1.6.5's
`rtde_control.RTDEControlInterface` / `rtde_receive.RTDEReceiveInterface` (checked against the signatures in the installed package),
but what drives them is `MujocoRobot`: speedL/moveL/moveJ advance MuJoCo kinematics and the virtual tabletop contact (contact force, stain removal),
and getActualTCPPose etc. read MuJoCo state. This way the API mapping, startup checks, staged modes and fault handling of rtde_backend
all run on a machine without URSim/docker.

Fault injection: protective stop (inject_protective_stop), speed slider ≠ 100% (set_speed_fraction), connection drop
(drop_connection / restore_connection), command latency (set_latency_ticks).

Logs and UI always say "fake RTDE (MuJoCo)"; no FakeRTDE call is ever a real call (project rule: never pass off a mock as a real call).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..io import ROBOT_MODE_RUNNING
from .mujoco_backend import MujocoRobot

FAKE_LABEL = "fake RTDE (MuJoCo)"
ROBOT_MODE_DISCONNECTED = 0   # ur_rtde getRobotMode docs: 0 = ROBOT_MODE_DISCONNECTED


class FakeRTDEDisconnected(RuntimeError):
    """Simulates calls on the interface after a connection drop (after a drop, real ur_rtde has isConnected() False and sending commands fails)."""


class FakeRTDE:
    """The set of fake interfaces and the clock belonging to one MuJoCo robot."""

    label = FAKE_LABEL

    def __init__(self, robot: Optional[MujocoRobot] = None):
        self.robot = robot or MujocoRobot(render=False)
        self.connected = True
        self.latency_ticks = 0
        self._pending: deque[tuple[int, Callable[[], Any]]] = deque()     # commands whose dispatch is delayed
        self._injections: list[tuple[int, Callable[[], Any]]] = []          # timed fault injections
        self._tick = 0
        self.calls: list[tuple[Any, ...]] = []                  # all control interface calls: (name, *args)
        self.control_created = 0                                # how many FakeRTDEControl instances were created
        self.receive = FakeRTDEReceive(self)
        self.clock = FakeClock(self)

    # ---- Factories (matching how the real RTDEControlInterface(hostname) is called) ----
    def control_factory(self) -> "FakeRTDEControl":
        self._require_connected("RTDEControlInterface()")
        self.control_created += 1
        return FakeRTDEControl(self)

    # ---- Fault injection ----
    def set_speed_fraction(self, f: float) -> None:
        self.robot.speed_fraction = float(f)

    def inject_protective_stop(self, reason: str = "fake RTDE (MuJoCo): injected protective stop") -> None:
        self.robot._protective_stop(reason)

    def clear_protective_stop(self) -> None:
        self.robot.clear_protective_stop()

    def drop_connection(self) -> None:
        self.connected = False
        self._pending.clear()
        self.robot.speed_stop()     # a real controller stops by its own watchdog after an RTDE drop

    def restore_connection(self) -> None:
        self.connected = True

    def set_latency_ticks(self, n: int) -> None:
        self.latency_ticks = max(0, int(n))

    def at_tick(self, tick: int, fn: Callable[[], Any]) -> None:
        """Run fn at control cycle tick (e.g. inject a protective stop or a disconnect halfway through a skill)."""
        self._injections.append((int(tick), fn))

    # ---- Internals ----
    def _require_connected(self, what: str) -> None:
        if not self.connected:
            raise FakeRTDEDisconnected(f"{FAKE_LABEL}: connection dropped, {what} failed")

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, *args))

    def _schedule(self, fn: Callable[[], Any]) -> None:
        if self.latency_ticks == 0:
            fn()
        else:
            self._pending.append((self._tick + self.latency_ticks, fn))

    def _flush_due(self) -> None:
        while self._pending and self._pending[0][0] <= self._tick:
            self._pending.popleft()[1]()
        due = [fn for t, fn in self._injections if t <= self._tick]
        self._injections = [(t, fn) for t, fn in self._injections if t > self._tick]
        for fn in due:
            fn()

    def step(self) -> None:
        """Advance one control cycle: first run due delayed commands, then advance MuJoCo."""
        self._tick += 1
        self._flush_due()
        self.robot.wait_tick()

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c[0] == name)


class FakeClock:
    """Clock injection point of rtde_backend: the real robot sleeps to an absolute deadline by wall clock, here the simulation is advanced."""

    is_fake = True

    def __init__(self, fake: FakeRTDE):
        self.fake = fake
        self.dt = fake.robot.control_dt

    def now(self) -> float:
        return self.fake.robot.now()

    def wait_tick(self) -> None:
        self.fake.step()


class FakeRTDEReceive:
    """Counterpart of rtde_receive.RTDEReceiveInterface (read-only). After a drop isConnected() is False and readings stay at the last value."""

    def __init__(self, fake: FakeRTDE):
        self.fake = fake
        self.disconnect_calls = 0

    @property
    def _r(self) -> MujocoRobot:
        return self.fake.robot

    def isConnected(self) -> bool:
        return self.fake.connected

    def getActualQ(self) -> list[float]:
        return self._r.get_q().tolist()

    def getActualTCPPose(self) -> list[float]:
        return self._r.get_tcp_pose().tolist()

    def getActualTCPSpeed(self) -> list[float]:
        return self._r.get_tcp_speed().tolist()

    def getActualTCPForce(self) -> list[float]:
        return self._r.get_tcp_force().tolist()

    def getRobotMode(self) -> int:
        return ROBOT_MODE_RUNNING if self.fake.connected else ROBOT_MODE_DISCONNECTED

    def getSafetyMode(self) -> int:
        return int(self._r.get_safety_mode())

    def isProtectiveStopped(self) -> bool:
        return self._r.get_safety_mode() == 3

    def isEmergencyStopped(self) -> bool:
        return self._r.get_safety_mode() in (6, 7)

    def getTargetSpeedFraction(self) -> float:
        return float(self._r.get_speed_fraction())

    def getPayload(self) -> float:
        return float(self._r.get_payload())

    def getPayloadCog(self) -> list[float]:
        return [-0.002, -0.039, 0.044]

    def getTimestamp(self) -> float:
        return float(self._r.now())

    def getRuntimeState(self) -> int:
        return 2    # PLAYING

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def reconnect(self) -> bool:
        return self.fake.connected


class FakeAsyncStatus:
    def isAsyncOperationRunning(self) -> bool:
        return False


class FakeRTDEControl:
    """Counterpart of rtde_control.RTDEControlInterface (the only writer). Every call is recorded in fake.calls."""

    def __init__(self, fake: FakeRTDE):
        self.fake = fake
        self.disconnected = False

    @property
    def _r(self) -> MujocoRobot:
        return self.fake.robot

    def _guard(self, name: str, *args: Any) -> None:
        self.fake._record(name, *args)
        if self.disconnected:
            raise FakeRTDEDisconnected(f"{FAKE_LABEL}: control interface was disconnect()ed, {name} failed")
        self.fake._require_connected(name)

    def isConnected(self) -> bool:
        return self.fake.connected and not self.disconnected

    def isProgramRunning(self) -> bool:
        return self.isConnected()

    def speedL(self, xd: Sequence[float], acceleration: float = 0.25, time: float = 0.0) -> bool:
        self._guard("speedL", list(map(float, xd)), float(acceleration), float(time))
        if self._r.get_safety_mode() != 1:
            return False
        xd_l = list(map(float, xd))
        self.fake._schedule(lambda: self._r.speed_l(xd_l, float(acceleration), float(time)))
        return True

    def speedStop(self, a: float = 10.0) -> bool:
        self._guard("speedStop", float(a))
        self.fake._pending.clear()
        self._r.speed_stop(float(a))
        return True

    def moveL(self, pose: Sequence[float], speed: float = 0.25, acceleration: float = 1.2,
              asynchronous: bool = False) -> bool:
        self._guard("moveL", list(map(float, pose)), float(speed), float(acceleration), bool(asynchronous))
        return bool(self._r.move_l(list(map(float, pose)), float(speed), float(acceleration)))

    def moveJ(self, q: Sequence[float], speed: float = 1.05, acceleration: float = 1.4,
              asynchronous: bool = False) -> bool:
        self._guard("moveJ", list(map(float, q)), float(speed), float(acceleration), bool(asynchronous))
        return bool(self._r.move_j(list(map(float, q)), float(speed), float(acceleration)))

    def setWatchdog(self, min_frequency: float = 10.0) -> bool:
        self._guard("setWatchdog", float(min_frequency))
        return True

    def kickWatchdog(self) -> bool:
        self._guard("kickWatchdog")
        return True

    def getAsyncOperationProgressEx(self) -> "FakeAsyncStatus":
        """fake moveL / moveJ (including asynchronous=True) finish synchronously inside the call, so polling always sees "not running"."""
        self._guard("getAsyncOperationProgressEx")
        return FakeAsyncStatus()

    def stopL(self, a: float = 10.0, asynchronous: bool = False) -> None:
        self._guard("stopL", float(a))
        self._r.speed_stop(float(a))

    def stopJ(self, a: float = 2.0, asynchronous: bool = False) -> None:
        self._guard("stopJ", float(a))
        self._r.speed_stop(float(a))

    def forceMode(self, *args: Any, **kwargs: Any) -> bool:
        """Our real-robot path never calls forceMode; here it is only recorded, so tests can assert the call count is 0."""
        self._guard("forceMode", args)
        return True

    def forceModeStop(self) -> bool:
        self._guard("forceModeStop")
        return True

    def zeroFtSensor(self) -> bool:
        self._guard("zeroFtSensor")
        return True

    def getTCPOffset(self) -> list[float]:
        self._guard("getTCPOffset")
        return self._r.get_tcp_offset().tolist()

    def getInverseKinematicsHasSolution(self, x: Sequence[float], qnear: Sequence[float] = (),
                                        max_position_error: float = 1e-10, max_orientation_error: float = 1e-10) -> bool:
        self._guard("getInverseKinematicsHasSolution")
        return self._r.get_ik(x, list(qnear) if len(qnear) else self._r.get_q()) is not None

    def getInverseKinematics(self, x: Sequence[float], qnear: Sequence[float] = (),
                             max_position_error: float = 1e-10, max_orientation_error: float = 1e-10) -> list[float]:
        self._guard("getInverseKinematics")
        q = self._r.get_ik(x, list(qnear) if len(qnear) else self._r.get_q())
        if q is None:
            raise RuntimeError(f"{FAKE_LABEL}: inverse kinematics has no solution")
        return np.asarray(q, dtype=float).tolist()

    def reuploadScript(self) -> bool:
        self._guard("reuploadScript")
        return True

    def stopScript(self) -> None:
        self.fake._record("stopScript")
        if self.isConnected():
            self._r.stop_script()

    def disconnect(self) -> None:
        self.fake._record("disconnect")
        self.disconnected = True

    def reconnect(self) -> bool:
        self.fake._record("reconnect")
        if self.fake.connected:
            self.disconnected = False
        return self.isConnected()
