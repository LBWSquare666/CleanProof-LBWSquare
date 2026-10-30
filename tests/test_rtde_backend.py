"""ur_rtde backend: API mapping, startup checks, address and authorization policy, graded gate and modes, fault injection, closed loop through fake RTDE (MuJoCo).

All offline: no RTDE interface connecting to a network address is created; the real ur_rtde is only used to check method names (that one is skipped if not installed).
"""

from __future__ import annotations

import json
import time
import types

import numpy as np
import pytest

from cleanproof.common.config import load_yaml
from cleanproof.common.zones import ZoneGrid
from cleanproof.robot import commissioning as cm
from cleanproof.robot.backends import rtde_backend as rb
from cleanproof.robot.backends.fake_rtde import FAKE_LABEL, FakeRTDE, FakeRTDEControl, FakeRTDEReceive
from cleanproof.robot.backends.mujoco_backend import MujocoRobot
from cleanproof.robot.backends.rtde_backend import RTDERobot, SingleWriterViolation, make_rtde_robot
from cleanproof.robot.io import RobotIO
from cleanproof.robot.skills import SkillExecutor
from cleanproof.sim.run import run_sim_scenario

RCFG = load_yaml("robot.yaml")
Z_SURFACE = ZoneGrid().z_surface


@pytest.fixture(autouse=True)
def _single_writer_reset():
    rb._reset_single_writer_for_tests()
    yield
    rb._reset_single_writer_for_tests()


def make(mode="full", fake=None, **kw):
    fake = fake or FakeRTDE(MujocoRobot(render=False))
    robot = make_rtde_robot("rtde_fake", mode, rcfg=RCFG, z_surface=Z_SURFACE, fake=fake, **kw)
    return fake, robot


def executor(robot):
    return SkillExecutor(robot, grid=robot_grid(robot), rcfg=RCFG)


def robot_grid(robot):
    return ZoneGrid()


# ---------------------------------------------------------------- API mapping
def test_fake_names_match_installed_ur_rtde():
    rtde_control = pytest.importorskip("rtde_control")
    rtde_receive = pytest.importorskip("rtde_receive")
    C, R = rtde_control.RTDEControlInterface, rtde_receive.RTDEReceiveInterface
    fake_c = [n for n in vars(FakeRTDEControl) if not n.startswith("_") and callable(getattr(FakeRTDEControl, n))]
    fake_r = [n for n in vars(FakeRTDEReceive) if not n.startswith("_") and callable(getattr(FakeRTDEReceive, n))]
    assert fake_c and fake_r
    assert [n for n in fake_c if not hasattr(C, n)] == []
    assert [n for n in fake_r if not hasattr(R, n)] == []
    # the names the backend actually uses
    for n in ("speedL", "speedStop", "moveL", "moveJ", "getTCPOffset", "getInverseKinematics",
              "getInverseKinematicsHasSolution", "stopScript", "disconnect", "isConnected"):
        assert hasattr(C, n), n
    for n in ("getActualQ", "getActualTCPPose", "getActualTCPSpeed", "getActualTCPForce", "getRobotMode",
              "getSafetyMode", "isProtectiveStopped", "isEmergencyStopped", "getTargetSpeedFraction", "getPayload",
              "isConnected", "disconnect"):
        assert hasattr(R, n), n


def test_robot_io_mapping_to_ur_rtde_calls():
    fake, r = make()
    assert isinstance(r, RobotIO) and r.backend_name == "rtde_fake" and r.label == FAKE_LABEL
    mj = fake.robot
    assert np.allclose(r.get_q(), mj.get_q()) and np.allclose(r.get_tcp_pose(), mj.get_tcp_pose())
    assert np.allclose(r.get_tcp_force(), mj.get_tcp_force()) and r.get_robot_mode() == 7 and r.get_safety_mode() == 1
    assert r.get_speed_fraction() == 1.0 and r.get_payload() == pytest.approx(0.88)
    assert np.allclose(r.get_tcp_offset(), mj.get_tcp_offset())
    assert fake.count("getTCPOffset") == 1                    # read once at startup and cached (the /health thread does not touch the control interface)
    r.get_tcp_offset()
    assert fake.count("getTCPOffset") == 1
    assert fake.count("setWatchdog") == 1                     # the controller watchdog is enabled right after the control interface is built
    assert r.speed_l([0, 0, 0.01, 0, 0, 0], 0.09, 0.02)
    assert fake.calls[-1] == ("speedL", [0, 0, 0.01, 0, 0, 0], 0.09, 0.02)
    for _ in range(50):                                       # the speedL time parameter of 0.02 s decelerates on expiry, so it must be resent every tick
        r.speed_l([0, 0, 0.01, 0, 0, 0], 0.09, 0.02)
        r.wait_tick()
    assert r.get_tcp_speed()[2] > 0.005                       # speedL really drove MuJoCo
    r.speed_stop(0.5)
    assert fake.calls[-1] == ("speedStop", 0.5)
    pose = r.get_tcp_pose()
    pose[2] += 0.02
    assert r.move_l(pose, 0.03, 0.045)
    mv = [c for c in fake.calls if c[0] == "moveL"][-1]
    assert mv[2:] == (0.03, 0.045, True)                      # sent asynchronously, polled every tick (stopL possible along the way)
    assert np.allclose(r.get_tcp_pose()[:3], pose[:3], atol=1e-4)
    q = r.get_q()
    q[0] += 0.1
    assert r.move_j(q, 0.18, 0.108)
    mj_call = [c for c in fake.calls if c[0] == "moveJ"][-1]
    assert mj_call[2:] == (0.18, 0.108, True)
    assert np.allclose(r.get_q(), q, atol=1e-6)
    ik = r.get_ik(r.get_tcp_pose(), r.get_q())
    assert ik is not None and np.allclose(ik, r.get_q(), atol=1e-3)
    assert fake.count("forceMode") == 0
    r.close()
    assert [c[0] for c in fake.calls[-3:]] == ["speedStop", "stopScript", "disconnect"]
    assert fake.receive.disconnect_calls == 1
    r.close()                                                # can be called repeatedly


def test_single_writer_per_process():
    fake, r1 = make()
    with pytest.raises(SingleWriterViolation):
        RTDERobot("rtde_fake", "full", fake.receive, fake.control_factory, fake.clock, rcfg=RCFG)
    assert fake.control_created == 1
    r1.close()
    r2 = RTDERobot("rtde_fake", "full", fake.receive, fake.control_factory, fake.clock, rcfg=RCFG)
    assert fake.control_created == 2
    r2.close()


def test_realtime_clock_deadline_scheduling():
    clk = rb.RealtimeClock(0.002)
    t0 = time.perf_counter()
    for _ in range(25):
        clk.wait_tick()
    assert 0.045 <= time.perf_counter() - t0 <= 0.2
    time.sleep(0.02)                                          # more than one tick behind: reset, do not catch up
    clk.wait_tick()
    assert clk.resets == 1


# ---------------------------------------------------------------- startup checks
def test_speed_slider_not_100_percent_refuses_before_control_is_created():
    fake = FakeRTDE(MujocoRobot(render=False))
    fake.set_speed_fraction(0.3)
    with pytest.raises(cm.CommissioningRefused, match="100%"):
        make(fake=fake)
    assert fake.control_created == 0
    fake.set_speed_fraction(1.0)
    _, r = make(fake=fake)
    assert fake.control_created == 1
    r.close()


def test_disconnected_receive_refuses_startup():
    fake = FakeRTDE(MujocoRobot(render=False))
    fake.drop_connection()
    with pytest.raises(cm.CommissioningRefused):
        make(fake=fake)
    assert fake.control_created == 0


# ---------------------------------------------------------------- address and authorization policy
@pytest.mark.parametrize("host,ok", [("127.0.0.1", True), ("localhost", True), ("172.17.0.2", True),
                                     ("10.20.30.101", False), ("172.16.9.7", False), ("10.20.30.100", False)])
def test_ursim_address_whitelist(host, ok):
    if ok:
        cm.check_address("ursim", host, RCFG)
    else:
        with pytest.raises(cm.CommissioningRefused):
            cm.check_address("ursim", host, RCFG)


def test_ur3e_address_must_be_configured_robot_ip():
    cm.check_address("ur3e", "10.20.30.101", RCFG)
    with pytest.raises(cm.CommissioningRefused):
        cm.check_address("ur3e", "127.0.0.1", RCFG)
    with pytest.raises(cm.CommissioningRefused):
        cm.check_address("rtde_fake", "127.0.0.1", RCFG)


def _spy_modules(record):
    class Control:
        def __init__(self, host):
            record.append(("RTDEControlInterface", host))

    class Receive:
        def __init__(self, host):
            record.append(("RTDEReceiveInterface", host))

    return {"rtde_control": types.SimpleNamespace(RTDEControlInterface=Control),
            "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=Receive)}


def test_server_refuses_to_construct_ur3e(tmp_path):
    """Dev machine (not robot-pc) / not armed / gate not passed: all three cases are refused before any real-robot interface is constructed."""
    record = []
    arming = cm.ArmingToken(tmp_path / "arming.json", ttl_s=60)
    gate = cm.CommissioningGate(tmp_path / "gate.json")
    for mode in ("shadow", "dry_run", "full"):
        with pytest.raises(cm.CommissioningRefused, match="robot-pc"):
            make_rtde_robot("ur3e", mode, rcfg=RCFG, z_surface=Z_SURFACE, rtde_modules=_spy_modules(record),
                            hostname="dev-pc", arming=arming, gate=gate)
    with pytest.raises(cm.CommissioningRefused, match="authorization"):          # on robot-pc but not armed
        make_rtde_robot("ur3e", "full", rcfg=RCFG, z_surface=Z_SURFACE, rtde_modules=_spy_modules(record),
                        hostname="robot-pc", arming=arming, gate=gate)
    arming.arm("tester", now=time.time())
    with pytest.raises(cm.CommissioningRefused, match="dry_run"):      # armed but the gate was not passed
        make_rtde_robot("ur3e", "full", rcfg=RCFG, z_surface=Z_SURFACE, rtde_modules=_spy_modules(record),
                        hostname="robot-pc", arming=arming, gate=gate)
    assert record == []                                                  # no real-robot interface was ever constructed
    # authorization expired
    tok = json.loads((tmp_path / "arming.json").read_text())
    assert not arming.status(now=tok["authorized_until"] + 1).armed
    assert not arming.status(now=tok["authorized_until"] + 1).armed


def test_ur3e_constructs_only_after_every_check_passes(tmp_path):
    """RTDEReceiveInterface(host) is only called when all conditions are met (an injected spy, no network); shadow does not touch the control interface."""
    record = []
    arming = cm.ArmingToken(tmp_path / "arming.json", ttl_s=60)
    gate = cm.CommissioningGate(tmp_path / "gate.json")
    ev = tmp_path / "summary.json"
    ev.write_text(json.dumps({"run_id": "x", "status": "needs_human", "robot": "ur3e"}))
    gate.record("ur3e", "shadow", ev, "tester")
    fake = FakeRTDE(MujocoRobot(render=False))
    mods = _spy_modules(record)
    mods["rtde_receive"].RTDEReceiveInterface = lambda host: (record.append(("RTDEReceiveInterface", host)), fake.receive)[1]
    arming.arm("tester")
    r = make_rtde_robot("ur3e", "shadow", rcfg=RCFG, z_surface=Z_SURFACE, rtde_modules=mods, hostname="robot-pc",
                        arming=arming, gate=gate)
    assert record == [("RTDEReceiveInterface", "10.20.30.101")] and r.control is None and r.backend_name == "ur3e"
    r.close()


# ---------------------------------------------------------------- graded gate
def test_gate_requires_previous_stage_per_backend(tmp_path):
    gate = cm.CommissioningGate(tmp_path / "gate.json")
    ev = tmp_path / "summary.json"
    ev.write_text("{}")
    assert gate.check("ur3e", "shadow") == (True, "")
    assert not gate.check("ur3e", "dry_run")[0] and not gate.check("ur3e", "full")[0]
    with pytest.raises(ValueError, match="shadow"):
        gate.record("ur3e", "dry_run", ev, "tester")
    with pytest.raises(ValueError, match="evidence"):
        gate.record("ur3e", "shadow", tmp_path / "missing.json", "tester")
    gate.record("ur3e", "shadow", ev, "tester")
    assert gate.check("ur3e", "dry_run")[0] and not gate.check("ur3e", "full")[0]
    gate.record("ur3e", "dry_run", ev, "tester", extra={"max_speed_scale": 0.3})
    assert gate.check("ur3e", "full")[0]
    assert not gate.check("ursim", "dry_run")[0]                      # evidence is kept per backend: fake/ursim ones do not count for the real robot
    assert not gate.check("rtde_fake", "dry_run")[0]
    data = json.loads((tmp_path / "gate.json").read_text())
    assert data["ur3e"]["dry_run"]["max_speed_scale"] == 0.3 and data["ur3e"]["shadow"]["recorded_by"] == "tester"
    assert any("ur3e" in line and "passed" in line for line in gate.status_lines())
    assert cm.main(["--gate-path", str(tmp_path / "gate.json"), "status"]) == 0


def test_fake_backend_can_rehearse_gate(tmp_path):
    gate = cm.CommissioningGate(tmp_path / "gate.json")
    with pytest.raises(cm.CommissioningRefused):
        make("full", gate=gate, enforce_gate=True)
    _, r = make("shadow", gate=gate, enforce_gate=True)             # shadow is always allowed
    r.close()
    _, r = make("full", gate=gate, enforce_gate=False)              # fake does not enforce the gate by default
    r.close()


# ---------------------------------------------------------------- modes
def test_shadow_creates_no_control_interface_and_moves_nothing():
    fake, r = make("shadow")
    ex = executor(r)
    q0, pose0 = r.get_q(), r.get_tcp_pose()
    assert ex.startup_problems() == []
    assert ex.survey().ok
    o = ex.wipe(["B2"], 5.0, "zigzag", 1)
    assert o.status == "rejected" and "shadow" in o.reason and "contact forbidden" in o.reason
    assert fake.control_created == 0 and fake.calls == [] and r.control is None
    assert np.allclose(r.get_q(), q0) and np.allclose(r.get_tcp_pose(), pose0)
    cmds = [e["cmd"] for e in r.shadow_log]
    assert "moveJ" in cmds and "speedStop" in cmds           # the commands the skill would have sent were recorded
    r.close()


def test_dry_run_rejects_contact_and_scales_speed():
    fake, r = make("dry_run")
    ex = executor(r)
    assert r.speed_scale == 0.3 and r.floor_z == pytest.approx(Z_SURFACE + 0.03)
    o = ex.wipe(["B2"], 5.0, "zigzag", 1)
    assert o.status == "rejected" and "dry_run" in o.reason
    assert fake.count("speedL") == 0
    assert ex.closeup("B2").ok                                 # close-look height 0.08 m is above clearance, allowed (moveL for close range)
    assert ex.survey().ok                                      # moveJ to the survey pose
    mv = [c for c in fake.calls if c[0] == "moveL"]
    assert mv and all(c[2] == pytest.approx(0.03 * 0.3) and c[3] == pytest.approx(0.045 * 0.09) for c in mv)
    mj = [c for c in fake.calls if c[0] == "moveJ"]
    assert mj and all(c[2] == pytest.approx(0.18 * 0.3) for c in mj)
    # the backend's own backstop: a command pressing the TCP below clearance is refused (even if the skill layer let it through)
    low = r.get_tcp_pose()
    low[2] = Z_SURFACE + 0.01
    assert not r.move_l(low, 0.03, 0.045) and "below" in r.last_refusal
    hover = r.get_tcp_pose()
    hover[2] = Z_SURFACE + 0.031
    assert r.move_l(hover, 0.03, 0.045)
    assert not r.speed_l([0, 0, -0.003, 0, 0, 0], 0.09, 0.02) and "press" in r.last_refusal
    assert r.speed_l([0, 0, 0.003, 0, 0, 0], 0.09, 0.02)       # upward is allowed, with speed x 0.3 and acceleration x 0.09
    assert fake.calls[-1] == ("speedL", [0, 0, pytest.approx(0.0009), 0, 0, 0], pytest.approx(0.0081), 0.02)
    r.close()


def test_full_mode_wipes_through_fake_rtde_with_force_regulation():
    fake, r = make("full")
    fake.set_latency_ticks(2)                                  # 8 ms command latency
    ex = executor(r)
    fake.robot.dirt.add_blob("light", fake.robot.grid["C3"].center, 0.022, seed=3)
    o = ex.wipe(["C3"], 5.0, "zigzag", 1)
    assert o.ok, o.reason
    f = o.force["C3"]
    assert f["mean_normal_n"] == pytest.approx(5.0, abs=0.3) and f["coverage_ratio"] > 0.95
    assert fake.robot.dirt.zone_stats("C3")["clean"]
    assert fake.count("speedL") > 1000 and fake.count("forceMode") == 0
    r.close()


# ---------------------------------------------------------------- fault injection (skill layer → task loop)
def test_protective_stop_injection_is_a_fault(tmp_path):
    def on_runner(runner, sc):
        sc.fake.at_tick(1500, lambda: sc.fake.inject_protective_stop("injected: protective stop"))

    res = run_sim_scenario("S1_light", "rules", "cv", "topdown", run_root=tmp_path, robot_backend="rtde_fake",
                           on_runner=on_runner)
    s = res["summary"]
    assert s["status"] == "fault" and "protective stop" in s["reason"]
    assert res["judge"]["rtde_calls"]["speedStop"] >= 1 and s["robot_label"] == FAKE_LABEL
    assert s["robot"] == "rtde_fake"


def test_connection_drop_stops_safely_and_asks_human(tmp_path):
    holder = {}

    def on_runner(runner, sc):
        holder["sc"] = sc
        sc.fake.at_tick(1500, sc.fake.drop_connection)

    res = run_sim_scenario("S1_light", "rules", "cv", "topdown", run_root=tmp_path, robot_backend="rtde_fake",
                           on_runner=on_runner)
    s, sc = res["summary"], holder["sc"]
    assert s["status"] == "needs_human" and "RTDE connection lost" in s["reason"]
    assert s["human_requests"] and "RTDE connection lost" in s["human_requests"][-1]["reason"]
    assert sc.io.connection_lost and sc.io.get_robot_mode() == 0
    names = [c[0] for c in sc.fake.calls]
    assert "speedStop" in names[names.index("speedL") + 1:]    # a safe stop was still attempted after the disconnect
    assert np.linalg.norm(sc.robot.get_tcp_speed()[:3]) < 1e-3  # the controller's own watchdog has stopped it
    assert sc.executor.fault is None                            # fault is not latched: after a person checks and reconnects it can continue


def test_connection_drop_during_move_is_aborted_not_fault():
    fake, r = make("full")
    ex = executor(r)
    fake.drop_connection()
    o = ex.survey()
    assert o.status == "aborted" and "RTDE connection lost" in o.reason and ex.fault is None
    r.close()


# ---------------------------------------------------------------- same result as the mujoco backend
def test_s1_light_same_outcome_via_rtde_fake_and_mujoco(tmp_path, quick_run_dir):
    base = json.loads((quick_run_dir / "scenario.json").read_text())
    res = run_sim_scenario("S1_light", "rules", "cv", "topdown", run_root=tmp_path, robot_backend="rtde_fake")
    j = res["judge"]
    assert (j["status"], j["passed"], j["total_passes"]) == (base["status"], base["passed"], base["total_passes"])
    assert j["robot_time_s"] == pytest.approx(base["robot_time_s"], abs=0.5)
    assert j["rtde_calls"]["forceMode"] == 0 and j["rtde_calls"]["speedL"] > 0
    meta = json.loads((tmp_path / res["run_dir"].split("/")[-1] / "reference_meta.json").read_text())
    assert meta["robot"]["label"] == FAKE_LABEL


def test_software_stop_during_async_move_calls_stopL():
    """Software stop received while an async moveL is running: stopL on the next tick, and return failure."""
    fake, r = make("full")

    class Running:
        def isAsyncOperationRunning(self):
            return True

    calls = {"n": 0}
    ctrl = r.control
    orig = ctrl.getAsyncOperationProgressEx

    def progress():
        calls["n"] += 1
        if calls["n"] == 3:
            r.emergency_stop("unit test")
        orig()
        return Running()

    ctrl.getAsyncOperationProgressEx = progress
    pose = r.get_tcp_pose()
    pose[2] += 0.01
    assert not r.move_l(pose, 0.03, 0.045)
    assert fake.count("stopL") == 1 and "software stop" in r.last_refusal
    r.close()


def test_watchdog_kicker_runs_for_real_backends():
    """Real-controller backends (ursim / ur3e) kick the watchdog from a separate thread; the thread exits after closing. A fake interface stands in for ursim, no network."""
    import time

    from cleanproof.robot.backends.rtde_backend import RTDERobot, _reset_single_writer_for_tests

    _reset_single_writer_for_tests()
    fake = FakeRTDE(MujocoRobot(render=False))
    r = RTDERobot("ursim", "full", fake.receive, fake.control_factory, fake.clock, rcfg=fake.robot.rcfg)
    time.sleep(0.3)
    assert r.watchdog_kicks >= 3 and fake.count("kickWatchdog") >= 3
    r.close()
    n = fake.count("kickWatchdog")
    time.sleep(0.2)
    assert fake.count("kickWatchdog") == n and not r._kicker.is_alive()
