"""Closed-loop test of the wiping skill on the MuJoCo backend (the same skill code will be used with URSim and the real robot)."""

import numpy as np
import pytest

from cleanproof.common.config import load_yaml
from cleanproof.common.zones import ZoneGrid
from cleanproof.robot.backends.mujoco_backend import MujocoRobot
from cleanproof.robot.kinematics import rotvec_to_matrix
from cleanproof.robot.skills import SkillExecutor
from cleanproof.sim.contact import Obstacle


def setup(forbidden=(), obstacles=None, grid=None):
    r = MujocoRobot(grid=grid, obstacles=obstacles, render=False)
    return r, SkillExecutor(r, grid=r.grid, forbidden_zones=forbidden)


def test_light_dirt_light_preset_cleans_and_regulates_force():
    r, ex = setup()
    r.dirt.add_blob("light", r.grid["C3"].center, 0.022, seed=3)
    o = ex.wipe(["C3"], 5.0, "zigzag", 1)
    assert o.ok, o.reason
    f = o.force["C3"]
    assert f["mean_normal_n"] == pytest.approx(5.0, abs=0.3)
    assert f["within_1n_ratio"] > 0.9 and f["peak_normal_n"] < 7.0
    assert f["coverage_ratio"] > 0.95 and f["contact_loss_count"] == 0
    assert r.dirt.zone_stats("C3")["clean"]
    assert any(e["event"] == "tare" and e["ok"] for e in o.events)   # tare cancelled the simulated zero offset
    assert r.get_tcp_pose()[2] > r.grid.z_surface + 0.03             # lifted after finishing


def test_dried_dirt_needs_higher_force():
    r, ex = setup()
    r.dirt.add_blob("dried", r.grid["B2"].center, 0.02, seed=5)
    assert ex.wipe(["B2"], 5.0, "zigzag", 1).ok
    assert not r.dirt.zone_stats("B2")["clean"]          # 5 N is below the removal threshold for dried stains
    o = ex.wipe(["B2"], 15.0, "zigzag", 1)
    assert o.ok and o.force["B2"]["mean_normal_n"] == pytest.approx(15.0, abs=0.5)
    assert r.dirt.zone_stats("B2")["clean"]


def test_obstacle_triggers_safety_stop_not_protective_stop():
    grid = ZoneGrid()
    z = grid["C3"]
    # there is a 20 mm high obstacle on the right of the zone; the start point and descent path do not touch it, but while wiping the pad side hits the side wall,
    # tangential force exceeds 20 N, and the safety layer latches a stop (before the controller protective stop)
    obs = [Obstacle(z.xmin, z.center[1] + 0.015, z.xmax, z.ymax + 0.02, 0.02)]
    r, ex = setup(obstacles=obs, grid=grid)
    o = ex.wipe(["C3"], 5.0, "zigzag", 1)
    assert o.status in ("fault", "aborted"), (o.status, o.reason)
    assert any(e["event"] == "safety" for e in o.events)
    assert r.get_safety_mode() == 1                       # the safety layer acts before the controller protective stop
    assert np.linalg.norm(r.get_tcp_speed()[:3]) < 1e-3
    if o.status == "fault":
        assert ex.wipe(["C3"], 5.0, "zigzag", 1).status == "rejected"   # fault latched
        ex.reset_fault()


def test_forbidden_zone_rejected_and_neighbor_margin():
    r, ex = setup(forbidden=["C3"])
    assert ex.wipe(["C3"], 5.0, "zigzag", 1).status == "rejected"
    path = ex._plan_path("C2", "zigzag", None)
    c3 = r.grid["C3"]
    assert path[:, 1].max() <= c3.ymin - ex.pad_r        # the pad edge does not enter the forbidden zone
    assert ex._path_problem(path) is None


def test_unreachable_zone_rejected_before_motion():
    cfg = load_yaml("zones.yaml")
    cfg["work_area"]["origin_xy"] = [0.30, -0.18]
    cfg["work_area"]["size_xy"] = [0.24, 0.30]           # the far end exceeds 0.49 m
    r, ex = setup(grid=ZoneGrid(cfg))
    q0 = r.get_q()
    o = ex.wipe(["D5"], 5.0, "zigzag", 1)
    assert o.status == "rejected" and "horizontal radius" in o.reason
    assert np.allclose(r.get_q(), q0)


def test_protective_stop_during_move_is_fault():
    grid = ZoneGrid()
    z = grid["C3"]
    obs = [Obstacle(z.xmin, z.ymin + 0.02, z.xmax, z.ymax, 0.02)]      # obstacle beside the start point descent path
    r, ex = setup(obstacles=obs, grid=grid)
    o = ex.wipe(["C3"], 5.0, "zigzag", 1)
    assert o.status == "fault" and "protective stop" in o.reason
    assert r.get_safety_mode() == 3


def test_no_contact_when_surface_is_lower_than_configured():
    r, ex = setup()
    r.surface.z_surface = -0.08                          # tabletop is 8 cm lower than configured
    o = ex.wipe(["C3"], 5.0, "zigzag", 1)
    assert o.status == "no_contact"
    assert r.get_safety_mode() == 1


def test_two_passes_and_spot_scrub():
    r, ex = setup()
    o1 = ex.wipe(["A2"], 5.0, "edge_trace", 1)
    o2 = ex.wipe(["A2"], 5.0, "edge_trace", 2)
    assert o1.ok and o2.ok
    assert o2.force["A2"]["contact_duration_s"] > 1.8 * o1.force["A2"]["contact_duration_s"]
    r.dirt.add_blob("light", (r.grid["B4"].xmin + 0.015, r.grid["B4"].ymin + 0.015), 0.008, seed=7)
    o3 = ex.wipe(["B4"], 5.0, "zigzag", 1, bbox_norm=[0.1, 0.1, 0.4, 0.4])
    assert o3.ok and o3.skill == "spot_scrub"
    assert r.dirt.zone_stats("B4")["clean"]


def test_startup_checks_speed_slider():
    r, ex = setup()
    assert ex.startup_problems() == []
    r.speed_fraction = 0.3
    assert any("100%" in p for p in ex.startup_problems())


def test_closeup_centers_wrist_camera_over_zone():
    r, ex = setup()
    assert ex.closeup("B2").ok
    pose = r.get_tcp_pose()
    R = rotvec_to_matrix(pose[3:])
    cam = np.array(ex.cfg["wrist_camera"]["pos_in_flange"])
    tcp = np.array(ex.cfg["startup"]["expected_tcp_offset"][:3])
    cam_base = pose[:3] + R @ (cam - tcp)
    assert np.allclose(cam_base[:2], r.grid["B2"].center, atol=2e-3)
    assert ex.survey().ok
