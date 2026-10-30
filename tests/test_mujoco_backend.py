import os

import numpy as np
import pytest

from cleanproof.robot.backends.mujoco_backend import MujocoRobot
from cleanproof.robot.io import SAFETY_MODE_PROTECTIVE_STOP, RobotIO
from cleanproof.robot.kinematics import horizontal_radius


def hover(robot: MujocoRobot, x=0.30, y=-0.03, z=0.012):
    pose = robot.get_tcp_pose()
    pose[:3] = [x, y, z]
    assert robot.move_l(pose, 0.05, 0.1)
    return pose


def run(robot, seconds, twist=None, acc=0.09):
    for _ in range(int(round(seconds / robot.control_dt))):
        if twist is not None:
            robot.speed_l(twist, acc, 0.02)
        robot.wait_tick()


def test_implements_robot_io():
    assert isinstance(MujocoRobot(render=False), RobotIO)


def test_move_l_reaches_target():
    r = MujocoRobot(render=False)
    p = hover(r)
    assert np.allclose(r.get_tcp_pose()[:3], p[:3], atol=1e-4)
    assert r.get_tcp_pose()[2] > r.grid.z_surface      # hovering, no contact
    assert abs(r.get_tcp_force()[2] - r.ft_bias[2]) < 0.5


def test_press_down_force_follows_stiffness():
    r = MujocoRobot(render=False)
    hover(r, z=0.004)
    run(r, 2.0, [0, 0, -0.003, 0, 0, 0])          # contact after about 1.33 s, then press about 2 mm further
    run(r, 0.3)                                    # decelerate and stop after the command expires
    depth = r.grid.z_surface - r.get_tcp_pose()[2]
    fz = r.get_tcp_force()[2] - r.ft_bias[2]
    assert depth > 0.001
    assert fz == pytest.approx(r.surface.k * depth, rel=0.1, abs=0.3)


def test_speedl_expires_without_new_command():
    r = MujocoRobot(render=False)
    hover(r, z=0.05)
    run(r, 0.5, [0.01, 0, 0, 0, 0, 0])
    assert np.linalg.norm(r.get_tcp_speed()[:3]) > 0.005
    run(r, 0.5)
    assert np.linalg.norm(r.get_tcp_speed()[:3]) < 1e-4


def test_sliding_contact_removes_light_dirt_only_where_wiped():
    r = MujocoRobot(render=False)
    r.dirt.add_rect("light", 0.27, -0.10, 0.33, 0.04)
    zone_before = r.dirt.zone_stats("C2")["residue_fraction"]
    hover(r, x=0.30, y=-0.09, z=0.002)
    run(r, 1.5, [0, 0, -0.003, 0, 0, 0])
    run(r, 6.0, [0, 0.009, 0, 0, 0, 0])            # slide about 5 cm along +y
    lv = r.dirt.level[r.dirt._tid("light")]
    wiped = (np.abs(r.dirt.cx - 0.30) < 0.02) & (r.dirt.cy > -0.07) & (r.dirt.cy < -0.05)
    untouched = (np.abs(r.dirt.cx - 0.30) < 0.02) & (r.dirt.cy > 0.02) & (r.dirt.cy < 0.04)
    assert lv[wiped].mean() < 0.2 and lv[untouched].mean() > 0.9
    assert zone_before > 0


def test_unreachable_move_is_rejected():
    r = MujocoRobot(render=False)
    start = r.get_tcp_pose()
    pose = start.copy()
    pose[:2] = [0.50, -0.20]
    assert horizontal_radius(pose) > 0.53
    assert r.move_l(pose, 0.05, 0.1) is False      # IK failure: the UR controller refuses it as well
    assert horizontal_radius(r.get_tcp_pose()) < 0.49


def test_protective_stop_on_excessive_force():
    r = MujocoRobot(render=False)
    hover(r, z=0.004)
    run(r, 4.0, [0, 0, -0.01, 0, 0, 0], acc=0.5)   # hard press without force control: about 3.6 cm penetration → far above 60 N
    assert r.get_safety_mode() == SAFETY_MODE_PROTECTIVE_STOP
    assert "contact force" in r.protective_stop_reason
    assert r.speed_l([0.01, 0, 0, 0, 0, 0], 0.1, 0.02) is False


def test_move_j_survey_and_back():
    r = MujocoRobot(render=False)
    home = r.get_q()
    assert r.move_j(r.grid.survey_q, 0.18, 0.108)
    assert np.allclose(r.get_q(), r.grid.survey_q, atol=1e-9)
    assert r.move_j(home, 0.18, 0.108)


def _gl_available() -> bool:
    return bool(os.environ.get("MUJOCO_GL")) and (os.environ.get("MUJOCO_GL") != "glfw" or bool(os.environ.get("DISPLAY")))


@pytest.mark.render
@pytest.mark.skipif(not _gl_available(), reason="needs MUJOCO_GL=egl (server) or glfw + DISPLAY")
def test_render_shows_dirt_at_projected_location():
    r = MujocoRobot()
    r.move_j(r.grid.survey_q, 0.5, 1.0)
    clean = r.render("fixed").astype(int)
    r.dirt.add_rect("permanent", 0.29, -0.05, 0.33, -0.01)
    dirty = r.render("fixed").astype(int)
    cam = r.camera_model("fixed")
    uv, _ = cam.project(np.array([[0.31, -0.03, r.grid.z_surface]]))
    u, v = np.round(uv[0]).astype(int)
    assert np.abs(dirty[v, u] - clean[v, u]).sum() > 100
    far_uv, _ = cam.project(np.array([[0.20, 0.10, r.grid.z_surface]]))
    fu, fv = np.round(far_uv[0]).astype(int)
    assert np.abs(dirty[fv, fu] - clean[fv, fu]).sum() < 10
    r.close()
