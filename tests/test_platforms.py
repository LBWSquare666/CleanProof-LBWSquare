"""Platform adapters (cleanproof/robot/platforms.py): assembly, forward/inverse kinematics round trip, Jacobian, home / survey pose, work area reachability.

Needs MuJoCo Menagerie (CLEANPROOF_MENAGERIE or the default path); skipped if absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleanproof.robot.platforms import build_platform_scene, load_platforms, menagerie_root, workspace_points

try:
    menagerie_root()
    HAVE = True
except FileNotFoundError:
    HAVE = False

pytestmark = pytest.mark.skipif(not HAVE, reason="MuJoCo Menagerie is missing")
KEYS = [k for k, s in load_platforms().items() if not s.builtin]


@pytest.fixture(scope="module")
def kins():
    from cleanproof.robot.platforms import PlatformKinematics
    out = {}
    for k in KEYS:
        kin = PlatformKinematics(build_platform_scene(load_platforms()[k]))
        kin.solve_home()
        out[k] = kin
    return out


@pytest.mark.parametrize("key", KEYS)
def test_home_survey_and_workspace_reachable(kins, key):
    kin = kins[key]
    assert kin.home_q is not None and kin.survey_q is not None
    assert kin.limit_margin_deg(kin.home_q) >= 10.0
    for z in (0.0, 0.06):
        for T in workspace_points(z=z):
            q = kin.ik(T, kin.home_q)
            assert q is not None, f"{key}: work area point {T[:3, 3]} has no solution"
            Tq = kin.fk(q)
            assert np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < 1e-4
            assert kin.limit_margin_deg(q) >= 5.0


@pytest.mark.parametrize("key", KEYS)
def test_jacobian_matches_finite_difference(kins, key):
    from cleanproof.robot.kinematics import matrix_to_rotvec
    kin = kins[key]
    q = kin.home_q
    J = kin.jacobian(q)
    T0 = kin.fk(q)
    eps = 1e-6
    for i in range(len(q)):
        dq = np.zeros_like(q)
        dq[i] = eps
        T1 = kin.fk(q + dq)
        assert np.allclose((T1[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-4)
        assert np.allclose(matrix_to_rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps, J[3:, i], atol=1e-4)


@pytest.mark.parametrize("key", KEYS)
def test_twist_to_qdot_moves_tool_as_commanded(kins, key):
    """speedL semantics: given a TCP linear velocity, after integrating the joints the TCP moves in that direction (twist_to_qdot also holds for n != 6)."""
    from cleanproof.robot.kinematics import twist_to_qdot
    kin = kins[key]
    q = kin.home_q.copy()
    p0 = kin.fk(q)[:3, 3]
    v = np.array([0.02, -0.01, 0.0, 0, 0, 0])
    for _ in range(50):
        q = q + twist_to_qdot(kin, q, v) * 0.004
    assert np.allclose(kin.fk(q)[:3, 3] - p0, v[:3] * 0.2, atol=2e-4)


def test_free_yaw_ik_aligns_tool_axis_only(kins):
    """free_yaw (for render re-solving): position and tool z axis aligned, the rotation about the tool axis is unconstrained; it still converges when the target is rotated 1 rad about the tool axis."""
    kin = kins["rizon4"]
    T = kin.fk(kin.home_q)
    c, s = np.cos(1.0), np.sin(1.0)
    Tg = T.copy()
    Tg[:3, :3] = T[:3, :3] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Tg[:3, 3] += [0.03, -0.02, 0.01]
    q = kin.ik(Tg, kin.home_q, max_iter=800, free_yaw=True)
    assert q is not None
    Tq = kin.fk(q)
    assert np.linalg.norm(Tq[:3, 3] - Tg[:3, 3]) < 1e-4
    assert float(Tq[:3, 2] @ Tg[:3, 2]) > 1 - 1e-6
    assert float(np.max(np.abs(q - kin.home_q))) < 0.5          # no big pose change just to align the rotation
