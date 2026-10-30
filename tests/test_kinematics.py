import os
import json
from pathlib import Path

import numpy as np
import pytest

from cleanproof.robot.kinematics import (
    UR3eKinematics,
    elbow_margin_deg,
    matrix_to_rotvec,
    pose_to_matrix,
    rotvec_to_matrix,
    twist_to_qdot,
)

# Real-robot Home record (recorded UR3e home pose, read via RTDE on 2026-09-18)
HOME_Q = np.array([0.2784169614315033, -1.1763106149486084, -1.794198989868164,
                   -1.7396394215025843, 1.5636017322540283, -6.791584618875774])
HOME_TCP = np.array([0.2312768881895341, -0.07433365285334827, 0.06192041126213654,
                     2.9010442185828293, -1.1874544824069944, 0.008989094351195465])


def test_fk_matches_real_home_record():
    kin = UR3eKinematics()
    T = kin.fk(HOME_Q)
    assert np.linalg.norm(T[:3, 3] - HOME_TCP[:3]) < 0.003  # nominal DH vs factory calibration
    dR = T[:3, :3].T @ rotvec_to_matrix(HOME_TCP[3:])
    assert np.degrees(np.linalg.norm(matrix_to_rotvec(dR))) < 1.0


def test_rotvec_roundtrip_including_near_pi():
    rng = np.random.default_rng(0)
    for _ in range(50):
        rv = rng.normal(size=3)
        rv = rv / np.linalg.norm(rv) * rng.uniform(0, np.pi - 1e-3)
        assert np.allclose(matrix_to_rotvec(rotvec_to_matrix(rv)), rv, atol=1e-6)
    R = rotvec_to_matrix([np.pi, 0, 0])
    assert np.allclose(rotvec_to_matrix(matrix_to_rotvec(R)), R, atol=1e-6)


def test_jacobian_matches_finite_difference():
    kin = UR3eKinematics()
    J = kin.jacobian(HOME_Q)
    eps = 1e-6
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1, T0 = kin.fk(HOME_Q + dq), kin.fk(HOME_Q - dq)
        v = (T1[:3, 3] - T0[:3, 3]) / (2 * eps)
        w = matrix_to_rotvec(T1[:3, :3] @ T0[:3, :3].T) / (2 * eps)
        assert np.allclose(J[:3, i], v, atol=1e-5)
        assert np.allclose(J[3:, i], w, atol=1e-5)


def test_ik_recovers_nearby_pose():
    kin = UR3eKinematics()
    target = kin.fk(HOME_Q + np.array([0.05, -0.04, 0.03, 0.02, -0.05, 0.1]))
    q = kin.ik(target, HOME_Q)
    assert q is not None
    assert np.allclose(kin.fk(q), target, atol=1e-4)


def test_twist_to_qdot_produces_requested_tcp_velocity():
    kin = UR3eKinematics()
    tw = np.array([0.005, -0.003, -0.002, 0.0, 0.0, 0.01])
    qd = twist_to_qdot(kin, HOME_Q, tw)
    assert np.allclose(kin.jacobian(HOME_Q) @ qd, tw, atol=1e-6)


def test_home_elbow_margin():
    assert elbow_margin_deg(HOME_Q) > 15.0


@pytest.mark.skipif(not Path(os.environ.get("UR3E_START_POSE", "ur3e_start_pose.json")).exists(),
                    reason="recorded UR3e home pose file is not on this machine")
def test_home_constants_match_reference_file():
    rec = json.loads(Path(os.environ.get("UR3E_START_POSE", "ur3e_start_pose.json")).read_text())
    assert np.allclose(rec["joint_positions"], HOME_Q) and np.allclose(rec["tcp_pose"], HOME_TCP)


def test_pose_matrix_roundtrip():
    T = pose_to_matrix(HOME_TCP)
    from cleanproof.robot.kinematics import matrix_to_pose

    assert np.allclose(matrix_to_pose(T), HOME_TCP, atol=1e-9)
