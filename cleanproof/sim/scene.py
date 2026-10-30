"""MuJoCo scene: the arm built link by link from the UR3e nominal DH parameters, the tool, table, stain texture and two cameras.

The geometry uses only simple shapes (capsules / cylinders) for closed-loop tests and schematic rendering; when a realistic look is needed, the server can download
the official UR meshes to replace them. The arm is integrated kinematically in simulation (see mujoco_backend),
and contact forces are computed by the virtual table in sim/contact.py, so none of the geometry here takes part in physical collision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..common.zones import ZoneGrid
from ..robot.kinematics import DEFAULT_TCP_OFFSET, UR3E_DH_A, UR3E_DH_ALPHA, UR3E_DH_D

LINK_RADII = [0.045, 0.040, 0.034, 0.030, 0.030, 0.030]
DIRT_TEXTURE = "dirt"


def _quat_rx(alpha: float) -> str:
    return f"{np.cos(alpha / 2):.9f} {np.sin(alpha / 2):.9f} 0 0"


def camera_xyaxes(pos: np.ndarray, lookat: np.ndarray) -> str:
    """The MuJoCo camera looks along −z, with +y at the top of the image."""
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0]) if abs(f[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return " ".join(f"{v:.6f}" for v in (*x, *y))


def dirt_to_texture(rgb_xy: np.ndarray) -> np.ndarray:
    """Stain map (nx, ny, 3) (i along x, j along y) → MuJoCo plane texture (ny, nx, 3).

    Measured (tests/test_sim_render.py): texture columns correspond to +x, and texture row 0 corresponds to the largest y.
    """
    return np.ascontiguousarray(np.flipud(rgb_xy.transpose(1, 0, 2)))


def build_mjcf(grid: ZoneGrid, sim_cfg: dict[str, Any], dirt_shape: tuple[int, int],
               tcp_offset: np.ndarray = DEFAULT_TCP_OFFSET,
               wrist_cam_pos: tuple[float, float, float] = (0.0, -0.06, 0.16), arm: bool = True) -> str:
    """arm=False: emit only the table, work-area stain texture, fixed camera and lights (for multi-platform support the robot is attached by robot/platforms.py through MjSpec).
    With arm=True the output is identical to before, character for character."""
    pad_r = float(sim_cfg["contact"]["pad_radius_m"])
    cams = sim_cfg["cameras"]
    nx, ny = dirt_shape
    tex_h, tex_w = ny, nx
    zs = grid.z_surface
    wa_c = grid.origin + grid.size / 2
    tcp = np.asarray(tcp_offset, float)

    # arm: body i sits in DH frame i−1 and its joint rotates about its z axis; the child frame is at (a_i, 0, d_i), then rotated by α_i about x
    arm_xml: list[str] = []
    indent = "      "
    for i in (range(6) if arm else ()):
        a, d, al = UR3E_DH_A[i], UR3E_DH_D[i], UR3E_DH_ALPHA[i]
        r = LINK_RADII[i]
        arm_xml.append(f'{indent}<body name="link{i + 1}">')
        arm_xml.append(f'{indent}  <joint name="j{i + 1}" axis="0 0 1" limited="false"/>')
        if abs(a) + abs(d) > 1e-9:
            arm_xml.append(f'{indent}  <geom type="capsule" fromto="0 0 0 {a:.6f} 0 {d:.6f}" size="{r}" material="arm"/>')
        arm_xml.append(f'{indent}  <geom type="sphere" size="{r * 1.05:.4f}" material="joint"/>')
        arm_xml.append(f'{indent}  <frame pos="{a:.6f} 0 {d:.6f}" quat="{_quat_rx(al)}">')
        indent += "    "
    # flange (DH frame 6): gripper + polishing pad + TCP + wrist camera
    wc = cams["wrist"]
    pos_w = " ".join(f"{v:.4f}" for v in wrist_cam_pos)
    arm_xml.append(f'{indent}<body name="flange">')
    arm_xml.append(f'{indent}  <geom type="cylinder" fromto="0 0 0 0 0 0.20" size="0.033" material="tool"/>')
    arm_xml.append(f'{indent}  <geom type="cylinder" fromto="{tcp[0]:.5f} {tcp[1]:.5f} {tcp[2] - 0.05:.5f} '
               f'{tcp[0]:.5f} {tcp[1]:.5f} {tcp[2] - 0.012:.5f}" size="0.012" material="tool"/>')
    arm_xml.append(f'{indent}  <geom name="pad" type="cylinder" fromto="{tcp[0]:.5f} {tcp[1]:.5f} {tcp[2] - 0.012:.5f} '
               f'{tcp[0]:.5f} {tcp[1]:.5f} {tcp[2]:.5f}" size="{pad_r}" material="pad"/>')
    arm_xml.append(f'{indent}  <site name="tcp" pos="{tcp[0]:.5f} {tcp[1]:.5f} {tcp[2]:.5f}" size="0.004" rgba="1 0 0 1"/>')
    arm_xml.append(f'{indent}  <camera name="wrist" pos="{pos_w}" xyaxes="1 0 0 0 -1 0" fovy="{wc["fovy"]}"/>')
    arm_xml.append(f'{indent}</body>')
    for _ in (range(6) if arm else ()):
        indent = indent[:-4]
        arm_xml.append(f'{indent}  </frame>')
        arm_xml.append(f'{indent}</body>')

    if not arm:
        arm_xml = []
    robot_base = (f'    <geom name="robot_base" type="cylinder" fromto="0 0 {zs:.4f} 0 0 0.09" size="0.064" material="arm"/>\n' if arm else "")
    fc = cams["fixed"]
    fixed_pos = " ".join(f"{v:.4f}" for v in fc["pos"])
    xyaxes = camera_xyaxes(np.array(fc["pos"]), np.array(fc["lookat"]))
    table_half = [max(grid.size[0] / 2 + 0.12, 0.3), max(grid.size[1] / 2 + 0.15, 0.35)]
    tc = [wa_c[0], wa_c[1]]
    if getattr(grid, "table_half", None) is not None:       # the site registered the whole tabletop (configs/site/<site>/zones.yaml table)
        table_half = [float(v) for v in grid.table_half]
        tc = [float(v) for v in (grid.table_center if grid.table_center is not None else wa_c)]
    return f"""<mujoco model="cleanproof_ur3e">
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.38 0.38 0.38" specular="0.02 0.02 0.02"/>
  </visual>
  <asset>
    <texture name="{DIRT_TEXTURE}" type="2d" builtin="flat" width="{tex_w}" height="{tex_h}" rgb1="0.8 0.8 0.78"/>
    <material name="{DIRT_TEXTURE}" texture="{DIRT_TEXTURE}" texrepeat="1 1" texuniform="false"/>
    <material name="table" rgba="0.55 0.52 0.48 1"/>
    <material name="arm" rgba="0.72 0.76 0.80 1"/>
    <material name="joint" rgba="0.25 0.45 0.70 1"/>
    <material name="tool" rgba="0.2 0.2 0.22 1"/>
    <material name="pad" rgba="0.95 0.85 0.3 1"/>
  </asset>
  <default><geom contype="0" conaffinity="0"/></default>
  <worldbody>
    <light pos="0.3 0 1.2" dir="0 0 -1" diffuse="0.35 0.35 0.35"/>
    <geom name="table" type="box" pos="{tc[0]:.4f} {tc[1]:.4f} {zs - 0.02:.4f}" size="{table_half[0]:.3f} {table_half[1]:.3f} 0.02" material="table"/>
    <geom name="work_area" type="plane" pos="{wa_c[0]:.4f} {wa_c[1]:.4f} {zs + 0.0005:.4f}" size="{grid.size[0] / 2:.4f} {grid.size[1] / 2:.4f} 0.01" material="{DIRT_TEXTURE}"/>
{robot_base}    <camera name="fixed" pos="{fixed_pos}" xyaxes="{xyaxes}" fovy="{fc['fovy']}"/>
{chr(10).join(arm_xml)}
  </worldbody>
</mujoco>
"""
