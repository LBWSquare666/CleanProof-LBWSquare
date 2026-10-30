"""Scene dressing for the multi-platform section (project docs, section C: slope = wide handrail of a lobby staircase, vertical surface = wall, flat surface = factory line and tabletop, each with a different background).

Approach ("mounted facing the surface"): the unit of robot + work surface + inspection camera stays untouched in simulation (simulation gravity = 0, purely kinematic); only the scenery: floor, wall, stairs,
conveyor, lights, and the display camera are rotated around this unit to the matching orientation. So:
- the run (Cosmos 3 judging, Nemotron decisions, force-controlled wiping) is exactly the same as on a horizontal tabletop, and the work surface Cosmos sees is the work surface in the picture;
- visually it looks like a wall-mounted arm wiping a wall, and an arm riding on a handrail wiping a sloped handrail;
 "non-horizontal surfaces are handled by mounting facing the surface; a general work-surface frame is not implemented".

Coordinates: the scenery frame e (z up, floor z = 0) relates to the robot world frame w by p_e = R_ew · p_w + t_e. Geometry, lights and cameras are all written in the scenery frame and converted to w when modeling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

S30, C30 = math.sin(math.radians(30)), math.cos(math.radians(30))


@dataclass
class EnvSpec:
    name: str
    label: str                                    # scene name in the console / video
    R_ew: np.ndarray                              # columns = directions of the world-frame basis vectors in the scenery frame
    t_e: tuple[float, float, float]               # position of the world origin (robot base) in the scenery frame
    table_half: tuple[float, float]               # half size of the work-surface texture (material extension), world x / y
    camera: dict[str, Any]                        # display camera in the scenery frame: lookat / azimuth / elevation / distance / fovy
    build: Callable[["EnvBuilder"], None]
    key_light: tuple[float, float, float] = (2.0, -2.5, 3.6)   # key light position in the scenery frame (pointing at the workspace)
    mug: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class EnvBuilder:
    """Add geometry to the MjSpec worldbody in scenery-frame coordinates (visual only, no contact)."""

    def __init__(self, spec_parent: Any, mujoco: Any, env: EnvSpec, grid: Any):
        self.sp, self.mj, self.env, self.grid = spec_parent, mujoco, env, grid
        self.R = np.asarray(env.R_ew, float)
        self.t = np.asarray(env.t_e, float)
        self._n = 0
        self._mats: set[str] = set()
        self.base = (0.0, 0.0, 0.0)                    # position of the robot base in the world frame (filled in by the caller per platform)

    # ---- Coordinate conversion
    def to_w(self, p_e) -> np.ndarray:
        return self.R.T @ (np.asarray(p_e, float) - self.t)

    def to_e(self, p_w) -> np.ndarray:
        return self.R @ np.asarray(p_w, float) + self.t

    def quat_w(self, R_local_e: Optional[np.ndarray] = None) -> list[float]:
        """Orientation in the scenery frame (3×3, columns = directions of the geometry's local axes in the scenery frame) → world-frame quaternion (w, x, y, z)."""
        import mujoco
        Rl = np.eye(3) if R_local_e is None else np.asarray(R_local_e, float)
        Rw = self.R.T @ Rl
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rw.reshape(-1))
        return q.tolist()

    # ---- Materials
    def material(self, name: str, rgb: tuple[float, float, float], specular: float = 0.0, shininess: float = 0.0,
                 reflectance: float = 0.0, texture: Optional[dict[str, Any]] = None) -> str:
        if name in self._mats:
            return name
        mj = self.mj
        m = self.sp.add_material(name=name)
        m.rgba = [rgb[0], rgb[1], rgb[2], 1.0]
        m.specular, m.shininess, m.reflectance = specular, shininess, reflectance
        if texture is not None:
            tname = name + "_tex"
            kind = texture.get("kind", "checker")
            if kind == "flat":
                self.sp.add_texture(name=tname, type=mj.mjtTexture.mjTEXTURE_2D, builtin=mj.mjtBuiltin.mjBUILTIN_FLAT,
                                    width=16, height=16, rgb1=list(texture["rgb1"]))
            elif kind == "file":
                self.sp.add_texture(name=tname, type=mj.mjtTexture.mjTEXTURE_2D, file=str(texture["file"]))
            else:
                self.sp.add_texture(name=tname, type=mj.mjtTexture.mjTEXTURE_2D, builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
                                    width=int(texture.get("px", 512)), height=int(texture.get("px", 512)),
                                    rgb1=list(texture["rgb1"]), rgb2=list(texture["rgb2"]))
            m.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = tname
            m.texrepeat = list(texture.get("repeat", (1, 1)))
            m.texuniform = bool(texture.get("uniform", True))
            m.rgba = [1, 1, 1, 1]
        self._mats.add(name)
        return name

    def _name(self) -> str:
        self._n += 1
        return f"env_{self.env.name}_{self._n}"

    # ---- Geometry (sizes in meters; c = center in the scenery frame)
    def box(self, c, half, mat: str, R_local: Optional[np.ndarray] = None) -> None:
        self.sp.worldbody.add_geom(name=self._name(), type=self.mj.mjtGeom.mjGEOM_BOX, pos=self.to_w(c).tolist(), quat=self.quat_w(R_local),
                                   size=list(half), material=mat, contype=0, conaffinity=0)

    def cyl(self, p0, p1, r: float, mat: str) -> None:
        a, b = self.to_w(p0), self.to_w(p1)
        self.sp.worldbody.add_geom(name=self._name(), type=self.mj.mjtGeom.mjGEOM_CYLINDER, fromto=[*a, *b], size=[r, 0, 0],
                                   material=mat, contype=0, conaffinity=0)

    def plane(self, c, half, mat: str, R_local: Optional[np.ndarray] = None, grid: float = 0.04) -> None:
        """Subdivided plane (the old MuJoCo pipeline computes lighting per vertex: one face of a large box has only 4 vertices and goes dark all over under a spotlight; the plane is subdivided by grid so lighting matches the work surface).
        Local z = normal (towards the visible side)."""
        self.sp.worldbody.add_geom(name=self._name(), type=self.mj.mjtGeom.mjGEOM_PLANE, pos=self.to_w(c).tolist(), quat=self.quat_w(R_local),
                                   size=[half[0], half[1], grid], material=mat, contype=0, conaffinity=0)

    def ell(self, c, radii, mat: str) -> None:
        self.sp.worldbody.add_geom(name=self._name(), type=self.mj.mjtGeom.mjGEOM_ELLIPSOID, pos=self.to_w(c).tolist(), quat=self.quat_w(),
                                   size=list(radii), material=mat, contype=0, conaffinity=0)

    def box_w(self, c_w, half, mat: str) -> None:
        """Box placed directly in the world frame (robot unit frame): base mounting plate, riding cart and other things that move with the unit."""
        self.sp.worldbody.add_geom(name=self._name(), type=self.mj.mjtGeom.mjGEOM_BOX, pos=list(c_w), size=list(half), material=mat,
                                   contype=0, conaffinity=0)

    def light(self, pos_e, target_e, diffuse: float, shadow: bool, spot: bool = True, cutoff: float = 45.0) -> None:
        p, q = self.to_w(pos_e), self.to_w(target_e)
        d = (q - p) / np.linalg.norm(q - p)
        lt = self.sp.worldbody.add_light(name=self._name(), pos=p.tolist(), dir=d.tolist(), diffuse=[diffuse] * 3,
                                         specular=[0.02] * 3, castshadow=shadow)
        if spot:
            lt.type = self.mj.mjtLightType.mjLIGHT_SPOT
            lt.cutoff, lt.exponent, lt.bulbradius = cutoff, 1.0, 0.3


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


# ------------------------------------------------------------------------------------------------ Corridor wall (UR5e wall-mounted)
def _build_wall(b: EnvBuilder, plate: bool = True) -> None:
    wall = b.material("wall_paint", (1, 1, 1), texture={"kind": "flat", "rgb1": (236 / 255, 238 / 255, 236 / 255)})   # goes through the same texture shading path as the whiteboard work surface (no visible seam)
    floor = b.material("corridor_floor", (0.8, 0.8, 0.8), specular=0.15, shininess=0.4, reflectance=0.04,
                       texture={"rgb1": (0.78, 0.77, 0.74), "rgb2": (0.74, 0.73, 0.70), "repeat": (6, 6), "px": 256})
    skirting = b.material("skirting", (0.55, 0.56, 0.58))
    rail = b.material("corridor_rail", (0.62, 0.48, 0.34), specular=0.2, shininess=0.3)
    door = b.material("door", (0.70, 0.64, 0.56))
    frame = b.material("door_frame", (0.45, 0.47, 0.50))
    side = b.material("side_wall", (0.90, 0.90, 0.88))
    plate_m = b.material("mount_plate", (0.30, 0.31, 0.33), specular=0.3, shininess=0.4)
    b.plane((0, -3.0, 0.0), (8, 3.0), floor)                                       # floor (in front of the wall)
    b.plane((0, 0.0004, 1.5), (6, 1.5), wall, np.column_stack([(1, 0, 0), (0, 0, 1), (0, -1, 0)]).astype(float))                              # wall surface (same plane as the work surface, slightly behind)
    b.box((0, -0.008, 0.06), (6, 0.008, 0.06), skirting)                           # skirting board
    b.box((2.2, -0.03, 0.88), (3.8, 0.03, 0.03), rail)                             # handrail strip of a hospital corridor (ends at the door)
    for x in (-2.2,):                                                              # a door
        b.box((x, -0.012, 1.05), (0.48, 0.012, 1.05), frame)
        b.box((x, -0.018, 1.02), (0.44, 0.008, 1.0), door)
    b.box((3.4, 2.0, 1.5), (0.06, 2.0, 1.5), side)                                 # side wall at the corridor corner
    if plate:
        b.box_w((b.base[0], b.base[1], -0.004), (0.11, 0.11, 0.006), plate_m)      # wall-mounted base plate (moves with the unit: under the base)
    b.light((1.2, -2.8, 2.9), (0.0, 0.0, 1.15), 0.22, True, cutoff=80)        # cone angle enlarged: at 40° the cone edge falls in the upper right of the frame and moire shadows appear on the wall
    b.light((-2.0, -3.0, 2.6), (0.0, 0.0, 1.0), 0.10, False, spot=False)


# Corridor wall + floor-cleaning robot cart ("nobody mounts a robot arm sideways on a wall"; the Rizon 4 sits on top of the cart with its base up, wiping the wall).
# The wall and work surface are the same as WALL (same job W1_wall_marker, same stains and wipe path); the robot base moves to the cart top (scenery frame CART_XY / CART_H),
# and at render time IK is re-solved frame by frame along the TCP trajectory of the same job (platform_render._render_env).
CART_XY, CART_H = (-0.03, -0.55), 1.15          # position of the cart in the scenery frame and height of its top surface (floor-cleaning robot with a lifting column)
_TILT = math.radians(20.0)                       # the upper edge of a minivan side window tilts 20° towards the inside of the car
_ST, _CT = math.sin(_TILT), math.cos(_TILT)
_WIN_R = np.column_stack([(0, -_ST, -_CT), (1, 0, 0), (0, -_CT, _ST)]).astype(float)   # world x = down along the glass, world y = along the car body, world z = outward normal of the glass
_WIN_T = np.array([0.0, 0.30 * _ST, 1.55 + 0.30 * _CT])   # workspace center (world (0.30, −0.03, 0)) falls at scenery (−0.03, 0, 1.55)
_WALL_R = np.column_stack([(0, 0, -1), (1, 0, 0), (0, -1, 0)]).astype(float)
_WALL_T = np.array([0.0, 0.0, 1.45])


def _cart_base_world(R: np.ndarray = _WIN_R, T: np.ndarray = _WIN_T) -> tuple[list[float], list[float]]:
    """Position and orientation (quaternion w,x,y,z) of the cart-top base in the world frame (robot unit frame): base z axis = scenery +z (up), x axis = towards the car (scenery +y)."""
    p_e = np.array([CART_XY[0], CART_XY[1], CART_H])
    p_w = R.T @ (p_e - T)
    z_b = R.T @ np.array([0.0, 0.0, 1.0])
    x_b = R.T @ np.array([0.0, 1.0, 0.0])
    x_b = x_b - z_b * float(x_b @ z_b)
    x_b /= np.linalg.norm(x_b)
    y_b = np.cross(z_b, x_b)
    M = np.column_stack([x_b, y_b, z_b])
    tr = M[0, 0] + M[1, 1] + M[2, 2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (M[2, 1] - M[1, 2]) / S, (M[0, 2] - M[2, 0]) / S, (M[1, 0] - M[0, 1]) / S]
    elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
        S = math.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
        q = [(M[2, 1] - M[1, 2]) / S, 0.25 * S, (M[0, 1] + M[1, 0]) / S, (M[0, 2] + M[2, 0]) / S]
    elif M[1, 1] > M[2, 2]:
        S = math.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
        q = [(M[0, 2] - M[2, 0]) / S, (M[0, 1] + M[1, 0]) / S, 0.25 * S, (M[1, 2] + M[2, 1]) / S]
    else:
        S = math.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
        q = [(M[1, 0] - M[0, 1]) / S, (M[0, 2] + M[2, 0]) / S, (M[1, 2] + M[2, 1]) / S, 0.25 * S]
    return p_w.tolist(), q


def _build_cart(b: EnvBuilder) -> None:
    """Floor-cleaning robot cart (chassis + wheels + bumper + white body + teal light strip + top plate), the arm base is mounted on top."""
    dark = b.material("wc_dark", (0.18, 0.19, 0.21), specular=0.3, shininess=0.4)
    bot = b.material("wc_bot", (0.93, 0.94, 0.95), specular=0.3, shininess=0.5)
    metal = b.material("wc_metal", (0.45, 0.47, 0.50), specular=0.4, shininess=0.5)
    teal = b.material("wc_teal", (0.06, 0.46, 0.43))
    cx, cy = CART_XY
    H = CART_H
    b.box((cx, cy, 0.12), (0.24, 0.26, 0.10), dark)                                  # chassis
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.cyl((cx + sx * 0.245, cy + sy * 0.17, 0.06), (cx + sx * 0.265, cy + sy * 0.17, 0.06), 0.06, dark)
    b.box((cx, cy + 0.265, 0.10), (0.22, 0.012, 0.05), metal)                         # bumper on the side facing the car
    hb = (H - 0.24 - 0.02) / 2
    b.box((cx, cy, 0.24 + hb), (0.21, 0.23, hb), bot)                                  # body
    b.box((cx, cy - 0.232, 0.60), (0.17, 0.003, 0.012), teal)                          # status light strip (on the side facing the camera)
    b.box((cx, cy, H - 0.012), (0.22, 0.24, 0.012), dark)                              # top plate (arm mounting surface)


def _build_van(b: EnvBuilder) -> None:
    """White minivan (MPV) in a car wash bay: lower body vertical, the upper edge of the side window tilts 20° towards the inside; the glass surface = work surface (same color as the glass surface of G1_window_glass),
     "change wiping the wall to wiping a car ... wiping a slanted glass window is enough". The floor-cleaning cart (arm on top) is parked outside the car."""
    floor = b.material("cw_floor", (0.70, 0.71, 0.72), specular=0.25, shininess=0.5, reflectance=0.05,
                       texture={"rgb1": (0.72, 0.73, 0.74), "rgb2": (0.68, 0.69, 0.70), "repeat": (8, 8), "px": 256})
    glassm = b.material("cw_glass", (1, 1, 1), texture={"kind": "flat", "rgb1": (128 / 255, 146 / 255, 156 / 255)})   # same shading path as the work surface
    body = b.material("cw_body", (0.93, 0.94, 0.93), specular=0.5, shininess=0.7)
    dark = b.material("cw_dark", (0.16, 0.17, 0.19), specular=0.5, shininess=0.7)
    tyre = b.material("cw_tyre", (0.10, 0.10, 0.11))
    rim = b.material("cw_rim", (0.62, 0.64, 0.67), specular=0.6, shininess=0.8)
    seam = b.material("cw_seam", (0.55, 0.57, 0.58))
    wall = b.material("cw_wall", (0.88, 0.89, 0.90))
    stripe = b.material("cw_stripe", (0.85, 0.72, 0.20))
    lamp = b.material("cw_lamp", (0.97, 0.97, 0.95))
    b.plane((0, 0, 0.0), (9, 9), floor)
    for x in (-3.2, 3.2):
        b.box((x, -1.2, 0.003), (0.04, 2.2, 0.003), stripe)                            # wash bay boundary line
    b.box((0, 4.2, 2.2), (6, 0.08, 2.2), wall)
    b.box((-5.5, 0, 2.2), (0.08, 6, 2.2), wall)
    for x in (-2.0, 0.0, 2.0):
        b.box((x, 1.0, 3.6), (0.5, 0.06, 0.03), lamp)
    L = 2.45
    z_lo, z_hi = 1.28, 1.95                                                           # side window glass band
    y_lo, y_hi = math.tan(_TILT) * (z_lo - 1.55), math.tan(_TILT) * (z_hi - 1.55)
    Rg = np.column_stack([(1, 0, 0), (0, _ST, _CT), (0, -_CT, _ST)]).astype(float)   # local x along the car body, y up along the glass, z outward normal
    zc = (z_lo + z_hi) / 2
    yc = math.tan(_TILT) * (zc - 1.55)
    half_s = (z_hi - z_lo) / 2 / _CT
    b.plane((-0.05, yc + 0.0004 * _CT, zc - 0.0004 * _ST), (2.05, half_s), glassm, Rg)  # glass (right behind the work surface)
    for xp, w in ((0.62, 0.045), (-0.95, 0.045), (2.12, 0.07), (-2.15, 0.10)):         # window pillars (outside the workspace x −0.18…0.12)
        b.box((xp, yc - 0.004 * _CT, zc + 0.004 * _ST), (w, half_s, 0.004), dark, Rg)
    b.box((-0.05, y_hi - 0.01, z_hi + 0.02), (2.2, 0.03, 0.025), dark)               # window top edge
    b.box((-0.05, y_lo - 0.01, z_lo - 0.015), (2.2, 0.02, 0.015), dark)              # window bottom edge
    b.box((0.0, y_lo + 0.95, (0.32 + z_lo) / 2), (L, 0.95, (z_lo - 0.32) / 2), body)   # lower body (side vertical, y = y_lo)
    b.box((0.0, y_hi + 0.80, z_hi + 0.06), (L - 0.25, 0.80, 0.08), body)              # roof
    b.box((0.0, (y_lo + y_hi) / 2 + 0.90, zc), (L - 0.05, 0.86, (z_hi - z_lo) / 2), body)   # cabin (behind the glass)
    for xs in (0.55, -0.95):                                                          # door seam
        b.box((xs, y_lo - 0.003, (0.40 + z_lo) / 2), (0.006, 0.002, (z_lo - 0.40) / 2), seam)
    b.box((0.72, y_lo - 0.012, 1.10), (0.09, 0.008, 0.015), dark)                     # door handle
    b.box((0.0, y_lo - 0.004, 0.42), (L, 0.006, 0.05), dark)                          # lower skirt
    for x in (-1.60, 1.65):
        b.cyl((x, y_lo - 0.03, 0.34), (x, y_lo + 0.20, 0.34), 0.34, tyre)
        b.cyl((x, y_lo - 0.035, 0.34), (x, y_lo - 0.025, 0.34), 0.20, rim)
    b.light((1.4, -2.8, 3.6), (0.0, 0.0, 1.4), 0.22, True, cutoff=70)
    b.light((-2.0, -3.0, 3.0), (0.0, 0.0, 1.2), 0.10, False, spot=False)
    _build_cart(b)


VAN_CART = EnvSpec(
    name="carwash_van", label="Car wash · van window",
    R_ew=_WIN_R, t_e=tuple(_WIN_T), table_half=(0.25, 0.35),
    camera={"lookat": (-0.10, -0.40, 1.25), "azimuth": -50.0, "elevation": 12.0, "distance": 4.2, "fovy": 34.0},
    build=_build_van,
    extra={"base_world": _cart_base_world()},
)


WALL = EnvSpec(
    name="corridor_wall", label="Corridor wall",
    R_ew=np.column_stack([(0, 0, -1), (1, 0, 0), (0, -1, 0)]).astype(float),     # columns: x_w → down, y_w → horizontal along the wall, z_w → wall normal (outward)
    t_e=(0.0, 0.0, 1.45),
    table_half=(0.3, 0.35),
    camera={"lookat": (-0.20, -0.25, 1.30), "azimuth": -58.0, "elevation": 6.0, "distance": 2.7, "fovy": 34.0},   # nearly facing the wall: floor and skirting at the bottom of the frame, handrail strip horizontal (the original camera position looked like a top-down view of the floor)
    build=_build_wall,
)


# ------------------------------------------------------------------------------------------------ Lobby staircase wide handrail (Rizon 4 rides on the handrail, 30°)
RAIL_H0 = 1.6          # handrail top height at the robot base (scenery frame)
RAIL_DROP = 0.92       # vertical distance from the handrail top to the front edge of the stair treads


def _rail_top_z(y_e: float) -> float:
    return RAIL_H0 - y_e * S30 / C30


def _build_rail(b: EnvBuilder) -> None:
    stone = b.material("lobby_stone", (0.86, 0.84, 0.80), specular=0.25, shininess=0.5, reflectance=0.06,
                       texture={"rgb1": (0.88, 0.86, 0.82), "rgb2": (0.83, 0.81, 0.77), "repeat": (8, 8), "px": 256})
    step = b.material("stair_step", (0.80, 0.78, 0.74), specular=0.2, shininess=0.4)
    riser = b.material("stair_riser", (0.70, 0.68, 0.64))
    glass = b.material("balustrade_glass", (0.80, 0.88, 0.90), specular=0.6, shininess=0.8)
    post = b.material("balustrade_post", (0.55, 0.56, 0.58), specular=0.5, shininess=0.6)
    wood = b.material("rail_wood_side", (0.55, 0.42, 0.30), specular=0.15, shininess=0.3)
    wall = b.material("lobby_wall", (0.92, 0.91, 0.88))
    window = b.material("lobby_window", (0.80, 0.86, 0.90), specular=0.4, shininess=0.6)
    column = b.material("lobby_column", (0.88, 0.87, 0.84))
    cart = b.material("rail_carriage", (0.28, 0.29, 0.31), specular=0.3, shininess=0.4)
    b.plane((0, 0, 0.0), (9, 9), stone)                                           # lobby floor
    # Stairs: going up along −y (the handrail is on the −x side of the stairs); each tread 0.28 m deep, 0.16 m high (slope about 30°)
    run, rise = 0.28, 0.16
    y_bot = (RAIL_H0 - RAIL_DROP) * C30 / S30                                     # where the tread front edge line meets the floor
    k = 0
    while True:
        y0 = y_bot - k * run
        z = (k + 1) * rise
        if z > 3.0:
            break
        b.box((0.95, y0 - run / 2, z / 2), (0.75, run / 2, z / 2), riser)
        b.box((0.95, y0 - run / 2, z - 0.008), (0.76, run / 2 + 0.012, 0.010), step)
        k += 1
    # Handrail: top surface (the part outside the work-surface texture is filled out with same-color wooden strips on the sides) + glass balustrade + posts
    L = 6.0
    for s_ in np.arange(-1.5, 3.4, 0.6):                                          # posts: vertical from the tread line to the handrail underside
        y = s_ * C30
        zt = _rail_top_z(y) - 0.07
        zb = zt - RAIL_DROP + 0.05
        b.cyl((0.0, y, zb), (0.0, y, zt), 0.018, post)
    Rg = rot_x(-30.0)                                                              # balustrade parallel to the handrail (along the slope)
    yc = 0.9
    b.box((0.0, yc, _rail_top_z(yc) - 0.07 - (RAIL_DROP - 0.12) / 2 / C30), (0.008, L / 2, (RAIL_DROP - 0.12) / 2), glass, Rg)
    b.box((0.0, yc, _rail_top_z(yc) - 0.048), (0.20, L / 2, 0.035), wood, Rg)       # handrail wooden body (below the top-surface texture, not coplanar)
    # Background: lobby back wall, high windows, columns
    b.box((-3.2, 0, 3.0), (0.1, 9, 3.0), wall)
    for y in (-3.0, -0.6, 1.8, 4.2):
        b.box((-3.09, y, 2.6), (0.01, 0.9, 1.7), window)
    for y in (-1.8, 3.0):
        b.box((-2.0, y, 3.0), (0.25, 0.25, 3.0), column)
    # Cart riding on the handrail (moves with the unit: under the base, skirts on both sides of the handrail)
    bx = b.base[0]
    b.box_w((bx, -0.03, -0.004), (0.15, 0.225, 0.006), cart)
    for sy in (-0.245, 0.185):
        b.box_w((bx, sy, -0.035), (0.15, 0.014, 0.045), cart)
    b.light((2.4, 2.2, 4.2), (0.0, 0.6, 1.4), 0.24, True, cutoff=42)
    b.light((-2.0, -2.0, 4.0), (0.0, 0.0, 1.2), 0.10, False, spot=False)


RAIL = EnvSpec(
    name="lobby_rail", label="Lobby stair handrail",
    R_ew=np.column_stack([(0, C30, -S30), (-1, 0, 0), (0, S30, C30)]).astype(float),
    t_e=(0.0, 0.0, RAIL_H0),
    table_half=(1.6, 0.20),
    camera={"lookat": (0.0, 0.05, 1.55), "azimuth": 38.0, "elevation": 18.0, "distance": 3.3, "fovy": 34.0},
    build=_build_rail,
)


# ------------------------------------------------------------------------------------------------ Stair handrail + ramp rail cart (UR5e on top of a control cabinet)
# Scenery frame: stairs go down along +y (slope 30°); from −x to +x: stairs | bicycle ramp (a small rail in the middle, the cabinet cart rides on it) | balustrade and wide handrail.
# Unit (robot + work surface): handrail top = work surface plane (world z = 0), extending along world y; the robot base is at the world origin (cabinet top), the workspace is 0.3 m along its +x (on the handrail top).
# The cabinet stands on the ramp tilted with the slope, its top parallel to the handrail top ("mounted facing the surface" holds).
SR_H0 = 1.55           # height at the robot base (cabinet top)
SR_DROP = 0.92         # vertical distance from handrail top / cabinet top to the ramp surface


def _sr_top_z(y_e: float) -> float:
    return SR_H0 - y_e * S30 / C30


def _build_stair_rail(b: EnvBuilder, outdoor: bool = False) -> None:
    stone = b.material("sr_floor", (0.86, 0.84, 0.80), specular=0.25, shininess=0.5, reflectance=0.05,
                       texture={"rgb1": (0.88, 0.86, 0.82), "rgb2": (0.83, 0.81, 0.77), "repeat": (8, 8), "px": 256})
    step = b.material("sr_step", (0.80, 0.78, 0.74), specular=0.2, shininess=0.4)
    riser = b.material("sr_riser", (0.70, 0.68, 0.64))
    ramp = b.material("sr_ramp", (0.62, 0.62, 0.60), specular=0.1, shininess=0.3)
    track = b.material("sr_track", (0.40, 0.41, 0.43), specular=0.6, shininess=0.7)
    cab = b.material("sr_cabinet", (0.86, 0.87, 0.88), specular=0.3, shininess=0.4)
    cab_dark = b.material("sr_cabinet_dark", (0.30, 0.31, 0.33), specular=0.3, shininess=0.4)
    panel = b.material("sr_panel", (0.20, 0.42, 0.62))
    wood = b.material("sr_rail_wood", (0.55, 0.42, 0.30), specular=0.15, shininess=0.3)
    glass = b.material("sr_glass", (0.80, 0.88, 0.90), specular=0.6, shininess=0.8)
    post = b.material("sr_post", (0.55, 0.56, 0.58), specular=0.5, shininess=0.6)
    wall = b.material("sr_wall", (0.92, 0.91, 0.88))
    Rg = rot_x(-30.0)                                  # local y down along the slope, local z perpendicular to the ramp surface
    nrm = np.array([0.0, S30, C30])                    # ramp surface normal (scenery frame)
    b.plane((0, 0, 0.0), (9, 9), stone)
    # Stairs (x < −0.40)
    run, rise = 0.28, 0.16
    y_bot = (SR_H0 - SR_DROP) * C30 / S30
    k = 0
    while (k + 1) * rise <= 3.0:
        y0 = y_bot - k * run
        z = (k + 1) * rise
        b.box((-1.10, y0 - run / 2, z / 2), (0.70, run / 2, z / 2), riser)
        b.box((-1.10, y0 - run / 2, z - 0.008), (0.71, run / 2 + 0.012, 0.010), step)
        k += 1
    # Bicycle ramp (x −0.40 … 0.14): a sloped plate + a small rail in the middle (double rail). v12 ("stairs fine, rail is missing a section"):
    # length of ramp, rail, handrail and glass balustrade follows the actual start/end of the stairs (bottom starts at floor y_bot, top reaches the last step y_top plus 0.15 m), no longer a fixed 6 m
    y_top = y_bot - k * run
    L = (y_bot - y_top + 0.20) / C30                    # length along the slope (horizontal span = 0.05 beyond the bottom + stairs + 0.15 beyond the top)
    yc = ((y_bot + 0.05) + (y_top - 0.15)) / 2          # horizontal midpoint (scenery frame y)
    zr = _sr_top_z(yc) - SR_DROP                       # height of the ramp surface at yc
    b.box((-0.13, yc, zr - 0.03), (0.27, L / 2, 0.03), ramp, Rg)
    for xr in (-0.20, -0.06):
        b.box((xr, yc, zr + 0.012), (0.012, L / 2, 0.012), track, Rg)
    # Control cabinet cart: stands on the rail, tilted with the slope; cabinet top in the same plane as the handrail top (world z = 0 minus 1 cm for the base mounting plate)
    hc = SR_DROP - 0.06
    c0 = np.array([-0.13, 0.0, SR_H0]) - nrm * (0.01 + hc / 2)
    b.box(tuple(c0), (0.17, 0.22, hc / 2), cab, Rg)
    b.box(tuple(c0 + np.array([-0.171, 0, 0.12])), (0.002, 0.07, 0.05), panel, Rg)          # status panel on the cabinet door
    for dy in (-0.16, 0.16):                                                                    # rail wheels
        for xr in (-0.20, -0.06):
            cw = np.array([xr, dy * C30, SR_H0 - dy * S30]) - nrm * (hc + 0.02)
            b.box(tuple(cw), (0.02, 0.035, 0.025), cab_dark, Rg)
    b.box_w((b.base[0] - 0.13, b.base[1], -0.006), (0.17, 0.22, 0.006), cab_dark)           # cabinet-top mounting plate (moves with the unit)
    # Balustrade and wide handrail (handrail top = work-surface texture; x 0.17 … 0.43)
    for y in np.arange(y_top + 0.2, y_bot - 0.1, 0.52):                # posts: evenly spaced along the whole staircase
        zt = _sr_top_z(y) - 0.07
        b.cyl((0.30, y, zt - SR_DROP + 0.05), (0.30, y, zt), 0.018, post)
    b.box((0.30, yc, _sr_top_z(yc) - 0.07 - (SR_DROP - 0.12) / 2 / C30), (0.008, L / 2, (SR_DROP - 0.12) / 2), glass, Rg)
    b.box((0.30, yc, _sr_top_z(yc) - 0.048), (0.13, L / 2, 0.035), wood, Rg)
    if outdoor:                                        # outdoor pedestrian bridge (the scene section needs an outdoor one): sky, grass, trees, distant buildings, street lamps, no wall
        grass = b.material("sr_grass", (0.50, 0.62, 0.40), texture={"rgb1": (0.52, 0.64, 0.42), "rgb2": (0.47, 0.59, 0.37), "repeat": (20, 20), "px": 256})
        trunk = b.material("sr_trunk", (0.42, 0.33, 0.25))
        leaf = b.material("sr_leaf", (0.34, 0.52, 0.32))
        bld = b.material("sr_building", (0.86, 0.87, 0.88))
        bwin = b.material("sr_bwin", (0.62, 0.72, 0.80), specular=0.4, shininess=0.6)
        pole = b.material("sr_pole", (0.40, 0.42, 0.44), specular=0.4, shininess=0.5)
        b.plane((-4.0, 0, 0.002), (2.4, 9), grass)                 # grass outside the stairs
        b.plane((5.0, 0, 0.002), (3.0, 9), grass)
        for x, y, r in ((-5.6, -3.5, 0.9), (-6.2, 0.5, 1.1), (-5.4, 4.5, 0.8), (4.6, -3.5, 1.0), (5.2, 2.0, 1.2)):
            b.cyl((x, y, 0.0), (x, y, 1.6), 0.12, trunk)
            b.ell((x, y, 2.3), (r, r, r * 1.15), leaf)
        for x, y, w, h in ((-10.0, -6.0, 1.6, 6.5), (-10.5, -1.0, 2.0, 8.0), (-10.0, 4.5, 1.8, 5.5)):   # distant buildings
            b.box((x, y, h / 2), (0.8, w, h / 2), bld)
            for k in range(int(h / 1.1)):
                b.box((x + 0.81, y, 0.8 + k * 1.1), (0.01, w * 0.8, 0.28), bwin)
        for y in (-3.6, 3.2):                                         # street lamps
            b.cyl((1.3, y, 0.0), (1.3, y, 3.4), 0.05, pole)
            b.box((1.15, y, 3.42), (0.20, 0.08, 0.04), pole)
        try:                                                          # sky: bluish gradient (the default is an indoor light grey-white)
            sky = b.sp.texture("cp_sky")
            sky.rgb1, sky.rgb2 = [0.80, 0.88, 0.96], [0.60, 0.74, 0.90]
        except Exception:
            pass
        b.light((-2.2, 2.4, 5.0), (0.1, 0.4, 1.3), 0.26, True, cutoff=42)
        b.light((2.0, -2.0, 5.0), (0.0, 0.0, 1.2), 0.12, False, spot=False)
        return
    # Background: stairwell side wall and back wall
    b.box((1.6, 0, 3.0), (0.1, 9, 3.0), wall)
    b.box((0, -4.5, 3.0), (6, 0.1, 3.0), wall)
    b.light((-2.2, 2.4, 4.0), (0.1, 0.4, 1.3), 0.24, True, cutoff=42)
    b.light((2.0, -2.0, 4.0), (0.0, 0.0, 1.2), 0.10, False, spot=False)


STAIR_RAIL = EnvSpec(
    name="stair_handrail", label="Stair handrail",
    R_ew=np.column_stack([(1, 0, 0), (0, C30, -S30), (0, S30, C30)]).astype(float),
    t_e=(0.0, 0.0, SR_H0),
    table_half=(0.13, 1.6),
    camera={"lookat": (0.0, 0.25, 1.55), "azimuth": 122.0, "elevation": 20.0, "distance": 3.5, "fovy": 34.0},   # the arm does not leave the frame when raised
    build=_build_stair_rail,
)


# ------------------------------------------------------------------------------------------------ Office desk (Rizon 4 on top of a floor-cleaning robot cart)
# Same job (W1_wall_marker: marker strokes on a white surface): the work surface is horizontal anyway → scenery frame = world frame, no rotation; the robot base faces up, mounted on the cart top
OFFICE_H = 0.74        # desk height (scenery frame); cart top is at the same height as the desk


def _build_office(b: EnvBuilder) -> None:
    floor = b.material("of_floor", (0.80, 0.80, 0.78), specular=0.15, shininess=0.4, reflectance=0.03,
                       texture={"rgb1": (0.80, 0.80, 0.78), "rgb2": (0.76, 0.76, 0.74), "repeat": (10, 10), "px": 256})
    wall = b.material("of_wall", (0.93, 0.93, 0.91))
    window = b.material("of_window", (0.78, 0.86, 0.92), specular=0.4, shininess=0.6)
    frame = b.material("of_frame", (0.55, 0.57, 0.60))
    desk = b.material("of_desk", (0.95, 0.95, 0.94), specular=0.1, shininess=0.3)
    metal = b.material("of_metal", (0.45, 0.47, 0.50), specular=0.4, shininess=0.5)
    dark = b.material("of_dark", (0.18, 0.19, 0.21), specular=0.3, shininess=0.4)
    screen = b.material("of_screen", (0.10, 0.12, 0.15), specular=0.5, shininess=0.7)
    bot = b.material("of_bot", (0.93, 0.94, 0.95), specular=0.3, shininess=0.5)
    teal = b.material("of_teal", (0.06, 0.46, 0.43))
    pot = b.material("of_pot", (0.78, 0.74, 0.68))
    leaf = b.material("of_leaf", (0.36, 0.52, 0.34))
    chair = b.material("of_chair", (0.25, 0.27, 0.30))
    H = OFFICE_H
    b.plane((0, 0, 0.0), (8, 8), floor)
    b.plane((0, 2.2, 1.6), (6, 1.6), wall, np.column_stack([(1, 0, 0), (0, 0, 1), (0, -1, 0)]).astype(float))   # back wall (y = 2.2)
    b.box((-2.4, 0, 1.6), (0.06, 6, 1.6), wall)                                                                     # left wall
    for x in (-0.9, 1.1):
        b.box((x, 2.19, 1.6), (0.62, 0.012, 0.72), frame)
        b.box((x, 2.18, 1.6), (0.58, 0.012, 0.68), window)
    # White office desk (desk texture = extension of the work-surface material, see table_half; the desk board and legs are added below)
    c = b.to_e((0.30, -0.03, 0.0))
    b.box((c[0], c[1], H - 0.022), (0.38, 0.62, 0.02), desk)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.box((c[0] + sx * 0.34, c[1] + sy * 0.58, (H - 0.04) / 2), (0.02, 0.02, (H - 0.04) / 2), metal)
    b.box((c[0] + 0.30, c[1] + 0.42, H + 0.24), (0.015, 0.20, 0.13), screen)        # monitor (outside the workspace)
    b.cyl((c[0] + 0.32, c[1] + 0.42, H), (c[0] + 0.32, c[1] + 0.42, H + 0.11), 0.012, metal)
    b.box((c[0] + 0.32, c[1] + 0.42, H + 0.005), (0.06, 0.08, 0.005), metal)
    b.box((c[0] + 0.62, c[1] + 0.10, 0.46), (0.22, 0.22, 0.04), chair)               # chair
    b.box((c[0] + 0.80, c[1] + 0.10, 0.76), (0.03, 0.22, 0.28), chair)
    b.cyl((c[0] + 0.62, c[1] + 0.10, 0.0), (c[0] + 0.62, c[1] + 0.10, 0.42), 0.03, metal)
    b.cyl((1.8, 1.7, 0.0), (1.8, 1.7, 0.42), 0.18, pot)                              # potted plant
    b.ell((1.8, 1.7, 0.95), (0.30, 0.30, 0.42), leaf)
    # Floor-cleaning robot cart (arm base mounted on top): dark lower chassis + wheels + bumper, white upper body + teal status light strip, top plate
    bx, by = b.base[0], b.base[1]
    b.box((bx, by, 0.12), (0.26, 0.24, 0.10), dark)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.cyl((bx + sx * 0.18, by + sy * 0.245, 0.06), (bx + sx * 0.18, by + sy * 0.265, 0.06), 0.06, dark)
    b.box((bx + 0.265, by, 0.10), (0.012, 0.22, 0.05), metal)                        # front bumper
    hb = (H - 0.24 - 0.02) / 2
    b.box((bx, by, 0.24 + hb), (0.22, 0.21, hb), bot)
    b.box((bx + 0.222, by, 0.55), (0.003, 0.17, 0.012), teal)                         # status light strip
    b.box((bx, by, H - 0.012), (0.23, 0.22, 0.012), dark)                             # top plate (arm mounting surface)
    b.light((1.6, -2.4, 3.2), (0.2, 0.0, H), 0.22, True, cutoff=55)
    b.light((-1.5, 1.0, 2.8), (0.0, 0.0, 0.8), 0.10, False, spot=False)


OFFICE = EnvSpec(
    name="office_desk", label="Office desk",
    R_ew=np.eye(3),
    t_e=(0.0, 0.0, OFFICE_H),
    table_half=(0.36, 0.60),
    camera={"lookat": (-0.05, -0.05, 0.62), "azimuth": -70.0, "elevation": 18.0, "distance": 3.2, "fovy": 34.0},   # the whole cart is in frame
    build=_build_office,
)

STAIR_OUT = EnvSpec(
    name="footbridge_ramp", label="Footbridge handrail",
    R_ew=np.column_stack([(1, 0, 0), (0, C30, -S30), (0, S30, C30)]).astype(float),
    t_e=(0.0, 0.0, SR_H0),
    table_half=(0.13, 1.6),
    camera={"lookat": (0.0, 0.25, 1.55), "azimuth": 122.0, "elevation": 20.0, "distance": 3.5, "fovy": 34.0},
    build=lambda b: _build_stair_rail(b, outdoor=True),
)


# ------------------------------------------------------------------------------------------------ Factory line (Panda on a stainless steel workbench)
LINE_H = 0.92


def _build_line(b: EnvBuilder) -> None:
    epoxy = b.material("factory_floor", (0.62, 0.64, 0.64), specular=0.2, shininess=0.5, reflectance=0.05)
    stripe = b.material("floor_stripe", (0.93, 0.76, 0.18))
    steel = b.material("bench_steel", (0.70, 0.72, 0.74), specular=0.5, shininess=0.6)
    frame = b.material("conveyor_frame", (0.35, 0.40, 0.46), specular=0.3, shininess=0.4)
    belt = b.material("conveyor_belt", (0.16, 0.17, 0.18))
    roller = b.material("conveyor_roller", (0.62, 0.64, 0.66), specular=0.5, shininess=0.6)
    carton = b.material("carton", (0.72, 0.58, 0.40))
    guard = b.material("guard_yellow", (0.92, 0.74, 0.16))
    wall = b.material("factory_wall", (0.84, 0.85, 0.86))
    machine = b.material("machine_body", (0.88, 0.89, 0.90), specular=0.3, shininess=0.4)
    panel = b.material("machine_panel", (0.20, 0.42, 0.62))
    b.plane((0, 0, 0.0), (9, 9), epoxy)
    for y in (-0.95, 1.55):                                                        # yellow aisle lines
        b.box((0.0, y, 0.001), (5.0, 0.04, 0.002), stripe)
    # Workbench = double-door control cabinet (no hollow table frame): stainless steel top plate + closed cabinet body + two doors (seams, handles, vent louvers) + kick base
    c = b.to_e((0.30, -0.03, 0.0))
    cab = b.material("line_cabinet", (0.80, 0.82, 0.84), specular=0.3, shininess=0.4)
    door = b.material("line_door", (0.84, 0.86, 0.88), specular=0.35, shininess=0.45)
    seam = b.material("line_seam", (0.30, 0.32, 0.35))
    handle = b.material("line_handle", (0.25, 0.26, 0.28), specular=0.6, shininess=0.7)
    b.box((c[0], c[1], LINE_H - 0.025), (0.78, 0.38, 0.025), steel)                       # top plate (below the work-surface texture)
    hb = (LINE_H - 0.05 - 0.08) / 2                                                         # half height of the cabinet body (8 cm kick base underneath)
    b.box((c[0], c[1], 0.08 + hb), (0.74, 0.34, hb), cab)                                    # cabinet body
    b.box((c[0], c[1], 0.04), (0.72, 0.32, 0.04), frame)                                    # kick base
    yf = c[1] - 0.341                                                                        # front face with the doors (side facing the camera, −y)
    for sx in (-1, 1):                                                                       # two double-door groups
        xc = c[0] + sx * 0.37
        for dx in (-0.175, 0.175):
            b.box((xc + dx, yf - 0.004, 0.08 + hb), (0.168, 0.004, hb - 0.03), door)
        b.box((xc, yf - 0.009, 0.08 + hb), (0.003, 0.002, hb - 0.03), seam)                  # door seam
        for dx in (-0.03, 0.03):
            b.box((xc + dx, yf - 0.02, 0.08 + hb + 0.05), (0.008, 0.012, 0.09), handle)      # vertical handle
        for k in range(5):                                                                   # vent louvers
            b.box((xc - 0.175, yf - 0.009, 0.08 + 2 * hb - 0.12 - k * 0.03), (0.09, 0.002, 0.004), seam)
    b.box((c[0], yf - 0.009, 0.08 + hb), (0.004, 0.002, hb), seam)                            # post seam between the two groups
    # Conveyor belt: along x, right next to the +y side of the workbench
    yb = c[1] + 0.38 + 0.30
    b.box((0.0, yb, LINE_H - 0.10), (3.2, 0.26, 0.05), frame)
    b.box((0.0, yb, LINE_H - 0.045), (3.2, 0.22, 0.008), belt)
    for x in np.arange(-3.0, 3.2, 0.5):
        b.cyl((x, yb - 0.22, LINE_H - 0.06), (x, yb + 0.22, LINE_H - 0.06), 0.03, roller)
        b.box((x, yb, (LINE_H - 0.15) / 2), (0.03, 0.22, (LINE_H - 0.15) / 2), frame)
    for x in (-2.2, -1.1, 1.4, 2.6):
        b.box((x, yb, LINE_H + 0.09), (0.16, 0.13, 0.13), carton)
    b.box((0.0, yb + 0.27, LINE_H + 0.02), (3.2, 0.01, 0.05), guard)
    # Background: equipment, safety fence, factory walls
    b.box((-2.6, 2.2, 0.9), (0.7, 0.6, 0.9), machine)
    b.box((-2.6, 1.59, 1.1), (0.25, 0.01, 0.2), panel)
    b.box((2.8, 2.4, 1.2), (0.9, 0.5, 1.2), machine)
    for x in np.arange(-4.0, 4.1, 1.0):
        b.box((x, 3.2, 0.6), (0.025, 0.025, 0.6), guard)
    b.box((0, 3.2, 1.15), (4.0, 0.02, 0.025), guard)
    b.box((0, 5.0, 3.0), (9, 0.1, 3.0), wall)
    b.box((-5.0, 0, 3.0), (0.1, 9, 3.0), wall)
    b.light((2.2, -2.6, 3.8), (0.3, 0.0, LINE_H), 0.24, True, cutoff=40)
    b.light((-2.0, 2.0, 4.0), (0.0, 0.0, 1.0), 0.10, False, spot=False)


LINE = EnvSpec(
    name="factory_line", label="Factory line workbench",
    R_ew=np.eye(3),
    t_e=(0.0, 0.0, LINE_H),
    table_half=(0.75, 0.38),
    camera={"lookat": (0.15, 0.1, 1.05), "azimuth": -55.0, "elevation": 24.0, "distance": 2.4, "fovy": 34.0},
    build=_build_line,
)

# ------------------------------------------------------------------------------------------------ Cafe table (Spot + Arm, the same table as the principle section)
CAFE_H = 0.42          # table height (Spot stands on the floor, base z = −0.42)


def _build_cafe(b: EnvBuilder) -> None:
    floor = b.material("cafe_floor", (0.70, 0.60, 0.48), specular=0.15, shininess=0.4,
                       texture={"rgb1": (0.72, 0.62, 0.50), "rgb2": (0.68, 0.58, 0.46), "repeat": (14, 2), "px": 256})
    leg = b.material("cafe_leg", (0.30, 0.30, 0.31), specular=0.3, shininess=0.4)
    wall = b.material("cafe_wall", (0.93, 0.91, 0.87))
    window = b.material("cafe_window", (0.82, 0.88, 0.92), specular=0.4, shininess=0.6)
    frame = b.material("cafe_frame", (0.35, 0.33, 0.30))
    pot = b.material("plant_pot", (0.78, 0.74, 0.68))
    leaf = b.material("plant_leaf", (0.36, 0.52, 0.34))
    wood2 = b.material("cafe_table2", (0.62, 0.50, 0.38), specular=0.1, shininess=0.3)
    chair = b.material("cafe_chair", (0.42, 0.40, 0.38))
    b.plane((0, 0, 0.0), (9, 9), floor)
    c = b.to_e((0.30, -0.03, 0.0))
    b.box((c[0], c[1], CAFE_H - 0.02), (0.30, 0.35, 0.02), wood2)                  # table board (below the top texture)
    for sx in (-1, 1):
        for sy in (-1, 1):
            b.box((c[0] + sx * 0.26, c[1] + sy * 0.31, (CAFE_H - 0.04) / 2), (0.022, 0.022, (CAFE_H - 0.04) / 2), leg)
    b.box((-2.6, 0, 1.6), (0.08, 6, 1.6), wall)                                   # background wall (behind Spot) + two windows
    for y in (-0.6, 1.8):
        b.box((-2.51, y, 1.55), (0.01, 0.75, 0.75), frame)
        b.box((-2.50, y, 1.55), (0.01, 0.70, 0.70), window)
    b.box((0, 3.4, 1.6), (6, 0.08, 1.6), wall)
    for x, y in ((-2.0, 2.7), (-2.1, -2.2)):                                       # potted plants
        b.cyl((x, y, 0.0), (x, y, 0.35), 0.16, pot)
        for dx, dy, dz, r in ((0, 0, 0.75, 0.24), (0.12, 0.08, 0.98, 0.17), (-0.1, -0.06, 1.05, 0.15), (0.03, -0.1, 0.62, 0.18)):
            b.ell((x + dx, y + dy, dz), (r, r, r * 1.2), leaf)
    b.box((-1.0, 2.1, CAFE_H + 0.33), (0.35, 0.35, 0.02), wood2)                    # another table and chair in the distance
    b.cyl((-1.0, 2.1, 0.0), (-1.0, 2.1, CAFE_H + 0.31), 0.03, leg)
    b.box((-1.0, 2.7, 0.45), (0.2, 0.2, 0.02), chair)
    b.light((1.8, -1.6, 2.6), (0.3, -0.05, CAFE_H), 0.22, True, cutoff=40)
    b.light((-1.0, 1.2, 2.2), (0.0, 0.0, 0.5), 0.10, False, spot=False)


CAFE = EnvSpec(
    name="cafe_table", label="Cafe table",
    R_ew=np.eye(3),
    t_e=(0.0, 0.0, CAFE_H),
    table_half=(0.3, 0.35),
    camera={"lookat": (-0.1, 0.0, 0.6), "azimuth": -38.0, "elevation": 20.0, "distance": 2.1, "fovy": 34.0},
    build=_build_cafe,
    mug=True,
)

ENVS: dict[str, EnvSpec] = {e.name: e for e in (WALL, RAIL, STAIR_RAIL, LINE, CAFE, OFFICE, STAIR_OUT, VAN_CART)}


def env_camera_pose(env: EnvSpec, cam: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Scenery-frame camera parameters → world frame (position, 3×3 orientation: columns = camera x right, y up, z backward), fovy."""
    c = dict(env.camera, **(cam or {}))
    la = np.asarray(c["lookat"], float)
    az, el = math.radians(c["azimuth"]), math.radians(c["elevation"])
    pos_e = la + c["distance"] * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    fwd = la - pos_e
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.asarray(env.R_ew, float)
    t = np.asarray(env.t_e, float)
    pos_w = R.T @ (pos_e - t)
    M = np.column_stack([R.T @ right, R.T @ up, R.T @ (-fwd)])
    return pos_w, M, float(c["fovy"])
