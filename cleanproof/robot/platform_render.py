"""Display rendering of a platform scene: white-background look and camera positions, used for the camera view of the built-in simulation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .platforms import build_platform_scene, get_platform

# Display camera positions per platform (lookat in the world frame; distance / azimuth / elevation same as MjvCamera)
CAMS: dict[str, dict[str, Any]] = {
    "ur5e": {"lookat": (0.24, -0.03, 0.22), "distance": 1.45, "azimuth": 145, "elevation": -25},
    "panda": {"lookat": (0.12, -0.03, 0.20), "distance": 1.55, "azimuth": 145, "elevation": -24},
    "rizon4": {"lookat": (0.06, -0.03, 0.18), "distance": 1.50, "azimuth": 145, "elevation": -24},
    "spot_arm": {"lookat": (-0.05, -0.03, 0.05), "distance": 1.85, "azimuth": 145, "elevation": -20},
}


def dress_scene(spec_parent: Any, mujoco: Any) -> None:
    """White-background look: light gradient sky, soft ambient light, two directional lights (shadows), light floor material. Visual only, geometry and state untouched."""
    spec_parent.add_texture(name="cp_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                            rgb1=[0.96, 0.95, 0.93], rgb2=[0.86, 0.88, 0.91], width=512, height=512)
    v = spec_parent.visual
    import os
    soft = os.environ.get("CP_LIGHT", "soft") == "soft"
    # the long shadow of the robot on the table looks bad → key light close to overhead (shadow stays right under the robot), weakened; ambient light raised (shadow fainter)
    v.headlight.ambient = [0.46, 0.46, 0.46] if soft else [0.34, 0.34, 0.34]
    v.headlight.diffuse = [0.36, 0.36, 0.36] if soft else [0.26, 0.26, 0.26]
    v.headlight.specular = [0.0, 0.0, 0.0]
    w = spec_parent.worldbody
    if soft:
        for lt in list(spec_parent.lights):                  # the base scene (sim/scene.build_mjcf) has its own 1.2 m high ceiling lamp: close, so the shadow is big and black; removed in the video
            spec_parent.delete(lt)
        # no shadow at all is unrealistic. Key light placed high on the camera side (+x): the shadow falls behind the object (towards Spot), not on the stain; weak key light, strong ambient → faint shadow
        kx, ky, kz = (float(v) for v in os.environ.get("CP_KEY", "1.6,-0.4,2.4").split(","))
        kd = float(os.environ.get("CP_KEY_DIFF", "0.14"))
        d = np.array([0.3 - kx, -0.05 - ky, 0.0 - kz]); d /= np.linalg.norm(d)
        key = w.add_light(name="cp_key", pos=[kx, ky, kz], dir=d.tolist(), diffuse=[kd, kd, kd], specular=[0.02, 0.02, 0.02],
                          castshadow=os.environ.get("CP_SHADOW", "1") == "1")
        key.type = mujoco.mjtLightType.mjLIGHT_SPOT                       # spotlight: the shadow map covers only the tabletop and the robot (high resolution, no stair-step aliasing)
        key.cutoff = float(os.environ.get("CP_KEY_CUTOFF", "40"))
        key.exponent = 1.0
        key.bulbradius = float(os.environ.get("CP_BULB", "0.25"))            # soft shadow: blurred edges, not a hard cut
        spec_parent.visual.quality.shadowsize = 8192
        spec_parent.visual.quality.numslices = 128                          # no facets on cylinder / wiping head sides
        spec_parent.visual.quality.offsamples = 8                           # offscreen multisampling (edge anti-aliasing)
    else:
        w.add_light(name="cp_key", pos=[1.2, -1.0, 2.2], dir=[-0.45, 0.35, -0.82], diffuse=[0.42, 0.42, 0.41], specular=[0.03, 0.03, 0.03],
                    castshadow=True)
    w.add_light(name="cp_fill", pos=[-1.0, 1.2, 1.8], dir=[0.4, -0.5, -0.75], diffuse=[0.14, 0.15, 0.16], specular=[0, 0, 0], castshadow=False)
    for name in ("dirt", "table"):                       # tabletop and workspace: remove specular reflection (it would wash out wood grain and stains in close-ups)
        try:
            m = spec_parent.material(name)
            m.specular, m.shininess, m.reflectance = 0.0, 0.0, 0.0
        except Exception:
            pass


# Video color scheme (yellow looks bad, change to a light blue close to the UR3e + steel color, wiping head also light blue). Look only, replaced by material / color group
LIGHT_BLUE = (0.56, 0.74, 0.89)
STEEL = (0.66, 0.69, 0.72)
STEEL_DARK = (0.46, 0.49, 0.53)


RESTYLE = False      # changed again: leave colors alone for now, keep the original black and yellow (function kept for later)


def restyle_robot(model: Any, mujoco: Any, platform: str) -> None:
    if not RESTYLE or platform != "spot_arm":
        return
    for mid in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or ""
        if name == "r_wrap":                                   # Spot yellow shell
            model.mat_rgba[mid, :3] = LIGHT_BLUE
            model.mat_specular[mid], model.mat_shininess[mid] = 0.25, 0.4
        elif name == "r_BlackAbs":                             # black plastic → steel color
            model.mat_rgba[mid, :3] = STEEL
            model.mat_specular[mid], model.mat_shininess[mid] = 0.45, 0.6
    for g in range(model.ngeom):
        b = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if b.startswith("r_") and model.geom_matid[g] >= 0 and max(model.geom_rgba[g, :3]) < 0.3:
            model.geom_rgba[g, :3] = STEEL_DARK                # parts with an explicit dark rgba (e.g. the arm shoulder mount) override the material color
            continue
        if b != "r_cp_tool" or model.geom_matid[g] >= 0:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        model.geom_rgba[g, :3] = LIGHT_BLUE if gname == "r_pad" else STEEL_DARK




class LiveView:
    """Display picture of a platform scene (white-background look, CAMS camera positions): render one frame for a set of joint angles and a stain map.
    Shared by PlatformRenderer (per-frame re-rendering of recordings, the camera pictures of the video software section) and the camera picture of the software live mode (cleanproof/ui/session.py),
    so what the camera shows in a live job is the viewpoint in the video. The GL context of the renderer is bound to the thread that created it: render in the thread where it was built."""

    def __init__(self, platform: str, dirt_shape: tuple[int, int] = (120, 150), size: tuple[int, int] = (1920, 1080)):
        import mujoco

        from ..sim.scene import DIRT_TEXTURE
        self.mj = mujoco
        self.spec = get_platform(platform)
        self.scene = build_platform_scene(self.spec, dirt_shape=dirt_shape, dress=dress_scene)
        m = self.scene.model
        restyle_robot(m, mujoco, platform)
        self.renderer = mujoco.Renderer(m, size[1], size[0])
        self.tex_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, DIRT_TEXTURE)
        c = CAMS.get(platform, CAMS["ur5e"])
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = c["lookat"]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = c["distance"], c["azimuth"], c["elevation"]

    def render(self, q, dirt_rgb: Optional[np.ndarray] = None) -> np.ndarray:
        """dirt_rgb: pass the stain map (uint8, (nx, ny, 3)) when it changed, None if unchanged (the texture is not re-uploaded)."""
        from ..sim.scene import dirt_to_texture
        mj, s = self.mj, self.scene
        s.set_q(q)
        mj.mj_forward(s.model, s.data)
        if dirt_rgb is not None and self.tex_id >= 0:
            tex = dirt_to_texture(dirt_rgb)
            adr = s.model.tex_adr[self.tex_id]
            s.model.tex_data[adr: adr + tex.size] = tex.ravel()
            if getattr(self.renderer, "_gl_context", None) is not None:
                self.renderer._gl_context.make_current()
            mj.mjr_uploadTexture(s.model, self.renderer._mjr_context, self.tex_id)
        self.renderer.update_scene(s.data, camera=self.cam)
        return self.renderer.render().copy()

    @staticmethod
    def dirt_rgb(dirt: Any) -> np.ndarray:
        """Live stain map → exactly the same blend as the recording replay (RecordedStates.rgb): plain surface base color + each stain type overlaid by concentration, (nx, ny, 3) uint8."""
        lv = np.asarray(dirt.level, dtype=np.float32)
        img = np.broadcast_to(np.array(dirt.surface_rgb, np.float32), lv.shape[1:] + (3,)).copy()
        for t, k in enumerate(dirt.types):
            a = np.clip(lv[t], 0, 1)[..., None]
            img = img * (1 - a) + np.array(dirt.params[k]["rgb"], np.float32) * a
        return np.clip(img, 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.renderer.close()
