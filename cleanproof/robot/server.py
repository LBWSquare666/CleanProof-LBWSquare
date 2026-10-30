"""skill server: the only RTDE writer on the robot side; exposes only whitelisted skills (PLAN §4.4).

    survey / inspect_closeup / wipe_zone / spot_scrub     (HTTP POST /skill, pydantic validation; invalid requests get 422 / rejected)
    /stop    software stop (rejects further motion; the physical E-stop and the teach pendant remain the first line of defense)
    /reset   manual reset of faults and the software stop (a controller protective stop can only be cleared on the teach pendant)
    /health  start-up checks, robot state, motion authorization, busy/idle; /config the zone and robot config currently in effect (the client builds its grid from it)

Backends:
    mujoco / rtde_fake   simulation: dev-machine self-test, deployment self-test; also serves the simulated view at /frame/<camera>, including /frame/view: the same display
                         viewpoint as the built-in simulation in the software, so client code matches the real robot; --pace throttles the simulation to a few times real time
    ursim                URSim (whitelisted addresses only)
    ur3e                 real robot: can only run on robot-pc (make_rtde_robot checks the hostname, the gate and the motion authorization);
                         in dry_run / full mode **every skill request that moves** re-checks the motion authorization (an arm lasts 300 s; after it expires it must be re-armed on site);
                    while the site is not calibrated only shadow mode may start (so the on-site calibration in the software can read /state), and every skill request is refused with 423

By default it listens on 127.0.0.1 only: other machines (the agent on the GPU workstation) reach it through an SSH tunnel, and other people on the campus network cannot send commands directly.
The real-controller backends (ursim / ur3e) also require a shared token stored in <user config dir>/robot_token, default ~/.config/cleanproof, overridable with $CLEANPROOF_CONFIG_DIR;
permissions 0600, request header X-CleanProof-Token:
the GPU workstation is shared by several people and the tunnel port is visible to every local user, so the token limits who can send commands to the account that holds the file.
Only one skill runs at a time (409 when busy); an HTTP disconnect does not interrupt a running skill: the skill finishes and lifts on this machine, and the result goes to the local log.

  # on robot-pc (real robot, shadow first):
  CLEANPROOF_SITE=ur3e_site python -m cleanproof.robot.server --backend ur3e --mode shadow
  # dev-machine self-test (simulation):
  python -m cleanproof.robot.server --backend mujoco --scenario S1_light --source topdown --port 8100
  # deployment self-test (the robot computer runs the home dining table site in simulation, the software connects through the tunnel):
  MUJOCO_GL=egl python -m cleanproof.robot.server --backend mujoco --scenario H5_dining_permanent --pace 3 --port 8100
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..common.config import load_yaml, runs_root, site_dir, user_config_dir
from ..common.schemas import Pattern
from ..common.zones import ZoneGrid
from .commissioning import MODES, ArmingToken, commissioning_cfg, git_rev
from .skills import SkillExecutor, SkillOutcome

log = logging.getLogger("cleanproof.robot.server")

TOKEN_HEADER = "X-CleanProof-Token"
DEFAULT_TOKEN_PATH = "~/.config/cleanproof/robot_token"     # location when $CLEANPROOF_CONFIG_DIR is not set (see token_path() for the actual path)


def token_path() -> Path:
    """Shared token file: <user config dir>/robot_token ($CLEANPROOF_CONFIG_DIR is read at call time)."""
    return user_config_dir() / "robot_token"


def read_token(path: Optional[str] = None) -> Optional[str]:
    p = Path(path).expanduser() if path else token_path()
    if not p.is_file():
        return None
    if p.stat().st_mode & 0o077:
        raise PermissionError(f"{p} permissions too open (must be 0600)")
    t = p.read_text(encoding="utf-8").strip()
    return t or None


def auth_headers() -> dict[str, str]:
    """Request header the client (agent, calibration, smoke scripts) sends to the skill server / camera service; empty when there is no token file (simulation self-test)."""
    t = read_token()
    return {TOKEN_HEADER: t} if t else {}

BACKENDS = ("mujoco", "rtde_fake", "ursim", "ur3e")
SITE_CALIB_KEYS = {"table", "work_area", "survey_q"}           # keys zones.yaml must have after the real-robot site is calibrated
NOT_CALIBRATED = "site not calibrated: calibrate it from the CleanProof setup wizard (Site step) or scripts/calibrate_site.py"
MOTION_SKILLS = ("survey", "inspect_closeup", "wipe_zone", "spot_scrub")


class SkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill: Literal["survey", "inspect_closeup", "wipe_zone", "spot_scrub"]
    zone_ids: list[str] = Field(default_factory=list, max_length=6)
    force_n: Optional[float] = Field(default=None, ge=0.0, le=50.0)   # the range is checked by SkillExecutor.force_problem
    pattern: Optional[Pattern] = None
    passes: int = Field(default=1, ge=1, le=2)
    bbox_norm: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    forbidden_zones: list[str] = Field(default_factory=list)   # can only **add** forbidden zones on top of the server config
    request_id: str = ""

    @field_validator("bbox_norm")
    @classmethod
    def _bbox(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is not None and not (all(0.0 <= x <= 1.0 for x in v) and v[0] < v[2] and v[1] < v[3]):
            raise ValueError("bbox_norm must be normalized coordinates inside the zone [u0, v0, u1, v1], 0 ≤ u0 < u1 ≤ 1, 0 ≤ v0 < v1 ≤ 1")
        return v


class StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reason: str = "operator stop"


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    by: str = Field(min_length=1)


class SkillService:
    """Service core independent of HTTP (tests call it directly): owns the single executor and runs skills serially."""

    def __init__(self, executor: SkillExecutor, backend: str, mode: str, rcfg: dict[str, Any],
                 log_dir: Optional[Path] = None, arming: Optional[ArmingToken] = None,
                 site_forbidden: tuple[str, ...] = (), sim: Any = None, source: Any = None, token: Optional[str] = None,
                 calibration_problem: Optional[str] = None):
        self.ex, self.backend, self.mode, self.rcfg = executor, backend, mode, rcfg
        # site not calibrated (only allowed to start like this in shadow mode): the zones are a placeholder default grid and are never used for motion; /skill always returns 423
        self.calibration_problem = calibration_problem
        self.robot = executor.robot
        self.site_forbidden = set(site_forbidden)
        self.ex.forbidden = set(self.site_forbidden)
        self.arming = arming
        self.sim, self.source = sim, source            # simulation backends only: scene and simulated cameras
        self.view: Any = None                          # simulation backends only: display view (built in create_app, see SimView)
        self.token = token
        # robot platform (key of configs/platforms.yaml): simulation backends use the scene's platform, the RTDE backend is the UR3e; /health reports it to the software (report, camera view)
        self.platform = str(((getattr(sim, "meta", None) or {}).get("platform")) or ("ur3e" if backend in ("ursim", "ur3e", "rtde_fake") else ""))
        bad = [z for z in self.site_forbidden if z not in executor.grid]
        if bad:
            raise ValueError(f"site forbidden zone {bad} is not in the current zones (was site_forbidden_zones not updated after the row/column count changed?)")
        self.lock = threading.Lock()
        self.busy: Optional[str] = None
        self.stopped: Optional[str] = None
        self.n_requests = 0
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.time()

    # ---------- logging ----------
    def _log(self, kind: str, **data: Any) -> None:
        if not self.log_dir:
            return
        rec = {"ts": time.time(), "kind": kind, **data}
        with open(self.log_dir / "skill_server.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ---------- state ----------
    def arming_problem(self) -> Optional[str]:
        """Real robot dry_run / full: the motion authorization must still be valid (when the arm granted by on-site staff expires, new motion is refused)."""
        if self.backend != "ur3e" or self.mode == "shadow":
            return None
        st = (self.arming or ArmingToken()).status()
        return None if st.armed else st.reason

    def health(self) -> dict[str, Any]:
        r = self.robot
        arm = None
        if self.backend == "ur3e":
            st = (self.arming or ArmingToken()).status()
            arm = {"armed": st.armed, "by": st.by, "remaining_s": round(st.remaining_s, 1), "reason": st.reason}
        try:
            problems = self.ex.startup_problems()
        except Exception as e:  # noqa: BLE001
            problems = [f"startup check failed: {type(e).__name__}: {e}"]
        if self.calibration_problem:
            problems = [self.calibration_problem, *problems]
        return {
            "backend": self.backend, "mode": self.mode, "label": getattr(r, "label", self.backend),
            "git": git_rev(), "site": site_dir().name if site_dir() else "",     # only the site name, no absolute path
            "now": float(r.now()), "robot_mode": int(r.get_robot_mode()), "safety_mode": int(r.get_safety_mode()),
            "speed_fraction": float(r.get_speed_fraction()), "startup_problems": problems,
            "busy": self.busy, "fault": self.ex.fault, "stopped": self.stopped, "arming": arm,
            "contact_allowed": bool(getattr(r, "contact_allowed", True)), "requests": self.n_requests,
            "site_forbidden": sorted(self.site_forbidden), "platform": self.platform,
            "sim_source": getattr(self.source, "mode", None),   # simulation backends only: camera mode (topdown / render)
            "calibrated": self.calibration_problem is None,
            **({"host": socket.gethostname()} if self.token else {}),   # the host name is returned only when a token protects the server (a simulation backend without a token does not expose it)
        }

    def state(self) -> dict[str, Any]:
        r = self.robot
        return {"now": float(r.now()), "q": r.get_q().tolist(), "tcp_pose": r.get_tcp_pose().tolist(),
                "tcp_force": r.get_tcp_force().tolist(), "safety_mode": int(r.get_safety_mode()), "busy": self.busy}

    def config(self) -> dict[str, Any]:
        g = self.ex.grid
        zones = {"table": {"z_surface": float(g.z_surface)},
                 "work_area": {"origin_xy": g.origin.tolist(), "size_xy": g.size.tolist(), "rows": g.rows, "cols": g.cols},
                 "tool_rotvec": g.tool_rotvec.tolist()}
        if g.survey_q is not None:
            zones["survey_q"] = g.survey_q.tolist()
        return {"zones": zones, "robot": self.rcfg, "backend": self.backend, "mode": self.mode,
                "calibrated": self.calibration_problem is None}       # False: zones are a placeholder default grid (site not calibrated)

    # ---------- skills ----------
    def run(self, req: SkillRequest) -> tuple[int, dict[str, Any]]:
        """Returns (HTTP status code, result). 409: busy; 423: stopped or not authorized; 200: skill executed (status may be rejected / fault)."""
        self.n_requests += 1
        if self.stopped:
            return 423, {"error": f"software stop active: {self.stopped}; reset after on-site confirmation (scripts/robot_ctl.py reset)"}
        if self.calibration_problem:                 # not calibrated: refuse before any dispatch (placeholder geometry must not be used for motion)
            self._log("refused", request=req.model_dump(mode="json"), reason=self.calibration_problem)
            return 423, {"error": self.calibration_problem}
        if self.ex.fault:
            return 423, {"error": f"unreset fault exists: {self.ex.fault}; reset after on-site confirmation"}
        bad = [z for z in req.forbidden_zones if z not in self.ex.grid]
        if bad:
            return 422, {"error": f"unknown forbidden zone id {bad}"}
        if (p := self.arming_problem()):
            self._log("refused", request=req.model_dump(mode="json"), reason=p)
            return 423, {"error": p}
        if not self.lock.acquire(blocking=False):
            return 409, {"error": f"busy executing {self.busy}"}
        try:
            self.busy = f"{req.skill}{req.zone_ids or ''}"
            self.ex.forbidden = self.site_forbidden | set(req.forbidden_zones)
            self._log("request", request=req.model_dump(mode="json"))
            n_shadow = len(getattr(self.robot, "shadow_log", []) or [])
            out = self._dispatch(req)
            d = out.to_dict(with_trace=True)
            self._log("outcome", request_id=req.request_id, **{k: v for k, v in d.items() if k != "trace"})
            sl = getattr(self.robot, "shadow_log", None)
            if sl:                                   # shadow: write the commands that "would have been sent" to the log (gate evidence) and clear the in-memory buffer
                self._log("shadow_commands", request_id=req.request_id, commands=sl[n_shadow:])
                del sl[:]
            return 200, d
        except Exception as e:  # noqa: BLE001
            log.exception("skill execution failed")
            try:
                self.robot.speed_stop()
            finally:
                self._log("exception", request_id=req.request_id, error=f"{type(e).__name__}: {e}")
            return 500, {"error": f"{type(e).__name__}: {e}"[:500]}
        finally:
            self.busy = None
            self.lock.release()

    def _dispatch(self, req: SkillRequest) -> SkillOutcome:
        grid = self.ex.grid
        bad = [z for z in req.zone_ids if z not in grid]
        if bad:
            return SkillOutcome(req.skill, "rejected", f"unknown zone {bad}", zone_ids=req.zone_ids)
        if req.skill == "survey":
            return self.ex.survey()
        if req.skill == "inspect_closeup":
            if len(req.zone_ids) != 1:
                return SkillOutcome(req.skill, "rejected", "inspect_closeup accepts only one zone", zone_ids=req.zone_ids)
            if req.zone_ids[0] in self.site_forbidden:   # something sits in a site forbidden zone: a close-up would lower the tool to 8 cm above the table
                return SkillOutcome(req.skill, "rejected", f"{req.zone_ids[0]} is a site forbidden zone, close-up not allowed", zone_ids=req.zone_ids)
            return self.ex.closeup(req.zone_ids[0])
        if not req.zone_ids:
            return SkillOutcome(req.skill, "rejected", "zone missing", zone_ids=[])
        if req.force_n is None:
            return SkillOutcome(req.skill, "rejected", "target force force_n missing", zone_ids=req.zone_ids)
        if req.skill == "spot_scrub":
            if req.bbox_norm is None or len(req.zone_ids) != 1:
                return SkillOutcome(req.skill, "rejected", "spot_scrub needs one zone and bbox_norm", zone_ids=req.zone_ids)
            return self.ex.wipe(req.zone_ids, req.force_n, "zigzag", req.passes, req.bbox_norm)
        return self.ex.wipe(req.zone_ids, req.force_n, (req.pattern or Pattern.zigzag).value, req.passes, None)

    # ---------- stop and reset ----------
    def stop(self, reason: str) -> dict[str, Any]:
        self.stopped = reason or "operator stop"
        es = getattr(self.robot, "emergency_stop", None)
        if es is not None:
            es(self.stopped)
        self._log("stop", reason=self.stopped, busy=self.busy)
        return {"stopped": self.stopped, "busy": self.busy}

    def reset(self, by: str) -> tuple[int, dict[str, Any]]:
        if not self.lock.acquire(blocking=False):
            return 409, {"error": f"busy executing {self.busy}, reset after it exits"}
        try:
            r = self.robot
            if hasattr(r, "clear_software_stop"):
                r.clear_software_stop()
            elif self.backend == "mujoco" and hasattr(r, "clear_protective_stop"):
                r.clear_protective_stop()            # simulation only
            sm = int(r.get_safety_mode())
            if sm != 1:                              # controller is still in protective stop: keep the service-level stop flag, skills keep being refused
                return 409, {"error": f"safety_mode={sm}: the controller protective stop / E-stop must be cleared on the teach pendant first"}
            self.stopped = None
            self.ex.reset_fault()
            self._log("reset", by=by)
            return 200, {"ok": True, "by": by}
        finally:
            self.lock.release()

    def close(self) -> None:
        if self.view is not None:
            try:
                self.view.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.robot.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------- construction
def pace_robot(robot: Any, pace: float) -> None:
    """The simulated robot runs at about pace times real time (after wait_tick, sleep if ahead of the wall clock); idle gaps between skills are not made up (when behind, timing restarts from now)."""
    orig = robot.wait_tick
    base = [time.time(), robot.now()]

    def wait_tick() -> None:
        orig()
        ahead = (robot.now() - base[1]) / pace - (time.time() - base[0])
        if ahead > 0.002:
            time.sleep(min(ahead, 0.05))
        elif ahead < -0.1:
            base[0], base[1] = time.time(), robot.now()

    robot.wait_tick = wait_tick


class SimView:
    """Display view of the simulation backend (GET /frame/view): the same viewpoint as the camera view of the built-in simulation in the software (platform_render.LiveView).
    The renderer's GL context is bound to a thread, so all rendering runs in one dedicated thread; if two requests come close together the previous frame is returned."""

    EVERY_S = 0.15

    def __init__(self, sc: Any):
        from concurrent.futures import ThreadPoolExecutor
        self.sc = sc
        self.platform = (sc.meta or {}).get("platform")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simview")
        self.lv: Any = None
        self.key: Any = None
        self.jpg: Optional[bytes] = None
        self.t = 0.0
        self.failed = False

    def available(self) -> bool:
        from .platforms import get_platform
        try:
            return bool(self.platform) and not get_platform(self.platform).builtin and not self.failed
        except Exception:  # noqa: BLE001
            return False

    def get(self) -> Optional[bytes]:
        from concurrent.futures import TimeoutError as FutTimeout
        try:
            return self.pool.submit(self._render).result(timeout=20.0)
        except FutTimeout:                           # render thread is busy (the first frame is slow while the scene is built): return the previous frame, or 404 if there is none
            return self.jpg
        except RuntimeError:                         # already closed (service is exiting)
            return None

    def _render(self) -> Optional[bytes]:
        if self.jpg is not None and time.time() - self.t < self.EVERY_S:
            return self.jpg
        robot = self.sc.robot
        try:
            if self.lv is None:
                from .platform_render import LiveView
                self.lv = LiveView(self.platform, (int(robot.dirt.nx), int(robot.dirt.ny)), size=(1280, 720))
            lv = robot.dirt.level
            key = (float(lv.sum()), float(lv.max()))
            rgb = self.lv.dirt_rgb(robot.dirt) if key != self.key else None
            self.key = key
            from ..perception.sources import encode_jpeg
            self.jpg, self.t = encode_jpeg(self.lv.render(robot.get_q(), rgb), 85), time.time()
        except Exception as e:  # noqa: BLE001  a missing display view does not affect skills: the client falls back to the fixed camera
            log.warning("sim view unavailable: %s", e)
            self.failed = True
            return None
        return self.jpg

    def close(self) -> None:
        def _close() -> None:
            if self.lv is not None:
                try:
                    self.lv.close()
                except Exception:  # noqa: BLE001
                    pass
        try:
            self.pool.submit(_close).result(timeout=10.0)
        finally:
            self.pool.shutdown(wait=False)


def build_service(backend: str, mode: str = "full", scenario: str = "S1_light", source: str = "topdown",
                  host: Optional[str] = None, log_dir: Optional[Path] = None, pace: float = 0.0) -> SkillService:
    if backend not in BACKENDS:
        raise ValueError(f"backend ∈ {BACKENDS}")
    if mode not in MODES:
        raise ValueError(f"mode ∈ {MODES}")
    if backend == "mujoco":
        mode = "full"                            # the mujoco backend has no graded modes (skills drive the simulation directly); report it truthfully
    if backend in ("mujoco", "rtde_fake"):
        from ..perception.sources import SimCameraSource
        from ..sim.scenarios import build

        sc = build(scenario, render=(source == "render"), robot_backend=backend, mode=mode)
        src = SimCameraSource(sc.robot, sc.grid, mode=source)
        sc.executor.survey()
        sc.reference = src.rectified()          # clean reference image before stains are placed (GET /sim/reference)
        sc.add_stains()
        if pace > 0:
            pace_robot(sc.robot, pace)
        return SkillService(sc.executor, backend, mode, sc.executor.cfg, log_dir=log_dir,
                            site_forbidden=tuple(sc.task.forbidden_zones), sim=sc, source=src,
                            token=read_token())        # with a token file the simulation backend also authenticates by token (same connection method as the real deployment)
    from .backends.rtde_backend import make_rtde_robot

    calibration_problem: Optional[str] = None
    if backend == "ur3e":                            # the real robot must have site calibration (otherwise it would use the z_surface=0 placeholder geometry)
        sd = site_dir()
        zf = sd / "zones.yaml" if sd else None
        site_z = (load_yaml(zf) if zf and zf.is_file() else {})
        if not sd or not SITE_CALIB_KEYS <= set(site_z):
            if mode != "shadow":
                raise RuntimeError(f"ur3e needs site calibration: zones.yaml under CLEANPROOF_SITE={str(sd or '')!r} lacks table / work_area / survey_q"
                                   "(scripts/calibrate_site.py; while not calibrated it can only start with --mode shadow, so the on-site calibration in the software can read the robot pose)")
            calibration_problem = NOT_CALIBRATED   # shadow: start as usual (/state and /health are available for calibration), all skills are refused
    token = read_token()
    if token is None:
        raise RuntimeError(f"{backend} needs a shared token {token_path()} (permissions 0600; scripts/robot_pc/sync_to_robot_pc.sh generates and syncs it)")
    rcfg = load_yaml("robot.yaml")
    grid = ZoneGrid()
    cc = commissioning_cfg(rcfg)
    arming = ArmingToken(cc["arming_token_path"], float(rcfg.get("arming", {}).get("ttl_s", 300)))
    robot = make_rtde_robot(backend, mode, rcfg=rcfg, host=host, z_surface=grid.z_surface, arming=arming)
    site_forbidden = tuple(load_yaml("zones.yaml").get("site_forbidden_zones", []) or [])
    if calibration_problem:                          # there are no site forbidden zones on a placeholder grid (skills are all refused anyway)
        site_forbidden = ()
    ex = SkillExecutor(robot, grid=grid, rcfg=rcfg, forbidden_zones=site_forbidden)
    return SkillService(ex, backend, mode, rcfg, log_dir=log_dir, arming=arming, site_forbidden=site_forbidden, token=token,
                        calibration_problem=calibration_problem)


def create_app(svc: SkillService):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response

    app = FastAPI(title="CleanProof skill server")

    if svc.token:
        import hmac

        @app.middleware("http")
        async def require_token(request, call_next):
            if not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), svc.token):
                return JSONResponse({"error": "missing or wrong token"}, status_code=401)
            return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return svc.health()

    @app.get("/now")
    def now() -> dict[str, Any]:
        return {"now": float(svc.robot.now())}

    @app.get("/state")
    def state() -> dict[str, Any]:
        return svc.state()

    @app.get("/config")
    def config() -> dict[str, Any]:
        return svc.config()

    @app.post("/skill")
    def skill(req: SkillRequest):
        code, body = svc.run(req)
        return JSONResponse(body, status_code=code)

    @app.post("/stop")
    def stop(req: StopRequest) -> dict[str, Any]:
        return svc.stop(req.reason)

    @app.post("/reset")
    def reset(req: ResetRequest):
        code, body = svc.reset(req.by)
        return JSONResponse(body, status_code=code)

    # ---- simulation backends only: simulated camera and reference image (the real cameras are served by cleanproof.perception.camera_server) ----
    if svc.source is not None:
        from ..perception.sources import encode_jpeg

        if svc.view is None:                         # one service builds only one display view (reused when create_app is called several times on a service, so no render thread leaks)
            svc.view = SimView(svc.sim)
        view = svc.view

        @app.get("/frame/{camera}")
        def frame(camera: str):
            if camera == "view":                     # display view (the software camera image); if this platform has none, 404 and the client falls back to the fixed camera
                jpg = view.get() if view.available() else None
                if jpg is None:
                    return JSONResponse({"error": "no display view for this platform"}, status_code=404)
                return Response(jpg, media_type="image/jpeg")
            return Response(encode_jpeg(svc.source.frame(camera), quality=95), media_type="image/jpeg")

        @app.get("/sim/reference")
        def reference():
            return Response(encode_jpeg(svc.sim.reference, quality=98), media_type="image/jpeg")

        @app.get("/sim/truth")
        def truth() -> dict[str, Any]:
            return {k: bool(v) for k, v in svc.sim.truth().items()}

    return app


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--mode", default="shadow", choices=MODES, help="graded mode of the rtde backend; the real robot starts from shadow")
    ap.add_argument("--robot-host", default=None, help="ursim address (whitelisted prefix); ur3e always uses configs/robot.yaml hosts.ur3e")
    ap.add_argument("--scenario", default="S1_light", help="scenario of the simulation backend")
    ap.add_argument("--source", default="topdown", choices=("topdown", "render"), help="camera mode of the simulation backend")
    ap.add_argument("--bind", default="127.0.0.1", help="listen address; local machine only by default (remote access through an SSH tunnel)")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--log-dir", default=None, help="default runs/skill_server/<date>")
    ap.add_argument("--pace", type=float, default=0.0, help="simulation backend: run at about this many times real time (0 = as fast as possible; use 3 for deployment self-test, same as the desktop version)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.bind not in ("127.0.0.1", "localhost") and args.backend == "ur3e":
        print("refused: the ur3e backend only listens on local addresses (use an SSH tunnel for remote access)")
        return 2
    log_dir = Path(args.log_dir) if args.log_dir else runs_root() / "skill_server" / time.strftime("%Y%m%d")
    svc = build_service(args.backend, args.mode, args.scenario, args.source, host=args.robot_host, log_dir=log_dir, pace=args.pace)
    h = svc.health()
    print(json.dumps({k: h[k] for k in ("backend", "mode", "label", "startup_problems", "arming")}, ensure_ascii=False))
    import uvicorn

    try:
        uvicorn.run(create_app(svc), host=args.bind, port=args.port, log_level="info", workers=1)
    finally:
        svc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
