"""Run a job against the robot computer (GPU workstation side): executor = skill server client, camera = camera service, same TaskRunner as simulation.

The terminal trial (scripts/run_real.py) and the software's live job (cleanproof/ui/session.RemoteSession) share the connection and checks here:
  - when the skill server reports a simulation backend (mujoco / rtde_fake), the reference image and camera both come from the skill server itself (/sim/reference, /frame/*);
  - with a real-robot backend (ursim / ur3e), use this site's field calibration (calib.json, reference.png); before starting a job, check that the zones match the skill server and
    the camera markers have not drifted.
This process creates no RTDE interface: whether the robot can move and how fast is decided by the skill server's staged mode and the on-site arming.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from ..common.config import REPO_ROOT, site_dir

SIM_BACKENDS = ("mujoco", "rtde_fake")


class RemoteSetupError(RuntimeError):
    """Cannot connect or cannot start a job (robot computer unreachable, token rejected, already E-stopped, zones mismatch, camera was bumped ...): the message is in English and shown directly to the operator."""


def describe_link_error(e: BaseException, url: str) -> str:
    """Exception while connecting to the robot computer → one English sentence for the operator (the software UI shows only the first line)."""
    import httpx

    from ..robot.server import token_path

    if isinstance(e, RemoteSetupError):
        return str(e)
    if isinstance(e, PermissionError):
        return f"The robot access token file must be private: run chmod 600 {token_path()}"
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 401:
            return "The robot computer rejected this computer's access token. Pair the computers again (install_edge.sh --pair)."
        return f"The robot computer at {url} answered HTTP {code}."
    if isinstance(e, httpx.TimeoutException) and not isinstance(e, httpx.ConnectTimeout):
        return f"The robot computer at {url} is not responding."
    if isinstance(e, (httpx.TransportError, OSError)):
        return f"Robot computer not reachable at {url}. Check that it is on and paired (secure tunnel up)."
    return f"Could not connect to the robot computer at {url} ({type(e).__name__}: {e})"[:300]


def readiness_error(health: dict[str, Any]) -> Optional[str]:
    """When the state reported by the skill server has a reason a job cannot start, return one sentence for the operator (primary reason first); return None if a job can start."""
    from ..robot.remote import readiness_problems

    probs = readiness_problems(health)
    if not probs:
        return None
    if health.get("stopped"):                         # E-stop and fault are mentioned first; other reasons are appended after
        head, key = "The robot computer is stopped (E-stop). Reset it on site before the next job.", "stopped (E-stop)"
    elif health.get("fault"):
        head, key = ("The robot computer has an unreset fault. Check the robot and reset it on site before the next job.",
                     "unreset fault")
    else:
        head, key = probs[0][:1].upper() + probs[0][1:] + ".", probs[0]
    rest = [p for p in probs if key not in p]
    return head + ("" if not rest else " Also: " + "; ".join(rest) + ".")


def drift_problem(drift: dict[str, Any], max_px: float) -> Optional[str]:
    """Camera marker check before starting a job (result of scripts/calibrate_site.verify_drift): reject if drift exceeds the limit or any marker is not visible."""
    if drift.get("max_px") is None or float(drift["max_px"]) > max_px or drift.get("missing"):
        return (f"The site camera may have moved (marker drift over {max_px:g} px or a marker is missing). "
                "Calibrate the camera and the clean reference again.")
    return None


def topdown_identity(grid: Any, px_per_m: float):
    """The simulated topdown image is itself a top-down view: homography from base xy to top-down image pixels (rows along x, columns along y)."""
    from ..perception.camera import PlaneHomography
    from ..perception.rectify import rectified_shape

    k = px_per_m
    x0, y0 = grid.origin
    H = np.array([[0.0, k, -k * y0], [k, 0.0, -k * x0], [0.0, 0.0, 1.0]])
    h, w = rectified_shape(grid, px_per_m)
    return PlaneHomography(H, w, h)


def fetch_jpeg(url: str, timeout: float = 10.0) -> np.ndarray:
    import httpx

    from ..robot.server import auth_headers

    r = httpx.get(url, timeout=timeout, headers=auth_headers())
    r.raise_for_status()
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


@dataclass
class RemoteRobotLink:
    ex: Any                                   # RemoteSkillExecutor
    health: dict[str, Any]
    grid: Any
    src: Any                                  # HttpCameraSource
    reference: np.ndarray
    forbidden: list[str]
    sim: bool                                 # the robot computer runs a simulation backend
    camera_url: str                           # the camera image is fetched from here (simulation backend = the skill server itself)
    meta: dict[str, Any] = field(default_factory=dict)


def platform_label(key: str) -> str:
    from ..robot.platforms import load_platforms
    spec = load_platforms().get(key)
    return spec.label if spec else key


def connect_robot(skill_url: str, camera_url: str, site: Optional[str], forbidden: Sequence[str] = (),
                  expect_backend: Optional[str] = None, max_drift_px: float = 6.0, allow_sim: bool = True,
                  require_ready: bool = False, expect_platform: str = "", zones: Sequence[str] = ()) -> RemoteRobotLink:
    """Connect to the skill server, fetch zones and the reference image, build the camera source. Raise RemoteSetupError when a job cannot start (English message; same for unreachable / token rejected).
    require_ready: reject right away if the skill server is software-stopped, has an unreset fault, is busy, not armed, or fails the startup check, as used by the software's live jobs;
    the terminal trial scripts/run_real.py leaves it to the TaskRunner startup check, with the outcome recorded as rejected plus one trial record line.
    expect_platform: the robot platform registered for the site; reject if the platform reported by the robot computer (/health platform) differs.
    zones / forbidden: zones and forbidden zones named by the task; reject outright any not in the robot computer's zones (so each skill does not get its own 422)."""
    from ..robot.remote import RemoteSkillExecutor

    try:
        ex = RemoteSkillExecutor(skill_url, forbidden_zones=forbidden)
    except Exception as e:  # noqa: BLE001
        raise RemoteSetupError(describe_link_error(e, skill_url)) from e
    try:
        return _connect(ex, skill_url, camera_url, site, forbidden, expect_backend, max_drift_px, allow_sim, require_ready,
                        expect_platform, zones)
    except BaseException as e:
        ex.close()                                    # cannot start a job: close the connection pool
        if isinstance(e, Exception) and not isinstance(e, RemoteSetupError):
            raise RemoteSetupError(describe_link_error(e, skill_url)) from e
        raise


def _connect(ex: Any, skill_url: str, camera_url: str, site: Optional[str], forbidden: Sequence[str],
             expect_backend: Optional[str], max_drift_px: float, allow_sim: bool, require_ready: bool,
             expect_platform: str = "", zones: Sequence[str] = ()) -> RemoteRobotLink:
    from ..common.config import load_yaml
    from ..common.zones import ZoneGrid
    from ..perception.camera import PlaneHomography
    from ..perception.rectify import rectified_shape
    from ..perception.sources import HttpCameraSource

    health = ex.health()
    sim = health["backend"] in SIM_BACKENDS
    if sim and not allow_sim:
        raise RemoteSetupError(f"The robot computer runs a simulation backend ({health['backend']}), not a robot.")
    if expect_backend and health["backend"] != expect_backend:
        raise RemoteSetupError(f"The robot computer's skill server backend is {health['backend']}, expected {expect_backend}.")
    actual = str(health.get("platform") or "")
    if expect_platform and actual and actual != expect_platform:
        raise RemoteSetupError(f"The robot computer runs {platform_label(actual)}, but this site is set up for "
                               f"{platform_label(expect_platform)}. Choose the matching site or robot.")
    if require_ready and (msg := readiness_error(health)):
        raise RemoteSetupError(msg)
    grid = ex.grid
    for what, ids in (("No-wipe zone", forbidden), ("Zone", zones)):
        bad = [z for z in ids if z not in grid]
        if bad:
            many = len(bad) > 1
            raise RemoteSetupError(f"{what}{'s' if many else ''} {', '.join(bad)} {'are' if many else 'is'} not on the robot computer's surface "
                                   f"(its grid is {grid.rows} × {grid.cols}). Update the site's zones or calibrate it again.")
    fb = sorted(set(forbidden) | set(health.get("site_forbidden") or []))        # site forbidden zones are always merged into the task
    ex.forbidden = set(fb)
    meta: dict[str, Any] = {"skill_server": health, "site": None if sim else site}
    if sim:
        if health.get("sim_source") not in (None, "topdown"):    # below, the homography is built assuming "the fixed camera image is the top-down view"
            raise RemoteSetupError("The robot computer's simulated camera must be top-down: start its skill server with --source topdown.")
        cam = topdown_identity(grid, 2000.0)
        try:
            reference = fetch_jpeg(f"{skill_url.rstrip('/')}/sim/reference")
        except Exception as e:  # noqa: BLE001
            raise RemoteSetupError(f"Could not read the clean reference from the robot computer ({type(e).__name__}).") from e
        camera_url = skill_url
    else:
        sd = site_dir(site)
        local = ZoneGrid(load_yaml("zones.yaml", site=site)) if site else ZoneGrid()
        if not (np.allclose(local.origin, grid.origin) and np.allclose(local.size, grid.size)
                and (local.rows, local.cols) == (grid.rows, grid.cols) and abs(local.z_surface - grid.z_surface) < 1e-6):
            raise RemoteSetupError("This site's zone grid differs from the robot computer's. Sync the site to the robot computer "
                                   "or calibrate it again.")
        if sd is None or not (sd / "calib.json").is_file():
            raise RemoteSetupError(f"Site {site!r} has no camera calibration on this computer yet. Calibrate the site first.")
        if not (sd / "reference.png").is_file():
            raise RemoteSetupError(f"Site {site!r} has no clean reference image yet. Calibrate the site first.")
        calib = json.loads((sd / "calib.json").read_text(encoding="utf-8"))
        cam = PlaneHomography.from_dict(calib["fixed"])
        reference = cv2.cvtColor(cv2.imread(str(sd / "reference.png")), cv2.COLOR_BGR2RGB)
        if reference.shape[:2] != rectified_shape(grid):
            raise RemoteSetupError(f"The clean reference {reference.shape[:2]} does not match the zone grid {rectified_shape(grid)}. "
                                   "Capture the clean reference again.")
        import sys
        if str(REPO_ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from calibrate_site import verify_drift

        try:
            shot = fetch_jpeg(f"{camera_url.rstrip('/')}/frame/fixed")
        except Exception as e:  # noqa: BLE001
            raise RemoteSetupError(f"Camera service not reachable at {camera_url} ({type(e).__name__}).") from e
        drift = verify_drift(shot, calib)
        meta["marker_drift"] = drift
        if (msg := drift_problem(drift, max_drift_px)):
            raise RemoteSetupError(msg)
    src = HttpCameraSource(camera_url, grid, {"fixed": cam})
    return RemoteRobotLink(ex, health, grid, src, reference, fb, sim, camera_url, meta)
