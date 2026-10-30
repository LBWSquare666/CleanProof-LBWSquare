"""Site calibration (in-app; video narration "calibrate the site once", wizard Site step "Calibrated on site … · 15 zones · clean reference saved").

Two modes (the how parameter of /api/setup/site):
- auto: the robot computer is a simulation (built-in local simulation, or a paired robot computer running the mujoco / rtde_fake backend), one-click calibration:
  zone grid and tabletop height come from the simulated scene (built-in: scene built from the site job_scenario; robot computer: skill server GET /config),
  the clean reference is the top view before stains are placed (built-in: SimCameraSource(topdown).rectified(); robot computer: GET /sim/reference).
- guided: real robot (paired robot computer running ursim / ur3e), going step by step in the software through the same procedure as scripts/calibrate_site.py:
  markers (print the markers) → touch x4 (on-site staff Freedrive the polishing pad onto each marker center, the software reads the TCP pose) → survey (joint angles of the survey pose) →
  capture (one frame from the fixed camera, detect markers → homography + reprojection error) → zones (plane fit, inscribed rectangle, reachability) →
  reference (multi-frame median → top-view clean reference) → sync (scripts/robot_pc/push_site.sh syncs the site config to the robot computer and restarts the service).
  The robot is read only via skill server GET /state (tcp_pose, q) and images only via the camera service GET /frame/fixed; this process creates no RTDE interface and
  sends no motion command (the arm is moved by hand by on-site staff using Freedrive on the teach pendant). Intermediate results are written to configs/site/<site>/calib_points.json, same as the command line.

zones.yaml / robot.yaml / cameras.yaml only get the keys measured by calibration changed, via update_yaml_file: edits line by line, comments, scene keys such as table.center_xy / mug /
survey_arm_q are kept as is; the file is not written if no value changed.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from ..common.config import CONFIG_DIR, REPO_ROOT, load_yaml

MARKER_IDS = (0, 1, 2, 3)
PUSH_SITE = REPO_ROOT / "scripts" / "robot-pc" / "push_site.sh"     # deploy script: syncs the site config from the GPU workstation to the robot computer
STEPS = (("markers", "Print and place the markers"), ("touch", "Touch the four markers"), ("survey", "Record the survey pose"),
         ("capture", "Locate the camera"), ("zones", "Lay out the zones"), ("reference", "Capture the clean reference"),
         ("sync", "Send to the robot computer"))
STEP_IDS = tuple(s for s, _ in STEPS)
REF_FRAMES, REF_INTERVAL_S = 12, 0.25                            # clean reference: median of 12 frames (same as the command-line default)
STATE_POLL_S, STATE_POLL_HZ = 0.5, 20.0                          # reading the TCP pose: average of several readings within about 0.5 s
SYNC_TIMEOUT_S = 240.0


class CalibrationError(ValueError):
    """A calibration step cannot be done: the message is shown directly to the operator (in English)."""


# ---------------------------------------------------------------- YAML: change only the given keys (line by line)
_KEY = re.compile(r"^(?P<ind> *)(?P<key>[A-Za-z_][\w\-]*)\s*:(?P<rest>(?:\s.*)?)$")


def _plain(v: Any) -> Any:
    """numpy scalar / array → native Python type (the value written into YAML)."""
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def _fmt(v: Any) -> str:
    """Single-line YAML value (lists and mappings in flow style, matching the hand-written style in the repo)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(float(round(v, 6)))
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + " }"
    if v is None:
        return "null"
    s = yaml.safe_dump(v, default_flow_style=True, allow_unicode=True).strip()
    return s[:-3].strip() if s.endswith("...") else s


def _indent(ln: str) -> int:
    return len(ln) - len(ln.lstrip(" "))


def _content(ln: str) -> bool:
    s = ln.strip()
    return bool(s) and not s.startswith("#")


def _split_comment(rest: str) -> tuple[str, str]:
    """Text after the key → (value, trailing comment including leading whitespace). A # inside quotes is not a comment."""
    in_s = in_d = False
    for i, ch in enumerate(rest):
        if ch == "'" and not in_d:
            in_s = not in_s
        elif ch == '"' and not in_s:
            in_d = not in_d
        elif ch == "#" and not in_s and not in_d and (i == 0 or rest[i - 1] in " \t"):
            val = rest[:i].rstrip()
            return val, rest[len(val):]
    return rest.rstrip(), ""


def _extent(lines: list[str], i: int, ind: int) -> int:
    """Last line (inclusive) occupied by the value of the key on line i: child lines of block style, multi-line flow lists / mappings."""
    val, _ = _split_comment(_KEY.match(lines[i])["rest"])
    v = val.strip()
    if v.startswith(("[", "{")):
        depth, j = 0, i
        text = v
        while True:
            for ch in text:
                if ch in "[{":
                    depth += 1
                elif ch in "]}":
                    depth -= 1
            if depth <= 0 or j + 1 >= len(lines):
                return j
            j += 1
            text = _split_comment(lines[j])[0] if "#" in lines[j] else lines[j]
    if v and not v.startswith(("|", ">")):
        return i
    last = i
    for j in range(i + 1, len(lines)):
        if _content(lines[j]):
            if _indent(lines[j]) <= ind:
                break
            last = j
    return last


def _render(path: tuple[str, ...], value: Any, ind: int) -> list[str]:
    if len(path) == 1:
        return [f"{' ' * ind}{path[0]}: {_fmt(value)}"]
    return [f"{' ' * ind}{path[0]}:"] + _render(path[1:], value, ind + 2)


def _set_in(lines: list[str], lo: int, hi: int, path: tuple[str, ...], value: Any, ind: int) -> None:
    key = path[0]
    i = None
    for n in range(lo, hi):
        m = _KEY.match(lines[n])
        if m and len(m["ind"]) == ind and m["key"] == key:
            i = n
            break
    if i is None:                                            # key not present: append after the last non-empty line of this section (at the end if there is none)
        ins = hi
        for n in range(lo, hi):
            if _content(lines[n]):
                ins = n + 1
        lines[ins:ins] = _render(path, value, ind)
        return
    m = _KEY.match(lines[i])
    val, com = _split_comment(m["rest"])
    e = _extent(lines, i, ind)
    if not com and e > i and val.strip().startswith(("[", "{")):     # multi-line flow value: the trailing comment is on the last line
        com = _split_comment(lines[e])[1]
    if len(path) == 1:
        lines[i:e + 1] = [f"{' ' * ind}{key}: {_fmt(value)}{com}"]
        return
    v = val.strip()
    if v.startswith("{"):                                    # flow mapping (e.g. table: { z_surface: 0.0 }): parse, change the value, rewrite the whole line
        cur = yaml.safe_load("\n".join([v] + [ln for ln in lines[i + 1:e + 1]])) or {}
        node = cur
        for k in path[1:-1]:
            node = node.setdefault(k, {})
        node[path[-1]] = value
        lines[i:e + 1] = [f"{' ' * ind}{key}: {_fmt(cur)}{com}"]
        return
    if v:
        raise CalibrationError(f"Cannot update {'.'.join(path)}: {key} is not a mapping")
    if e == i:                                               # empty mapping: write the child key below
        lines[i + 1:i + 1] = _render(path[1:], value, ind + 2)
        return
    ci = next(_indent(lines[n]) for n in range(i + 1, e + 1) if _content(lines[n]))
    _set_in(lines, i + 1, e + 1, path[1:], value, ci)


def _get(d: Any, path: tuple[str, ...]) -> Any:
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _put(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    for k in path[:-1]:
        if not isinstance(d.get(k), dict):
            d[k] = {}
        d = d[k]
    d[path[-1]] = value


def _same(a: Any, b: Any, tol: float = 1e-6) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    return a == b


def update_yaml_file(path: Path, updates: dict[tuple[str, ...], Any], header: str = "") -> bool:
    """Change only the keys in updates (path → new value), leaving other lines as they are (comments, style, trailing comments). If no value changed the file is not written and False is returned.
    After the edit, re-parse and verify: the file is written only if just these keys changed and no other key moved (otherwise CalibrationError is raised and the file is untouched)."""
    updates = {tuple(p): _plain(v) for p, v in updates.items()}
    txt = path.read_text(encoding="utf-8") if path.is_file() else ""
    old = (yaml.safe_load(txt) or {}) if txt else {}
    need = {p: v for p, v in updates.items() if not _same(_get(old, p), v)}
    if not need:
        return False
    lines = txt.splitlines() if txt else ([header] if header else [])
    for p, v in need.items():
        _set_in(lines, 0, len(lines), p, v, 0)
    out = "\n".join(lines) + "\n"
    try:
        back = yaml.safe_load(out) or {}
    except yaml.YAMLError as e:
        raise CalibrationError(f"Cannot update {path.name} automatically ({type(e).__name__}); edit it by hand") from e
    want = copy.deepcopy(old)
    for p, v in need.items():
        _put(want, p, v)
    if not _same(back, want):
        raise CalibrationError(f"Cannot update {path.name} automatically; edit it by hand")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="utf-8")
    return True


# ---------------------------------------------------------------- calibration result files
def site_zones(d: Path) -> dict[str, Any]:
    """Effective zone config of a site: default configs/zones.yaml deep-merged with the site zones.yaml (same as load_yaml(site=), but read from the given directory)."""
    from ..common.config import _deep_merge
    base = load_yaml(CONFIG_DIR / "zones.yaml")
    f = d / "zones.yaml"
    return _deep_merge(base, (yaml.safe_load(f.read_text(encoding="utf-8")) or {}) if f.is_file() else {})


def grid_updates(z_surface: float, origin: Any, size: Any, rows: int, cols: int) -> dict[tuple[str, ...], Any]:
    """Zone keys determined by calibration: tabletop height, workspace corners and size, row / column counts (same keys as written by scripts/calibrate_site.py zones)."""
    return {("table", "z_surface"): round(float(z_surface), 4),
            ("work_area", "origin_xy"): [round(float(v), 4) for v in origin],
            ("work_area", "size_xy"): [round(float(v), 4) for v in size],
            ("work_area", "rows"): int(rows), ("work_area", "cols"): int(cols)}


ZONES_HEADER = "# written by in-app calibration (zone grid and tabletop height); keys not written use the defaults in configs/zones.yaml."


def read_record(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / "calibration.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_record(d: Path, source: str, zones: dict[str, Any], **extra: Any) -> dict[str, Any]:
    wa = zones["work_area"]
    rec = {"calibrated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source,
           "rows": int(wa["rows"]), "cols": int(wa["cols"]), "size_m": [float(v) for v in wa["size_xy"]], **_plain(extra)}
    (d / "calibration.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def _write_png(path: Path, rgb: np.ndarray) -> None:
    ok, enc = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise CalibrationError("Could not encode the clean reference")
    path.write_bytes(enc.tobytes())


def surface_render(d: Path, surface: str, sim_site: Optional[str] = None) -> np.ndarray:
    """Top-view render of the site tabletop (this is what the page shows when no reference image has been captured; sim_site = which site the simulation config is read from)."""
    from ..common.zones import ZoneGrid
    from ..sim.dirt import DirtMap
    sim_cfg = load_yaml("sim.yaml", site=sim_site if sim_site is not None else d.name)
    rgb = DirtMap(ZoneGrid(site_zones(d)), sim_cfg["dirt"], surface=surface or "sealed_wood").rgb()
    return np.clip(rgb, 0, 255).astype(np.uint8)


def visually_same(a: np.ndarray, b: np.ndarray) -> tuple[bool, float]:
    """Whether two top views look the same (small JPEG compression differences do not count): mean difference <= 2 levels, 99th percentile <= 16 levels."""
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mean = float(diff.mean())
    return bool(mean <= 2.0 and float(np.percentile(diff, 99)) <= 16.0), mean


# ---------------------------------------------------------------- auto: one-click simulation calibration
def capture_builtin_sim(job_scenario: str) -> tuple[dict[str, Any], np.ndarray, Optional[str]]:
    """Built-in local simulation: build the scene from the site job template (same spec as preview_frame / job creation), returns (zone config, clean top-view reference before stains are placed, scene site)."""
    from ..common.config import current_site, use_site
    from ..perception.sources import SimCameraSource
    from ..sim.scenarios import build, load_scenario
    prev = current_site()
    spec = load_scenario(job_scenario)
    try:
        sc = build(job_scenario, spec=spec)
        try:
            ref = SimCameraSource(sc.robot, sc.grid, mode="topdown").rectified()     # before stains are placed (build adds no stains, add_stains does)
            g = sc.grid
            zones = {"table": {"z_surface": float(g.z_surface)},
                     "work_area": {"origin_xy": g.origin.tolist(), "size_xy": g.size.tolist(), "rows": g.rows, "cols": g.cols}}
        finally:
            sc.close()
    finally:
        use_site(prev or None)                                 # build() switches the global site: switch back
    return zones, ref, spec.get("site")


def capture_robot_sim(skill_url: str) -> tuple[dict[str, Any], np.ndarray]:
    """Paired robot computer running a simulation backend: zones come from GET /config (the grid it is actually using), the clean reference from GET /sim/reference."""
    import httpx

    from ..agent.remote_job import fetch_jpeg
    from ..robot.server import auth_headers
    r = httpx.get(skill_url.rstrip("/") + "/config", headers=auth_headers(), timeout=10.0)
    r.raise_for_status()
    z = r.json()["zones"]
    zones = {"table": {"z_surface": float(z["table"]["z_surface"])},
             "work_area": {k: z["work_area"][k] for k in ("origin_xy", "size_xy", "rows", "cols")}}
    return zones, fetch_jpeg(skill_url.rstrip("/") + "/sim/reference", timeout=20.0)


def apply_auto(d: Path, zones: dict[str, Any], ref: np.ndarray, source: str, surface: str, compare_look: bool,
               sim_site: Optional[str] = None, **extra: Any) -> dict[str, Any]:
    """Write the results of one-click calibration: zones.yaml (only zone keys changed), reference.png, calibration.json.

    How the reference image is stored (keep the look from the video while recording the calibration honestly):
    - Built-in local simulation: what is captured is SimCameraSource(topdown).rectified(), which comes from the same function and
      the same config as the tabletop render the page showed before (DirtMap.rgb()), pixel-identical (measured difference 0 for home_dining and carwash_van); the captured image is stored directly.
    - Simulation backend on the robot computer (GET /sim/reference, JPEG): compared with the tabletop render; if they look the same (mean difference <= 2 levels, about 0.9 measured at JPEG q98)
      the captured image is stored. If they look different (skill server run with --source render: the reference is a rectified 3D render), reference.png stores the top-view tabletop render
      (this is the clean tabletop seen by the simulated top camera, so the page stays unchanged), the captured image is saved separately as reference_capture.png, and calibration.json records both and the difference.
      Jobs on the simulation backend already use the robot computer own /sim/reference as the baseline (remote_job.connect_robot), and reference.png is only for display,
      so storing it this way does not affect any detection result. For a real robot (guided) the median reference captured on site is always stored."""
    a = grid_updates(zones["table"]["z_surface"], zones["work_area"]["origin_xy"], zones["work_area"]["size_xy"],
                     zones["work_area"]["rows"], zones["work_area"]["cols"])
    changed = update_yaml_file(d / "zones.yaml", a, ZONES_HEADER)
    eff = site_zones(d)
    from ..common.zones import ZoneGrid
    from ..perception.rectify import rectified_shape
    h, w = rectified_shape(ZoneGrid(eff))
    if ref.shape[:2] != (h, w):
        ref = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    info: dict[str, Any] = {"file": "reference.png", "kind": "captured"}
    if compare_look:
        look = surface_render(d, surface, sim_site)
        same, mean = visually_same(ref, look)
        info["mean_diff_vs_surface_render"] = round(mean, 3)
        if not same:
            _write_png(d / "reference_capture.png", ref)
            info.update(kind="surface_render", capture="reference_capture.png")
            ref = look
    _write_png(d / "reference.png", ref)
    rec = write_record(d, source, eff, reference=info, zones_changed=changed, **extra)
    return rec


# ---------------------------------------------------------------- guided: on-site calibration of a real robot
def calib_cli() -> Any:
    """Pure functions from scripts/calibrate_site.py (plane fit, inscribed rectangle, marker detection, drift check, marker print page)."""
    mod = sys.modules.get("calibrate_site")
    if mod is None:
        spec = importlib.util.spec_from_file_location("calibrate_site", REPO_ROOT / "scripts" / "calibrate_site.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["calibrate_site"] = mod
        spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


def markers_png(size_mm: float = 40.0) -> bytes:
    page = calib_cli().marker_sheet(size_mm)
    ok, enc = cv2.imencode(".png", page)
    if not ok:
        raise CalibrationError("Could not render the marker sheet")
    return enc.tobytes()


def fetch_state(skill_url: str) -> dict[str, Any]:
    """Robot readings: only via skill server GET /state (tcp_pose, q; shared token)."""
    import httpx

    from ..robot.server import auth_headers
    r = httpx.get(skill_url.rstrip("/") + "/state", headers=auth_headers(), timeout=3.0)
    if r.status_code == 401:
        raise CalibrationError("The skill server refused the shared token")
    r.raise_for_status()
    return r.json()


def fetch_frame(camera_url: str) -> np.ndarray:
    """One frame from the fixed camera (RGB): only via the camera service GET /frame/fixed."""
    import httpx

    from ..robot.server import auth_headers
    r = httpx.get(camera_url.rstrip("/") + "/frame/fixed", params={"fmt": "png"}, headers=auth_headers(), timeout=8.0)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise CalibrationError("The camera service returned an image that cannot be read")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fetch_camera_health(camera_url: str) -> dict[str, Any]:
    import httpx

    from ..robot.server import auth_headers
    return httpx.get(camera_url.rstrip("/") + "/health", headers=auth_headers(), timeout=5.0).json()


def _avg_state(skill_url: str, key: str, seconds: float = STATE_POLL_S) -> np.ndarray:
    vals = []
    n = max(1, int(round(seconds * STATE_POLL_HZ)))
    for k in range(n):
        s = fetch_state(skill_url)
        if key not in s:
            raise CalibrationError(f"The skill server did not report {key}")
        vals.append(np.asarray(s[key], dtype=float))
        if k + 1 < n:
            time.sleep(1.0 / STATE_POLL_HZ)
    return np.mean(vals, axis=0)


def _robot_error(e: Exception, what: str) -> CalibrationError:
    if isinstance(e, CalibrationError):
        return e
    return CalibrationError(f"Cannot read the {what} ({type(e).__name__}); check that the robot computer is connected")


def load_points(d: Path) -> dict[str, Any]:
    p = d / "calib_points.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except json.JSONDecodeError:
        return {}


def save_points(d: Path, data: dict[str, Any]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "calib_points.json").write_text(json.dumps(_plain(data), indent=1, ensure_ascii=False), encoding="utf-8")


def _calib(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / "calib.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def guided_state(site: str, d: Path) -> dict[str, Any]:
    """Whether each step is done, with a result summary; if an earlier step is redone (newer timestamp), later steps that depend on it count as not done."""
    pts = load_points(d)
    app = pts.get("app") or {}
    touch = pts.get("touch") or {}
    touched = sorted(int(k) for k in touch if k.isdigit() and int(k) in MARKER_IDS)
    t_touch = max([float(v.get("t", 0)) for v in touch.values()] or [0.0])
    cal = _calib(d)
    t_cap = float(cal.get("t", 0)) if cal and "fixed" in cal else 0.0
    zinfo, rinfo, sinfo = app.get("zones") or {}, app.get("reference") or {}, app.get("sync") or {}
    t_z, t_r, t_s = float(zinfo.get("t", 0)), float(rinfo.get("t", 0)), float(sinfo.get("t", 0))
    done = {"markers": bool(app.get("markers")),
            "touch": len(touched) == len(MARKER_IDS),
            "survey": "survey_q" in pts,
            "capture": t_cap > 0 and t_cap >= t_touch and len(touched) == len(MARKER_IDS),
            "zones": t_z > 0 and t_z >= t_touch and len(touched) == len(MARKER_IDS),
            "reference": False, "sync": False}
    done["reference"] = t_r > 0 and done["capture"] and done["zones"] and t_r >= max(t_cap, t_z) and (d / "reference.png").is_file()
    done["sync"] = t_s > 0 and done["reference"] and t_s >= t_r
    detail = {
        "markers": "Marker sheet printed · tape the markers 3–4 cm outside the corners" if done["markers"]
        else "Print the marker sheet at 100 % and tape markers 0–3 just outside the corners of the area to clean",
        "touch": f"{len(touched)} of {len(MARKER_IDS)} markers touched"
        + (f" · surface flat within {app.get('touch_span_mm', 0):.1f} mm" if done["touch"] and "touch_span_mm" in app else ""),
        "survey": "Survey pose recorded" if done["survey"] else "Move the arm clear of the camera view, then record",
        "capture": (f"{len(MARKER_IDS)} markers found · {cal.get('mm_per_px_mean', 0):.2f} mm per pixel" if done["capture"]
                    else "Take a camera frame with all four markers in view"),
        "zones": zinfo.get("detail", "") if done["zones"] else "Choose rows and columns; the zone grid covers the area between the markers",
        "reference": "Clean reference saved" if done["reference"] else "Clean the surface, keep the arm in the survey pose, then capture",
        "sync": sinfo.get("detail", "") if done["sync"] else "Copy the calibration to the robot computer and restart its services",
    }
    steps = [{"id": s, "title": t, "done": bool(done[s]), "detail": detail[s]} for s, t in STEPS]
    nxt = next((s["id"] for s in steps if not s["done"]), None)
    return {"site": site, "steps": steps, "touched": touched, "next": nxt}


def _reach_problems(zones_cfg: dict[str, Any], rcfg: dict[str, Any]) -> list[str]:
    """Reachability (UR series, same checks as scripts/calibrate_site.py reach_report; English text shown to the operator)."""
    from ..common.zones import ZoneGrid
    from ..robot.kinematics import UR3eKinematics, elbow_margin_deg
    g = ZoneGrid(zones_cfg)
    kin = UR3eKinematics(np.array(rcfg["startup"]["expected_tcp_offset"], dtype=float))
    rmax = float(rcfg["safety"]["max_horizontal_radius_m"])
    pad = float(rcfg["tool"]["pad_radius_m"])
    zh = g.z_surface + float(rcfg["motion"]["hover_above_surface_m"])
    q_near = np.array(rcfg["home_q"], dtype=float)
    probs = []
    for zid in g.ids:
        z = g[zid]
        far = max(np.hypot(x, y) for x, y in [(z.xmin, z.ymin), (z.xmin, z.ymax), (z.xmax, z.ymin), (z.xmax, z.ymax)]) + pad
        if far > rmax:
            probs.append(f"zone {zid} reaches {far:.3f} m from the robot base (limit {rmax:.3f} m)")
        cx, cy = z.center
        q = kin.ik(np.array([cx, cy, zh, *g.tool_rotvec]), q_near)
        if q is None:
            probs.append(f"zone {zid}: the arm cannot reach the centre")
        elif elbow_margin_deg(q) < float(rcfg["safety"]["min_elbow_margin_deg"]):
            probs.append(f"zone {zid}: elbow too close to its limit ({elbow_margin_deg(q):.0f}°)")
    return probs


def guided_step(site: str, d: Path, step: str, body: dict[str, Any], skill_url: str, camera_url: str,
                host: Optional[str] = None, robot: str = "") -> dict[str, Any]:
    """Do one step and return the latest state. Raises CalibrationError when it cannot be done (the API returns 400 with this message)."""
    if step not in STEP_IDS:
        raise CalibrationError(f"Unknown calibration step {step!r}")
    d.mkdir(parents=True, exist_ok=True)
    pts = load_points(d)
    app = pts.setdefault("app", {})
    now = time.time()
    cli = calib_cli()

    if step == "markers":
        app["markers"] = {"t": now, "size_mm": float(body.get("size_mm") or 40.0)}

    elif step == "touch":
        try:
            i = int(body.get("marker"))
        except (TypeError, ValueError):
            raise CalibrationError("Say which marker (0–3) the pad is touching") from None
        if i not in MARKER_IDS:
            raise CalibrationError("Markers are numbered 0–3")
        try:
            pose = _avg_state(skill_url, "tcp_pose")
            st = fetch_state(skill_url)
        except Exception as e:  # noqa: BLE001
            raise _robot_error(e, "robot position") from e
        rec = {"tcp_pose": pose.tolist(), "t": now}
        if "tcp_force" in st:
            rec["tcp_force"] = [float(v) for v in st["tcp_force"]]
        pts.setdefault("touch", {})[str(i)] = rec
        touch = pts["touch"]
        if all(str(k) in touch for k in MARKER_IDS):
            P = np.array([touch[str(k)]["tcp_pose"][:3] for k in MARKER_IDS])
            app["touch_span_mm"] = cli.fit_plane_z(P)["z_span_mm"]

    elif step == "survey":
        try:
            q = _avg_state(skill_url, "q", 0.3)
        except Exception as e:  # noqa: BLE001
            raise _robot_error(e, "arm joint angles") from e
        pts["survey_q"] = q.tolist()
        # written to the site zones.yaml right away, like the command line (only this one key; without a workspace the site still counts as uncalibrated)
        update_yaml_file(d / "zones.yaml", {("survey_q",): [round(float(v), 6) for v in q]},
                         "# written by in-app on-site calibration (cleanproof/ui/calibration.py, same keys as scripts/calibrate_site.py)")

    elif step == "capture":
        touch = pts.get("touch") or {}
        if not all(str(k) in touch for k in MARKER_IDS):
            raise CalibrationError("Touch all four markers first")
        try:
            img = fetch_frame(camera_url)
        except Exception as e:  # noqa: BLE001
            raise _robot_error(e, "camera image") from e
        found = cli.detect_markers(img)
        miss = [k for k in MARKER_IDS if k not in found]
        if miss:
            cv2.imwrite(str(d / "capture_failed.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            raise CalibrationError(f"Marker{'s' if len(miss) > 1 else ''} {', '.join(map(str, miss))} not found in the camera image: "
                                   "check that the arm is not covering them, and the light and focus")
        from ..perception.camera import PlaneHomography
        xy = np.array([touch[str(k)]["tcp_pose"][:2] for k in MARKER_IDS])
        px = np.array([found[k] for k in MARKER_IDS])
        try:
            cam = PlaneHomography.from_points(xy, px, img.shape[1], img.shape[0])
        except ValueError:
            raise CalibrationError("The touch points and the markers do not form a usable quadrilateral; touch the markers again") from None
        err = cam.reprojection_error_px(xy, px)
        # a 4-point homography has no redundancy, so the error is always 0; check consistency with opposite-side length ratios (same as the command line)
        mm_per_px = [float(np.linalg.norm(xy[a] - xy[b]) * 1000 / max(np.linalg.norm(px[a] - px[b]), 1e-9))
                     for a, b in ((0, 1), (2, 3), (0, 2), (1, 3))]
        calib = {"fixed": cam.to_dict(), "marker_px": {str(k): found[k].tolist() for k in MARKER_IDS},
                 "marker_xy": {str(k): xy[n].tolist() for n, k in enumerate(MARKER_IDS)},
                 "reproj_err_px": err.tolist(), "mm_per_px_edges": mm_per_px, "mm_per_px_mean": float(np.mean(mm_per_px)), "t": now}
        spread = (max(mm_per_px) - min(mm_per_px)) / max(np.mean(mm_per_px), 1e-9)
        if spread > 0.15:                                      # inconsistent: do not overwrite the existing calib.json, this step does not count as done; the result is saved separately for reference
            calib["rejected"] = f"edge scales differ by {spread * 100:.0f} % (more than 15 %)"
            (d / "calib_rejected.json").write_text(json.dumps(_plain(calib), indent=1), encoding="utf-8")
            cv2.imwrite(str(d / "capture_rejected.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            raise CalibrationError("The four marker distances disagree by more than 15 %: a touch point or a marker detection is off; "
                                   "touch the markers again")
        (d / "calib.json").write_text(json.dumps(_plain(calib), indent=1), encoding="utf-8")
        cv2.imwrite(str(d / "capture_raw.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    elif step == "zones":
        touch = pts.get("touch") or {}
        if not all(str(k) in touch for k in MARKER_IDS):
            raise CalibrationError("Touch all four markers first")
        if "survey_q" not in pts:                              # the real-robot skill server requires the site to have a survey pose to start
            raise CalibrationError("Record the survey pose first")
        cur = site_zones(d)["work_area"]
        try:
            rows = int(body["rows"] if body.get("rows") is not None else cur["rows"])
            cols = int(body["cols"] if body.get("cols") is not None else cur["cols"])
            inset = float(body["inset_mm"]) / 1000.0 if body.get("inset_mm") is not None else 0.035
            pad_mm = float(body["pad_diameter_mm"]) if body.get("pad_diameter_mm") else None
        except (TypeError, ValueError):
            raise CalibrationError("Rows, columns, inset and pad diameter must be numbers") from None
        if not (1 <= rows <= 8 and 1 <= cols <= 12):
            raise CalibrationError("Rows must be 1–8 and columns 1–12")
        if not 0.0 <= inset <= 0.2:
            raise CalibrationError("Inset must be 0–200 mm")
        if pad_mm is not None and not 20.0 <= pad_mm <= 300.0:
            raise CalibrationError("Pad diameter must be 20–300 mm")
        P = np.array([touch[str(k)]["tcp_pose"][:3] for k in MARKER_IDS])
        plane = cli.fit_plane_z(P)
        try:
            origin, size = cli.inscribed_rect(P[:, :2], inset)
        except ValueError:
            raise CalibrationError("The area between the markers is too small after the inset; place the markers further apart") from None
        upd = grid_updates(plane["z_median"], origin, size, rows, cols)
        if "survey_q" in pts:
            upd[("survey_q",)] = [round(float(v), 6) for v in pts["survey_q"]]
        eff = copy.deepcopy(site_zones(d))
        for p, v in upd.items():
            _put(eff, p, _plain(v))
        from ..common.config import _deep_merge
        rf = d / "robot.yaml"
        rcfg = _deep_merge(load_yaml(CONFIG_DIR / "robot.yaml"), (yaml.safe_load(rf.read_text(encoding="utf-8")) or {}) if rf.is_file() else {})
        if pad_mm is not None:
            rcfg["tool"]["pad_radius_m"] = round(pad_mm / 2000.0, 4)
        from ..common.zones import ZoneGrid
        g = ZoneGrid(eff)
        ids = set(g.ids)
        fb_old = list((yaml.safe_load((d / "zones.yaml").read_text(encoding="utf-8")) or {}).get("site_forbidden_zones") or []) \
            if (d / "zones.yaml").is_file() else []
        if any(z not in ids for z in fb_old):                  # row / column counts changed: drop site forbidden zones not in the new grid (otherwise skill server start fails)
            upd[("site_forbidden_zones",)] = [z for z in fb_old if z in ids]
        warn = []
        if plane["z_span_mm"] > 4.0:
            warn.append(f"the four touch points differ by {plane['z_span_mm']:.1f} mm in height (more than 4 mm): "
                        "the surface is not level or the pad was pressed unevenly")
        pad_d = 2 * float(rcfg["tool"]["pad_radius_m"])
        if np.any(g.cell < pad_d):
            warn.append("zones are smaller than the polishing pad: use fewer rows or columns")
        if robot.startswith("ur"):
            warn += _reach_problems(eff, rcfg)
        if warn and not body.get("force"):
            raise CalibrationError("Not saved: " + "; ".join(warn[:4]) + (f" (+{len(warn) - 4} more)" if len(warn) > 4 else ""))
        update_yaml_file(d / "zones.yaml", upd, "# written by in-app on-site calibration (cleanproof/ui/calibration.py, same keys as scripts/calibrate_site.py)")
        if pad_mm is not None:
            update_yaml_file(rf, {("tool", "pad_radius_m"): round(pad_mm / 2000.0, 4)}, "# written by in-app on-site calibration (measured polishing pad radius)")
        app["zones"] = {"t": now, "plane": plane, "warnings": warn,
                        "detail": f"{rows} × {cols} zones · {size[0] * 1000:.0f} × {size[1] * 1000:.0f} mm · "
                                  f"surface height {plane['z_median']:.3f} m" + (" · reach OK" if robot.startswith("ur") and not warn else "")}

    elif step == "reference":
        cal = _calib(d)
        st_ = guided_state(site, d)
        done = {s["id"]: s["done"] for s in st_["steps"]}
        if not (cal and done["capture"]):
            raise CalibrationError("Locate the camera first (capture)")
        if not done["zones"]:
            raise CalibrationError("Lay out the zones first")
        from ..common.zones import ZoneGrid
        from ..perception.camera import PlaneHomography
        from ..perception.rectify import draw_grid, rectify
        cam = PlaneHomography.from_dict(cal["fixed"])
        grid = ZoneGrid(site_zones(d))
        try:                                                   # the arm must be at the survey pose (not blocking the camera): read the joint angles once to check
            q_now = np.asarray(fetch_state(skill_url)["q"], dtype=float)
        except Exception as e:  # noqa: BLE001
            raise _robot_error(e, "arm joint angles") from e
        if "survey_q" in pts and float(np.max(np.abs(q_now - np.asarray(pts["survey_q"], dtype=float)))) > 0.05:
            raise CalibrationError("Move the arm back to the survey pose first (it may be in the camera view)")
        frames = []
        try:
            for k in range(REF_FRAMES):
                frames.append(fetch_frame(camera_url))
                if k + 1 < REF_FRAMES:
                    time.sleep(REF_INTERVAL_S)
        except Exception as e:  # noqa: BLE001
            raise _robot_error(e, "camera image") from e
        med = np.median(np.stack(frames), axis=0).astype(np.uint8)
        drift = cli.verify_drift(med, cal)
        if drift["max_px"] is None or drift["max_px"] > 6.0:
            raise CalibrationError("The camera seems to have moved since it was located (markers shifted or hidden): locate the camera again")
        ref = rectify(med, cam, grid)
        _write_png(d / "reference.png", ref)
        over = med.copy()
        x0, y0 = grid.origin
        x1, y1 = grid.origin + grid.size
        poly = cam.project(np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]]))[0]
        cv2.polylines(over, [np.int32(poly)], True, (255, 0, 0), 3)
        cv2.imwrite(str(d / "calib_overlay.jpg"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(d / "reference_grid.jpg"), cv2.cvtColor(draw_grid(ref, grid), cv2.COLOR_RGB2BGR))
        try:                                                   # record the locked camera exposure settings in the site cameras.yaml (same as the command line; failing to read them does not affect the reference image)
            ex = (fetch_camera_health(camera_url).get("cameras") or {}).get("fixed", {}).get("exposure", {}) or {}
            if ex.get("locked"):
                update_yaml_file(d / "cameras.yaml", {("cameras", "fixed", k): ex[k] for k in ("exposure", "gain", "white_balance")
                                                      if ex.get(k) is not None}, "# written by in-app on-site calibration (locked exposure settings)")
        except Exception:  # noqa: BLE001
            pass
        app["reference"] = {"t": now, "frames": REF_FRAMES, "shape": list(ref.shape[:2])}
        write_record(d, "guided", site_zones(d), reference={"file": "reference.png", "kind": "captured", "frames": REF_FRAMES},
                     reproj_err_px=cal.get("reproj_err_px"), mm_per_px=cal.get("mm_per_px_mean"))

    elif step == "sync":
        st_ = guided_state(site, d)
        done = {s["id"]: s["done"] for s in st_["steps"]}
        if not (done["zones"] and done["reference"]):
            raise CalibrationError("Lay out the zones and capture the clean reference first")
        if not host:
            raise CalibrationError("No paired robot computer: pair one first (install_edge.sh --pair <host>)")
        if not Path(PUSH_SITE).is_file():
            raise CalibrationError("The sync script scripts/robot_pc/push_site.sh is not installed on this computer")
        try:
            p = subprocess.run(["bash", str(PUSH_SITE), site, "--restart"], capture_output=True, text=True, timeout=SYNC_TIMEOUT_S,
                               env={**os.environ, "ROBOT_PC_HOST": host}, cwd=str(REPO_ROOT))
        except subprocess.TimeoutExpired:
            raise CalibrationError(f"Sending the calibration to {host} timed out") from None
        if p.returncode != 0:
            tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or [f"exit code {p.returncode}"]
            raise CalibrationError(f"Sending the calibration to {host} failed: {tail[0][:200]}")
        app["sync"] = {"t": now, "host": host, "detail": f"Sent to {host} · robot services restarted"}

    pts["app"] = app
    save_points(d, pts)
    return guided_state(site, d)
