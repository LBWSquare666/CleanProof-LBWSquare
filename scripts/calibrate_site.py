#!/usr/bin/env python3
"""On-site calibration (run on robot-pc, by a person at the site). **Read-only** towards the robot: only builds an RTDEReceiveInterface to read the TCP pose / joint angles;
the arm is moved by hand by site staff using Freedrive on the teach pendant. Results are written to configs/site/<site>/ (see configs/site/README.md).

Steps (each can be rerun alone, intermediate results in configs/site/<site>/calib_points.json):
  0. markers   generate an A4 printout of 4 ArUco markers (DICT_4X4_50, id 0-3). After printing, stick them about 3-4 cm **outside the four corners of the work area**
               (markers are not inside any wiping zone and stay put for the whole trial, used to check at any time whether the camera has moved).
  1. touch     use Freedrive to put the polishing pad center on the center of markers 0-3 in turn, lightly touching the table; press Enter to record the TCP pose (0.5 s average).
  2. survey    use Freedrive / the pendant to move the arm to a survey pose that does not block the camera; press Enter to record joint angles → survey_q.
  3. capture   with the arm at the survey pose, grab one frame from the fixed camera, detect the 4 markers → homography (base xy → pixels) + reprojection error + overlay image.
  4. zones     from the 4 touch points compute the table height (plane fit, flatness) and the work area (inscribed axis-aligned rectangle, then shrunk by --inset),
               check reachability (horizontal radius, inverse kinematics, elbow margin), write zones.yaml / robot.yaml (pad diameter).
  5. reference with the table wiped clean and the arm at the survey pose: take the median of N frames → top-view reference image reference.png; also record the locked exposure settings.
  6. verify    any time: re-detect the markers and compare with their pixel positions at calibration time (if the camera was bumped the error will exceed the limit, redo 3 and 5).

  python scripts/calibrate_site.py --site ur3e_site markers --out ~/aruco_markers.png
  python scripts/calibrate_site.py --site ur3e_site touch
  python scripts/calibrate_site.py --site ur3e_site survey
  python scripts/calibrate_site.py --site ur3e_site capture
  python scripts/calibrate_site.py --site ur3e_site zones --rows 3 --cols 4 --pad-diameter-mm 110
  python scripts/calibrate_site.py --site ur3e_site reference
  python scripts/calibrate_site.py --site ur3e_site verify
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.common.config import CONFIG_DIR, load_yaml, site_root  # noqa: E402

MARKER_IDS = (0, 1, 2, 3)
ARUCO_DICT = cv2.aruco.DICT_4X4_50


# ---------------------------------------------------------------- pure functions (unit tested)
def fit_plane_z(pts: np.ndarray) -> dict[str, float]:
    """Least squares for z = a*x + b*y + c; returns the median height, tilt (mm / 100 mm) and residuals."""
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
    res = pts[:, 2] - A @ coef
    tilt = float(np.hypot(coef[0], coef[1]) * 100.0)
    return {"z_median": float(np.median(pts[:, 2])), "z_span_mm": float(np.ptp(pts[:, 2]) * 1000),
            "tilt_mm_per_100mm": tilt, "resid_max_mm": float(np.abs(res).max() * 1000)}


def inscribed_rect(xy: np.ndarray, inset: float) -> tuple[np.ndarray, np.ndarray]:
    """4 marker centers (roughly a rectangle, any order) → inscribed axis-aligned rectangle, then each side shrunk by inset. Returns (origin_xy, size_xy)."""
    c = xy.mean(axis=0)
    lo_x = xy[xy[:, 0] < c[0]][:, 0].max()
    hi_x = xy[xy[:, 0] >= c[0]][:, 0].min()
    lo_y = xy[xy[:, 1] < c[1]][:, 1].max()
    hi_y = xy[xy[:, 1] >= c[1]][:, 1].min()
    origin = np.array([lo_x + inset, lo_y + inset])
    size = np.array([hi_x - lo_x - 2 * inset, hi_y - lo_y - 2 * inset])
    if np.any(size <= 0.02):
        raise ValueError(f"work area too small after shrinking: {size} (markers too close together or inset too large)")
    return origin, size


def detect_markers(img_rgb: np.ndarray) -> dict[int, np.ndarray]:
    """Returns {id: center pixel (u, v)}."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    out: dict[int, np.ndarray] = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            out[int(i)] = c.reshape(4, 2).mean(axis=0)
    return out


def reach_report(zones_cfg: dict[str, Any], rcfg: dict[str, Any]) -> list[str]:
    """Four corners and the center of each zone: horizontal radius <= limit; an inverse-kinematics solution exists at hover height with enough elbow margin. Returns a list of problems (empty = pass)."""
    from cleanproof.common.zones import ZoneGrid
    from cleanproof.robot.kinematics import UR3eKinematics, elbow_margin_deg

    g = ZoneGrid(zones_cfg)
    kin = UR3eKinematics(np.array(rcfg["startup"]["expected_tcp_offset"], dtype=float))
    rmax = float(rcfg["safety"]["max_horizontal_radius_m"])
    pad = float(rcfg["tool"]["pad_radius_m"])
    zh = g.z_surface + float(rcfg["motion"]["hover_above_surface_m"])
    q_near = np.array(rcfg["home_q"], dtype=float)
    probs = []
    for zid in g.ids:
        z = g[zid]
        for (x, y) in [(z.xmin, z.ymin), (z.xmin, z.ymax), (z.xmax, z.ymin), (z.xmax, z.ymax)]:
            if np.hypot(x, y) + pad > rmax:
                probs.append(f"{zid} corner ({x:.3f}, {y:.3f}) horizontal radius with pad radius {np.hypot(x, y) + pad:.3f} m > {rmax}")
        cx, cy = z.center
        q = kin.ik(np.array([cx, cy, zh, *g.tool_rotvec]), q_near)
        if q is None:
            probs.append(f"{zid} no inverse kinematics solution at the hover pose over the center")
        elif elbow_margin_deg(q) < float(rcfg["safety"]["min_elbow_margin_deg"]):
            probs.append(f"{zid} elbow margin {elbow_margin_deg(q):.1f} deg < {rcfg['safety']['min_elbow_margin_deg']} deg")
    return probs


# ---------------------------------------------------------------- on-site I/O
def site_paths(site: str) -> Path:
    d = site_root() / site
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_points(d: Path) -> dict[str, Any]:
    p = d / "calib_points.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def save_points(d: Path, data: dict[str, Any]) -> None:
    (d / "calib_points.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def merge_yaml(path: Path, upd: dict[str, Any]) -> None:
    cur = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    cur = cur or {}
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    path.write_text("# written by scripts/calibrate_site.py (measured on site)\n" + yaml.safe_dump(cur, allow_unicode=True, sort_keys=False),
                    encoding="utf-8")


def receive_iface(host: str):
    import rtde_receive

    return rtde_receive.RTDEReceiveInterface(host)       # read-only telemetry


def avg_reading(fn, seconds: float = 0.5, hz: float = 50.0) -> np.ndarray:
    vals = []
    for _ in range(max(1, int(seconds * hz))):
        vals.append(np.asarray(fn(), dtype=float))
        time.sleep(1.0 / hz)
    return np.mean(vals, axis=0)


def grab(camera_url: str, camera: str = "fixed") -> np.ndarray:
    import httpx

    from cleanproof.robot.server import auth_headers

    r = httpx.get(f"{camera_url.rstrip('/')}/frame/{camera}", params={"fmt": "png"}, timeout=5.0, headers=auth_headers())
    r.raise_for_status()
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- subcommands
def marker_sheet(size_mm: float = 40.0, dpi: int = 300) -> np.ndarray:
    """A4 printout page of 4 ArUco markers (DICT_4X4_50, id 0-3) (grayscale, actual size at the given dpi); the software calibration page uses it too."""
    px = int(round(size_mm / 25.4 * dpi))
    page = np.full((int(297 / 25.4 * dpi), int(210 / 25.4 * dpi)), 255, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    pos = [(0.12, 0.10), (0.12, 0.55), (0.55, 0.10), (0.55, 0.55)]
    for i, (fy, fx) in zip(MARKER_IDS, pos):
        m = cv2.aruco.generateImageMarker(d, i, px)
        y, x = int(fy * page.shape[0]), int(fx * page.shape[1])
        page[y:y + px, x:x + px] = m
        cv2.putText(page, f"id {i}  ({size_mm:.0f} mm)", (x, y + px + 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 0, 3)
    return page


def cmd_markers(args) -> int:
    page = marker_sheet(args.size_mm)
    out = Path(args.out).expanduser()
    cv2.imwrite(str(out), page)
    print(f"wrote {out} (A4, 300 dpi, print at actual size; marker side {args.size_mm} mm). Cut out and stick about 3-4 cm outside the four corners of the work area.")
    return 0


def cmd_touch(args, d: Path) -> int:
    rcfg = load_yaml("robot.yaml")
    rr = receive_iface(str(rcfg["hosts"]["ur3e"]))
    data = load_points(d)
    pts = data.setdefault("touch", {})
    print("Switch the pendant to Freedrive. For each marker: put the polishing pad center on the marker center, touch the table lightly (do not press), keep your hand on the pendant, press Enter to record.")
    for i in MARKER_IDS:
        input(f"  marker {i}: press Enter once aligned ... ")
        pose = avg_reading(rr.getActualTCPPose)
        f = avg_reading(rr.getActualTCPForce)
        pts[str(i)] = {"tcp_pose": pose.tolist(), "tcp_force": f.tolist(), "t": time.time()}
        print(f"    x={pose[0]:.4f} y={pose[1]:.4f} z={pose[2]:.4f}  Fz={f[2]:.1f} N")
    save_points(d, data)
    P = np.array([pts[str(i)]["tcp_pose"][:3] for i in MARKER_IDS])
    print("plane fit:", json.dumps(fit_plane_z(P), ensure_ascii=False))
    return 0


def cmd_survey(args, d: Path) -> int:
    rcfg = load_yaml("robot.yaml")
    rr = receive_iface(str(rcfg["hosts"]["ur3e"]))
    input("Move the arm to a survey pose that does not block the fixed camera view (do not over-rotate wrist 3), press Enter to record joint angles ... ")
    q = avg_reading(rr.getActualQ, 0.3)
    data = load_points(d)
    data["survey_q"] = q.tolist()
    save_points(d, data)
    merge_yaml(d / "zones.yaml", {"survey_q": [float(v) for v in q]})
    print("survey_q =", np.round(q, 4).tolist())
    return 0


def cmd_capture(args, d: Path) -> int:
    data = load_points(d)
    if "touch" not in data:
        print("run touch first")
        return 2
    img = grab(args.camera_url)
    found = detect_markers(img)
    miss = [i for i in MARKER_IDS if i not in found]
    if miss:
        cv2.imwrite(str(d / "capture_failed.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"markers {miss} not detected (image saved to {d / 'capture_failed.jpg'}): check whether the arm blocks the markers, and the lighting and focus")
        return 2
    from cleanproof.perception.camera import PlaneHomography

    xy = np.array([data["touch"][str(i)]["tcp_pose"][:2] for i in MARKER_IDS])
    px = np.array([found[i] for i in MARKER_IDS])
    cam = PlaneHomography.from_points(xy, px, img.shape[1], img.shape[0])
    err = cam.reprojection_error_px(xy, px)
    # 4-point homography has no redundancy, the error is always 0; check consistency with the ratio of opposite side lengths (mm per pixel should be roughly the same)
    mm_per_px = [float(np.linalg.norm(xy[a] - xy[b]) * 1000 / np.linalg.norm(px[a] - px[b])) for a, b in ((0, 1), (2, 3), (0, 2), (1, 3))]
    calib = {"fixed": cam.to_dict(), "marker_px": {str(i): found[i].tolist() for i in MARKER_IDS},
             "marker_xy": {str(i): xy[k].tolist() for k, i in enumerate(MARKER_IDS)},
             "reproj_err_px": err.tolist(), "mm_per_px_edges": mm_per_px, "t": time.time()}
    (d / "calib.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")
    cv2.imwrite(str(d / "capture_raw.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"homography written to {d / 'calib.json'}; mm/px of the four edges: {np.round(mm_per_px, 3).tolist()} (a difference over 15% means the touch points or detection are wrong)")
    return 0


def cmd_zones(args, d: Path) -> int:
    """Compute the effective config in memory and run all checks first; write files only if they pass (or with --force). Site forbidden zones are rewritten from --forbidden every time (none given = cleared),
    so that after changing the row / column count old indices do not point to a different cell."""
    from cleanproof.common.zones import ZoneGrid

    data = load_points(d)
    P = np.array([data["touch"][str(i)]["tcp_pose"][:3] for i in MARKER_IDS])
    plane = fit_plane_z(P)
    origin, size = inscribed_rect(P[:, :2], args.inset)
    zones = {"table": {"z_surface": round(plane["z_median"], 4)},
             "work_area": {"origin_xy": [round(float(v), 4) for v in origin], "size_xy": [round(float(v), 4) for v in size],
                           "rows": int(args.rows), "cols": int(args.cols)},
             "site_forbidden_zones": list(args.forbidden or [])}
    rob = {"tool": {"pad_radius_m": round(args.pad_diameter_mm / 2000.0, 4)}} if args.pad_diameter_mm else {}
    base_z = load_yaml(CONFIG_DIR / "zones.yaml")
    base_r = load_yaml(CONFIG_DIR / "robot.yaml")
    old_z = yaml.safe_load((d / "zones.yaml").read_text(encoding="utf-8")) if (d / "zones.yaml").is_file() else {}
    old_r = yaml.safe_load((d / "robot.yaml").read_text(encoding="utf-8")) if (d / "robot.yaml").is_file() else {}
    eff_z = {**base_z, **(old_z or {}), **zones, "work_area": {**base_z["work_area"], **zones["work_area"]}}
    eff_r = {**base_r, "tool": {**base_r["tool"], **((old_r or {}).get("tool") or {}), **rob.get("tool", {})}}
    cell = np.array(size) / np.array([args.rows, args.cols])
    pad_d = 2 * float(eff_r["tool"]["pad_radius_m"])
    print("table:", json.dumps(plane, ensure_ascii=False))
    print(f"work area origin={zones['work_area']['origin_xy']} size={zones['work_area']['size_xy']}, "
          f"{args.rows}x{args.cols} cells, each {cell[0] * 1000:.0f}x{cell[1] * 1000:.0f} mm; pad diameter {pad_d * 1000:.0f} mm; "
          f"site forbidden zones {zones['site_forbidden_zones'] or 'none'}")
    warn = []
    if plane["z_span_mm"] > 4.0:
        warn.append(f"height spread of the four points {plane['z_span_mm']:.1f} mm > 4 mm: table not flat or touches pressed unevenly, the approach phase may touch early / time out")
    if np.any(cell < pad_d):
        warn.append("cell is smaller than the polishing pad: wiping one cell would touch neighboring cells (the skill refuses when forbidden zones are adjacent), reduce the row / column count")
    ids = ZoneGrid(eff_z).ids
    bad = [z for z in zones["site_forbidden_zones"] if z not in ids]
    if bad:
        warn.append(f"forbidden zone indices {bad} are not in the {args.rows}x{args.cols} grid")
    warn += reach_report(eff_z, eff_r)
    print("check: " + ("passed" if not warn else "\n  - " + "\n  - ".join(warn)))
    if warn and not args.force:
        print("not written (fix and rerun; add --force if you are sure it is fine)")
        return 1
    merge_yaml(d / "zones.yaml", zones)
    if rob:
        merge_yaml(d / "robot.yaml", rob)
    print(f"wrote {d / 'zones.yaml'}" + (f", {d / 'robot.yaml'}" if rob else "") + "; after changing the zones, retake the reference and restart the skill server")
    return 0


def cmd_reference(args, d: Path) -> int:
    from cleanproof.common.zones import ZoneGrid
    from cleanproof.perception.camera import PlaneHomography
    from cleanproof.perception.rectify import rectify

    calib = json.loads((d / "calib.json").read_text(encoding="utf-8"))
    cam = PlaneHomography.from_dict(calib["fixed"])
    grid = ZoneGrid()          # CLEANPROOF_SITE is set: read the site zones.yaml
    frames = []
    for _ in range(args.frames):
        frames.append(grab(args.camera_url))
        time.sleep(0.25)
    med = np.median(np.stack(frames), axis=0).astype(np.uint8)
    ref = rectify(med, cam, grid)
    cv2.imwrite(str(d / "reference.png"), cv2.cvtColor(ref, cv2.COLOR_RGB2BGR))
    from cleanproof.perception.rectify import draw_grid

    over = med.copy()
    poly = []
    for (x, y) in [(grid.origin[0], grid.origin[1]), (grid.origin[0] + grid.size[0], grid.origin[1]),
                   (grid.origin[0] + grid.size[0], grid.origin[1] + grid.size[1]), (grid.origin[0], grid.origin[1] + grid.size[1])]:
        poly.append(cam.project(np.array([[x, y, 0.0]]))[0][0])
    cv2.polylines(over, [np.int32(poly)], True, (255, 0, 0), 3)
    cv2.imwrite(str(d / "calib_overlay.jpg"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(d / "reference_grid.jpg"), cv2.cvtColor(draw_grid(ref, grid), cv2.COLOR_RGB2BGR))
    try:
        import httpx

        from cleanproof.robot.server import auth_headers

        h = httpx.get(f"{args.camera_url.rstrip('/')}/health", timeout=5.0, headers=auth_headers()).json()
        ex = h["cameras"]["fixed"].get("exposure", {})
        if ex.get("locked"):
            merge_yaml(d / "cameras.yaml", {"cameras": {"fixed": {k: ex[k] for k in ("exposure", "gain", "white_balance") if ex.get(k) is not None}}})
            print("wrote the locked exposure settings to the site cameras.yaml:", {k: ex.get(k) for k in ("exposure", "gain", "white_balance")})
    except Exception as e:  # noqa: BLE001
        print("failed to read camera exposure settings (does not affect the reference image):", e)
    print(f"reference image {d / 'reference.png'} ({ref.shape[1]}x{ref.shape[0]}); check {d / 'calib_overlay.jpg'} and {d / 'reference_grid.jpg'} by eye")
    return 0


def verify_drift(img: np.ndarray, calib: dict[str, Any]) -> dict[str, Any]:
    found = detect_markers(img)
    out = {}
    for i, p in calib["marker_px"].items():
        out[i] = None if int(i) not in found else float(np.linalg.norm(found[int(i)] - np.array(p)))
    vals = [v for v in out.values() if v is not None]
    return {"drift_px": out, "max_px": max(vals) if vals else None, "missing": [i for i, v in out.items() if v is None]}


def cmd_verify(args, d: Path) -> int:
    calib = json.loads((d / "calib.json").read_text(encoding="utf-8"))
    r = verify_drift(grab(args.camera_url), calib)
    print(json.dumps(r, ensure_ascii=False))
    if r["max_px"] is None or r["max_px"] > args.max_px:
        print(f"camera may have been bumped (drift > {args.max_px} px or markers missing): redo capture and reference")
        return 1
    print("camera position unchanged")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="ur3e_site")
    ap.add_argument("--camera-url", default="http://127.0.0.1:8101")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("markers")
    m.add_argument("--out", default="~/aruco_markers.png")
    m.add_argument("--size-mm", type=float, default=40.0)
    sub.add_parser("touch")
    sub.add_parser("survey")
    sub.add_parser("capture")
    z = sub.add_parser("zones")
    z.add_argument("--rows", type=int, required=True)
    z.add_argument("--cols", type=int, required=True)
    z.add_argument("--inset", type=float, default=0.035, help="inward offset from the marker center (m): markers are placed 3-4 cm outside the work area")
    z.add_argument("--pad-diameter-mm", type=float, default=None, help="measured polishing pad diameter")
    z.add_argument("--forbidden", nargs="*", default=None, help="site forbidden zones (e.g. cells holding objects), the agent cannot lift them; none given = no forbidden zones")
    z.add_argument("--force", action="store_true", help="write even if the checks have warnings (make sure it is fine)")
    r = sub.add_parser("reference")
    r.add_argument("--frames", type=int, default=12)
    v = sub.add_parser("verify")
    v.add_argument("--max-px", type=float, default=4.0)
    args = ap.parse_args()
    import os

    os.environ["CLEANPROOF_SITE"] = args.site
    if args.cmd == "markers":
        return cmd_markers(args)
    d = site_paths(args.site)
    return {"touch": cmd_touch, "survey": cmd_survey, "capture": cmd_capture, "zones": cmd_zones,
            "reference": cmd_reference, "verify": cmd_verify}[args.cmd](args, d)


if __name__ == "__main__":
    raise SystemExit(main())
