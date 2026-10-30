"""CleanProof operator software (the "our software" segment of the video; it must be actually deployable, with practical settings, tuning, controls and logs, not a dashboard).

Page / (static/operator.html, in English): first-run wizard (Robot → Connect → Models → Site) → main UI Jobs · Robots · Sites · Reports · Settings · Logs.
All data comes from real configs and run records; every button on the page is wired to a real endpoint:
- First-run wizard (align with a Bambu Studio style first-time setup): /api/setup/*. Connect / Models run the deploy checks
  (cleanproof/ui/deploy_checks.py; with --deploy-record the saved real check is returned), Models accepts an API key (written to secrets.env, 0600, only OK / MISSING is returned),
  Site reads the site calibration (zone grid + clean reference; calibration is done automatically by scripts/calibrate_site.py, people do not draw zones), Finish writes ~/.config/cleanproof/setup.json.
- Deploy checks: GET /api/deploy/checks checks device by device on site; ?recorded=1 returns the saved check (--deploy-record).
- Jobs: POST /api/jobs/start runs a job in a local background thread (the site job_scenario; policy / verifier per command line),
  /api/jobs/{approve,reject,pause,resume,stop,estop} control the orchestration loop directly (cleanproof/ui/session.py → agent/loop.py).
  When recording the video, --job selects an already recorded run: GET /api/jobs/state?t= gives the state of that job at time t (deterministic frame by frame).
- Requests: requests handed to human review during a run (summary.human_requests); POST /api/requests/resolve records to <run directory>/requests.json.
- Robots: all software adapters in configs/platforms.yaml + the robots registered at each site; --robot-take supplies the frames of each robot latest job.
- Sites: configs/site/*/site.yaml with the zone geometry and clean reference image of the site.
- Settings (build it out like a real product; every item shown can really be saved and takes effect, or is honestly marked Not configured):
  Jobs: max wipes per cell, per-task step limit, re-check all zones after every wipe, manual plan approval, Nemotron 3 Ultra audit, no-wipe zones → configs/site/<site>/operator.yaml,
  taking effect when the next job starts; Safety & force: max wipe force per surface, force step for re-wipes → site force_table.yaml, enforced by the guardrails from the next job;
  Vision / Cloud models / Robot (models, addresses, connection status; keys are reported only as OK / MISSING) · spending cap (site models.yaml, can only be lowered) ·
  Notifications (human-handoff webhook, empty = Not configured) · Data & logs · About (version, license, photo credits).
- Logs: model call log (runs/llm_calls.jsonl or --calls-log) and events of the current task.
- Reports: list of run directories and cleaning reports (cleanproof.report).

  python -m cleanproof.ui.operator --site home_dining [--policy nemotron --verifier vlm]        # live on this machine
  python -m cleanproof.ui.operator --site home_dining --job runs/submission_v12/take_H5_dining_permanent_a1/H5_dining_permanent_spot_arm \\
      --robot-take rizon4=... --deploy-record runs/submission_v12/deploy_checks.json --calls-log ... --ledger ...  # for recording the video
Binds to 127.0.0.1 only by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from ..common.config import REPO_ROOT, current_site, load_yaml, runs_root, site_root, use_site, user_config_dir
from . import calibration
from .fleet import SURFACE_LABEL, Take, _find_run_dir, report_row
from .siteconfig import ConfigError, check_site, require_valid
from .taskparse import parse_task

STATIC = Path(__file__).parent / "static"


# files in the local user config directory ($CLEANPROOF_CONFIG_DIR, or ~/.config/cleanproof if unset): the path is computed from the environment variable each time it is used.
# the four module attributes below default to None (= use the config directory); tests can replace them with temporary files directly.
SETUP_FILE: Optional[Path] = None                                          # written here after the first-run wizard finishes
SECRETS_FILE: Optional[Path] = None
ROBOT_HOST_FILE: Optional[Path] = None                                     # written by scripts/install/install_edge.sh --pair <host>
APP_FILE: Optional[Path] = None                                            # machine-level options on the settings page (planning / inspection mode, robot computer address)


def _user_file(override: Optional[Path], name: str) -> Path:
    return Path(override) if override is not None else user_config_dir() / name


def setup_file() -> Path:
    return _user_file(SETUP_FILE, "setup.json")


def secrets_file() -> Path:
    return _user_file(SECRETS_FILE, "secrets.env")


def robot_host_file() -> Path:
    return _user_file(ROBOT_HOST_FILE, "robot_host")


def app_file() -> Path:
    return _user_file(APP_FILE, "app.json")


def setup_site_choice() -> Optional[str]:
    """Site chosen at wizard Finish (setup.json); not used if the site no longer exists."""
    try:
        sid = str(json.loads(setup_file().read_text(encoding="utf-8")).get("site") or "")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    return sid if sid and (SITE_DIR / sid / "site.yaml").is_file() else None
PHOTO_DIR = REPO_ROOT / "assets" / "robots_photos" / "cutout"             # original product images for the Robots page (licensed photos cut out on white; white-background renders where none exist)
JOB_DEFAULTS: dict[str, Any] = {"max_passes_per_zone": 3, "max_steps": 30, "recheck_all": True, "audit": True, "no_wipe_zones": []}
SITE_TYPES = ["Home", "Office", "Factory", "Car wash", "Outdoor", "Public space"]
# venue-type buttons in the wizard (singular, exactly these in the video) <-> type in site.yaml (the category shown on the Sites page and in the site dropdown, plural)
SITE_TYPE_STORED = {"Home": "Homes", "Office": "Offices", "Factory": "Factories", "Car wash": "Car washes",
                    "Outdoor": "Outdoor", "Public space": "Public buildings"}
DEFAULT_SKILL_URL, DEFAULT_CAMERA_URL = "http://127.0.0.1:8100", "http://127.0.0.1:8101"   # two services mapped to this machine through the encrypted tunnel
FONT_DIR = REPO_ROOT / "assets" / "fonts"
SITE_DIR = site_root()
APP_VERSION = "1.0"


# ---------------------------------------------------------------- sites
def config_errors() -> list[dict[str, Any]]:
    """Problems in all site configs (file, line, reason); empty = none."""
    out: list[dict[str, Any]] = []
    for d in sorted(SITE_DIR.iterdir()):
        if (d / "site.yaml").is_file() or (d / "zones.yaml").is_file():
            out += [{"site": d.name, **p} for p in check_site(d.name, d)]
    return out


def ensure_site_ok(site: str) -> None:
    """Raise ConfigError if the site config has problems (the UI shows the file name and line); a missing site is handled by the caller."""
    d = SITE_DIR / site
    if d.is_dir():
        require_valid(site, d)


def list_sites() -> list[dict[str, Any]]:
    out = []
    for d in sorted(SITE_DIR.iterdir()):
        f = d / "site.yaml"
        if not f.is_file() or check_site(d.name, d):         # a site whose config is broken is not listed (config_errors says which file and line), so the whole UI does not return 500
            continue
        s = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        z = load_yaml("zones.yaml", site=d.name)
        wa = z["work_area"]
        out.append({"id": d.name, "name": s.get("name", d.name), "type": s.get("type", ""), "surface": s.get("surface", ""),
                    "surface_label": SURFACE_LABEL.get(s.get("surface", ""), str(s.get("surface", "")).replace("_", " ")),
                    "robot": s.get("robot", ""), "scenarios": list(s.get("scenarios") or []), "job_scenario": s.get("job_scenario"),
                    "grid": {"rows": int(wa["rows"]), "cols": int(wa["cols"]), "size_m": [float(v) for v in wa["size_xy"]]},
                    "calibrated": _calibrated(d)})
    return out


def _calibrated(d: Path) -> bool:
    """Site calibrated = the site zones.yaml contains a measured workspace. A site calibrated on site (with calib_points.json) also needs its clean reference already captured:
    for guided in-app calibration the reference step must be done, for command-line calibration reference.png must exist; halfway through, "clean reference saved" is not shown."""
    f = d / "zones.yaml"
    if not f.is_file():
        return False
    try:
        if "work_area" not in (yaml.safe_load(f.read_text(encoding="utf-8")) or {}):
            return False
    except yaml.YAMLError:
        return False
    if _has_calib_points(d):
        if not (d / "reference.png").is_file():
            return False
        pts = calibration.load_points(d)
        if pts.get("app"):
            steps = {x["id"]: x["done"] for x in calibration.guided_state(d.name, d)["steps"]}
            return bool(steps["reference"])
    return True


def _has_calib_points(d: Path) -> bool:
    """A site calibrated on site (guided in the software, or scripts/calibrate_site.py): has calib_points.json."""
    return (d / "calib_points.json").is_file()


def sim_templates() -> list[dict[str, Any]]:
    """Simulation scenes selectable for a new site in a simulated deployment: sites that have a job template (job_scenario)."""
    return [{"id": s["id"], "name": s["name"], "robot": s["robot"]} for s in list_sites() if s.get("job_scenario")]


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    return s[:40].strip("_") or "site"


_SITE_CREATE_LOCK = threading.Lock()


def create_site(name: str, type_: str, robot: str, sim: bool, template: Optional[str] = None, current: Optional[str] = None,
                surface: Optional[str] = None) -> dict[str, Any]:
    with _SITE_CREATE_LOCK:                                   # clicked twice at once / created from two pages at once: the second gets a duplicate-name 400
        return _create_site(name, type_, robot, sim, template, current, surface)


def _create_site(name: str, type_: str, robot: str, sim: bool, template: Optional[str] = None, current: Optional[str] = None,
                 surface: Optional[str] = None) -> dict[str, Any]:
    """Create a new site configs/site/<slug>/site.yaml (name, category, robot, tabletop material; in a simulated deployment it carries the simulation scene of the template site).
    Calibration (zone grid, clean reference) is done afterwards in the wizard Site step. Returns this entry as it appears in list_sites()."""
    from ..robot.platforms import load_platforms
    name = str(name or "").strip()
    if not name or len(name) > 80 or "\n" in name:
        raise ValueError("Give the site a name (up to 80 characters)")
    t = str(type_ or "").strip()
    stored = {v: v for v in SITE_TYPE_STORED.values()}
    if t in SITE_TYPE_STORED:
        t = SITE_TYPE_STORED[t]
    elif t not in stored:
        raise ValueError(f"Unknown site type {type_!r}")
    if robot not in load_platforms():
        raise ValueError(f"Unknown robot {robot!r}")
    sites = list_sites()
    by_id = {x["id"]: x for x in sites}
    if any(str(x["name"]).strip().casefold() == name.casefold() for x in sites):
        raise ValueError(f"A site named {name!r} already exists")
    tpl = None
    if sim:
        tpls = [x for x in sites if x.get("job_scenario")]
        if template:
            tpl = next((x for x in tpls if x["id"] == template), None)
            if tpl is None:
                raise ValueError(f"Unknown simulation template {template!r}")
        else:
            tpl = next((x for x in tpls if x["robot"] == robot), None) or next((x for x in tpls if x["id"] == current), None) \
                or (tpls[0] if tpls else None)
        if tpl is None:
            raise ValueError("No simulated scene to base the new site on")
    surfaces = set(load_yaml("force_table.yaml", site="")["surfaces"])
    surf = str(surface or (tpl or by_id.get(current or "") or {}).get("surface") or "sealed_wood")
    if surf not in surfaces:
        raise ValueError(f"Unknown surface {surf!r}")
    base = slugify(name)
    sid, n = base, 2
    while (SITE_DIR / sid).exists():
        sid, n = f"{base}_{n}", n + 1
    d = SITE_DIR / sid
    lines = [APP_SITE_HEADER + " (Sites / wizard). The zone grid and clean reference are written to the same directory after calibration in the wizard Site step.",
             f"name: {_yaml_scalar(name)}", f"type: {_yaml_scalar(t)}", f"surface: {surf}", f"robot: {robot}"]
    if tpl is not None:                                       # borrow only the simulation scene of the template site; scenarios (the template own recordings) are not copied
        lines += [f"job_scenario: {tpl['job_scenario']}        # simulation scene taken from site {tpl['id']}"]
    try:
        d.mkdir(parents=True)
    except FileExistsError:                                   # a site with the same name was created at the same time
        raise ValueError(f"A site named {name!r} is being created already") from None
    (d / "site.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return next(x for x in list_sites() if x["id"] == sid)


def _yaml_scalar(v: str) -> str:
    out = yaml.safe_dump(v, allow_unicode=True, default_flow_style=True).strip()
    return out[:-3].strip() if out.endswith("...") else out


def update_site(site: str, fields: dict[str, str]) -> None:
    """Wizard Finish: write the robot, venue type and name into configs/site/<site>/site.yaml. Empty values are not written.
    The venue type is validated against the wizard buttons (SITE_TYPES) and stored as the site category (SITE_TYPE_STORED, the word shown on the Sites page).
    Only the lines holding these top-level keys are changed (appended if absent); the rest of the file (comments, list style, trailing comments) is kept as is."""
    f = _site_file(site, "site.yaml")
    txt = f.read_text(encoding="utf-8")
    data = yaml.safe_load(txt) or {}
    if fields.get("robot"):
        from ..robot.platforms import load_platforms
        if fields["robot"] not in load_platforms():
            raise ValueError(f"Unknown robot {fields['robot']!r}")
    fields = dict(fields)
    if fields.get("type"):
        if fields["type"] not in SITE_TYPES:
            raise ValueError(f"Unknown site type {fields['type']!r}")
        fields["type"] = SITE_TYPE_STORED[fields["type"]]
    changed = {k: str(v)[:80] for k, v in fields.items() if v and data.get(k) != v}
    if not changed:
        return
    lines = txt.splitlines()
    for k, v in changed.items():
        val = yaml.safe_dump(v, allow_unicode=True, default_flow_style=True).strip()
        if val.endswith("..."):                                  # document end marker that safe_dump adds to scalars
            val = val[:-3].strip()
        i = next((n for n, ln in enumerate(lines) if ln.startswith(f"{k}:")), None)
        if i is None:
            lines.append(f"{k}: {val}")
        else:
            m = re.match(rf"{re.escape(k)}:\s*(?:'[^']*'|\"[^\"]*\"|[^#]*?)(\s+#.*)?$", lines[i])
            lines[i] = f"{k}: {val}" + ((m.group(1) or "") if m else "")
    out = "\n".join(lines) + "\n"
    try:
        back = yaml.safe_load(out) or {}
    except yaml.YAMLError:
        back = {}
    if {k: back.get(k) for k in changed} != changed:     # e.g. a multi-line value: do not force the edit, report it to the wizard
        raise ValueError(f"Cannot update {', '.join(changed)} in site.yaml automatically; edit it by hand")
    f.write_text(out, encoding="utf-8")


def site_of_scenario(scenario: str) -> Optional[str]:
    """Which site this simulation scene belongs to: first the site that registered its recording (scenarios), then the site that uses it as a job template (job_scenario);
    on a tie the original site wins (sites created in the software only borrow the simulation scene of the template site, so old jobs must not be renamed because of it)."""
    sites = list_sites()
    for match in (lambda x: scenario in x["scenarios"], lambda x: x.get("job_scenario") == scenario):
        hits = [x for x in sites if match(x)]
        if hits:
            orig = [x for x in hits if not _created_in_app(x["id"])]
            return (orig or hits)[0]["id"]
    return None


APP_SITE_HEADER = "# site created in the software"


def _created_in_app(site: str) -> bool:
    try:
        return (SITE_DIR / site / "site.yaml").read_text(encoding="utf-8").startswith(APP_SITE_HEADER)
    except OSError:
        return False


def _site_file(site: str, name: str) -> Path:
    d = (SITE_DIR / site).resolve()
    if SITE_DIR.resolve() not in d.parents or not (d / "site.yaml").is_file():
        raise KeyError(f"Unknown site {site!r}")
    return d / name


def _read_op(site: str) -> dict[str, Any]:
    f = _site_file(site, "operator.yaml")
    return (yaml.safe_load(f.read_text(encoding="utf-8")) or {}) if f.is_file() else {}


def job_options(site: str) -> dict[str, Any]:
    """The Jobs group on the settings page (site operator.yaml, defaults from JOB_DEFAULTS): applied to the task spec and loop options when a job starts."""
    return {**JOB_DEFAULTS, **(_read_op(site).get("jobs") or {})}


def read_settings(site: str) -> dict[str, Any]:
    """Settings page: max wipe force per surface of this site (adjustable range = [highest starting force of that surface, global cap]), re-wipe force step, task options, plan approval, notifications, models and budget."""
    import copy

    from ..agent.force_table import clamp_site_limits
    ft = load_yaml("force_table.yaml", site=site)
    base = load_yaml("force_table.yaml", site="")
    raw = copy.deepcopy(ft)
    clamp_site_limits(ft, base)                          # same rule as when the force table is loaded for a job: a hand-edited out-of-range value shows as the value actually in effect (after clamping)
    hard = float(ft["hard_max_n"])
    notes: list[str] = []
    if abs(float(raw["hard_max_n"]) - hard) > 1e-9:
        notes.append(f"The overall force limit is clamped to {hard:g} N (the file had {float(raw['hard_max_n']):g} N).")
    surfaces = []
    for k, v in ft["surfaces"].items():
        lo = max(float(row[k]) for row in ft["forces"].values())
        item = {"surface": k, "label": v.get("label", k), "max_n": float(v["max_n"]), "default_max_n": float(base["surfaces"][k]["max_n"]),
                "min_allowed_n": lo, "max_allowed_n": hard}
        was = float(raw["surfaces"][k]["max_n"])
        if abs(was - float(v["max_n"])) > 1e-9:
            item["clamped_from"] = was
            item["clamp_note"] = f"Clamped to {float(v['max_n']):g} N (file had {was:g} N)"
            notes.append(f"{item['label']}: {item['clamp_note']}")
        surfaces.append(item)
    esc_note = None
    if abs(float(raw["escalate_step_n"]) - float(ft["escalate_step_n"])) > 1e-9:
        esc_note = f"Clamped to {float(ft['escalate_step_n']):g} N (file had {float(raw['escalate_step_n']):g} N)"
        notes.append("Extra force on each re-wipe: " + esc_note)
    op = _read_op(site)
    m = load_yaml("models.yaml", site=site)
    m0 = load_yaml("models.yaml", site="")
    n_entries = sum(len(row) for row in ft["forces"].values())
    zones = load_yaml("zones.yaml", site=site)["work_area"]
    return {"site": site, "surfaces": surfaces, "hard_max_n": hard, "min_n": float(ft["min_n"]), "escalate_step_n": float(ft["escalate_step_n"]),
            "escalate": {"value": float(ft["escalate_step_n"]), "default": float(base["escalate_step_n"]), "min": 1.0, "max": 5.0, "clamp_note": esc_note},
            "limit_notes": notes,
            "require_plan_approval": bool(op.get("require_plan_approval", False)),
            "jobs": job_options(site), "grid": {"rows": int(zones["rows"]), "cols": int(zones["cols"])},
            "notifications": {"staff_check_webhook": str((op.get("notifications") or {}).get("staff_check_webhook") or "")},
            "force_db": {"entries": n_entries, "stains": len(ft["forces"]), "surfaces": len(ft["surfaces"]),
                         "site_override": _site_file(site, "force_table.yaml").is_file()},
            "models": {"verifier": {"model": m["roles"]["verifier"]["model"], "where": "GPU workstation",
                                    "endpoint": m["backends"][m["roles"]["verifier"]["backend"]]["base_url"]},
                       "planner": {"model": m["roles"]["decision"]["model"], "where": "Nebius Token Factory"},
                       "auditor": {"model": m["roles"]["audit"]["model"], "where": "Nebius Token Factory"},
                       "cloud_url": m["backends"]["tokenfactory"]["base_url"]},
            "budget": {"limit_usd": float((m.get("budget") or {}).get("usd_limit", 40.0)),
                       "max_usd": float((m0.get("budget") or {}).get("usd_limit", 40.0))}}


def _zone_ids(rows: int, cols: int) -> set[str]:
    return {f"{chr(65 + r)}{c + 1}" for r in range(rows) for c in range(cols)}


def zones_in_task(text: str, rows: int, cols: int) -> list[str]:
    """The zones to wipe in the task the operator typed ("Wipe B2 and C3 only", "A1-A3", "everything except C2"); nothing named = empty list (the whole surface).
    The rules for exclusions, ranges and "the rest" are in taskparse.parse_task; this function only takes the zones to wipe."""
    return parse_task(text, rows, cols)["zones"]


def read_app() -> dict[str, Any]:
    try:
        return json.loads(app_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_app(upd: dict[str, Any]) -> dict[str, Any]:
    """Merge-write app.json; keys with value None are removed (back to launch arguments / defaults)."""
    cur = {k: v for k, v in {**read_app(), **upd}.items() if v is not None}
    f = app_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    return cur


def _is_loopback(host: str) -> bool:
    import ipaddress
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _tf_reason(detail: str) -> str:
    """Reason a Token Factory check failed (shown to the operator; contains no key and no address from the raw exception)."""
    key = os.environ.get("NEBIUS_API_KEY") or ""
    d = detail.replace(key, "***") if key else detail
    if "401" in d or "403" in d:
        return "API key rejected by Nebius Token Factory · check the key"
    if any(x in d for x in ("ConnectError", "ConnectTimeout", "ReadTimeout", "NetworkError", "Timeout")):
        return "Nebius Token Factory not reachable · check the internet connection"
    if d.startswith("API key OK"):                        # connected, but the model used for planning / audit is not available
        return d
    return f"Nebius Token Factory check failed ({d.split(':', 1)[0][:40] or 'error'})"


def _cap(e: Any) -> str:
    """Error message shown to the operator: strip the KeyError quotes and capitalize the first letter."""
    m = str(e).strip().strip("'\"")
    return m[:1].upper() + m[1:]


def _check_url(u: str, robot_host: Optional[str] = None) -> str:
    """Address of the robot computer services. Requests carry the shared token, so only these are accepted: a local loopback address (127.0.0.0/8, localhost, ::1, through the encrypted tunnel),
    or https://<hostname of the paired robot computer>. Everything else is rejected, so the token is never sent to an arbitrary host."""
    from urllib.parse import urlparse
    u = str(u).strip().rstrip("/")
    try:
        p = urlparse(u)
        port_ok = p.port is None or 0 < p.port < 65536
    except ValueError:
        port_ok = False
    if p.scheme not in ("http", "https") or not p.hostname or not port_ok or any(c.isspace() for c in u) or len(u) > 300 \
            or p.username or p.password or (p.path not in ("", "/")) or p.query or p.fragment:
        raise ValueError(f"Not a service address: {u!r} (use http://127.0.0.1:<port>)")
    host = p.hostname
    if _is_loopback(host) or (p.scheme == "https" and robot_host and host.lower() == robot_host.strip().lower()):
        return u
    raise ValueError("Robot services are reached through the encrypted tunnel on this computer (127.0.0.1), "
                     + (f"or over https at the paired robot computer ({robot_host})" if robot_host else "set up by pairing")
                     + f"; {host} is not allowed")


def write_settings(site: str, body: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    """Save settings. Anything outside the adjustable range is rejected (no silent clamping); values equal to the default write no override.
    Every item is validated and the files to write are computed first, and writing happens only if all are valid (one invalid item rejects the whole request and nothing is written). dry_run: validate only."""
    from ..agent.force_table import ForceTable, reload_force_table
    cur = read_settings(site)
    lim = {s["surface"]: s for s in cur["surfaces"]}
    writes: list[tuple[Path, Optional[str]]] = []          # (file, new content; None = delete the override file and go back to the default)
    over: dict[str, Any] = {}
    f_ft = _site_file(site, "force_table.yaml")
    if f_ft.is_file():
        over = yaml.safe_load(f_ft.read_text(encoding="utf-8")) or {}
    ft_touched = False
    for k, v in (body.get("max_n") or {}).items():
        if k not in lim:
            raise ValueError(f"Unknown surface {k!r}")
        v = float(v)
        if not lim[k]["min_allowed_n"] - 1e-6 <= v <= lim[k]["max_allowed_n"] + 1e-6:
            raise ValueError(f"{lim[k]['label']}: {v:g} N outside {lim[k]['min_allowed_n']:g}–{lim[k]['max_allowed_n']:g} N")
        surf = over.setdefault("surfaces", {})
        if abs(v - lim[k]["default_max_n"]) < 1e-6:
            surf.pop(k, None)
        else:
            surf[k] = {"max_n": v}
        ft_touched = True
    if over.get("surfaces") == {}:
        over.pop("surfaces")
    if "escalate_step_n" in body:
        v = float(body["escalate_step_n"])
        e = cur["escalate"]
        if not e["min"] - 1e-6 <= v <= e["max"] + 1e-6:
            raise ValueError(f"Extra force per re-wipe: {v:g} N outside {e['min']:g}–{e['max']:g} N")
        if abs(v - e["default"]) < 1e-6:
            over.pop("escalate_step_n", None)
        else:
            over["escalate_step_n"] = v
        ft_touched = True
    if ft_touched:
        merged = load_yaml("force_table.yaml", site="")
        for k, v in (over.get("surfaces") or {}).items():
            merged["surfaces"][k] = {**merged["surfaces"][k], **v}
        if "escalate_step_n" in over:
            merged["escalate_step_n"] = over["escalate_step_n"]
        ForceTable.from_cfg(merged)                           # write only if validation passes
        if over:
            writes.append((f_ft, "# written by the software settings page (site override of the force table: max wipe force per surface, re-wipe force step). The guardrails approve / reject by this from the next job.\n"
                           + yaml.safe_dump(over, allow_unicode=True, sort_keys=False)))
        elif f_ft.is_file():
            writes.append((f_ft, None))
    op = _read_op(site)
    changed = False
    if "require_plan_approval" in body:
        op["require_plan_approval"] = bool(body["require_plan_approval"])
        changed = True
    if "jobs" in body:
        jo = {**job_options(site)}
        for k, v in (body.get("jobs") or {}).items():
            if k == "max_passes_per_zone":
                v = int(v)
                if not 1 <= v <= 6:
                    raise ValueError("Wipes per zone must be 1–6")
            elif k == "max_steps":
                v = int(v)
                if not 5 <= v <= 200:
                    raise ValueError("Steps per job must be 5–200")
            elif k in ("recheck_all", "audit"):
                v = bool(v)
            elif k == "no_wipe_zones":
                ok = _zone_ids(cur["grid"]["rows"], cur["grid"]["cols"])
                v = sorted({str(z).upper() for z in v})
                bad = [z for z in v if z not in ok]
                if bad:
                    raise ValueError(f"Unknown zones {bad}")
            else:
                raise ValueError(f"Unknown job setting {k!r}")
            jo[k] = v
        op["jobs"] = {k: v for k, v in jo.items() if v != JOB_DEFAULTS.get(k)}
        if not op["jobs"]:
            op.pop("jobs")
        changed = True
    if "staff_check_webhook" in body:
        url = str(body["staff_check_webhook"] or "").strip()
        if url and not url.startswith(("https://", "http://")):
            raise ValueError("Webhook must be an http(s) URL")
        if url:
            op["notifications"] = {"staff_check_webhook": url}
        else:
            op.pop("notifications", None)
        changed = True
    if changed:
        f_op = _site_file(site, "operator.yaml")
        if op:
            writes.append((f_op, "# written by the software settings page: task options, plan approval and notifications of this site. Takes effect when the next job starts.\n"
                           + yaml.safe_dump(op, allow_unicode=True, sort_keys=False)))
        elif f_op.is_file():
            writes.append((f_op, None))
    if "budget_usd" in body:                                    # can only be lowered (raising the global cap needs the user consent first, llm_client.py)
        v = float(body["budget_usd"])
        cap = cur["budget"]["max_usd"]
        if not 1.0 <= v <= cap + 1e-9:
            raise ValueError(f"Spending limit must be between $1 and ${cap:g}")
        f_m = _site_file(site, "models.yaml")
        if abs(v - cap) < 1e-9:
            writes.append((f_m, None))
        else:
            writes.append((f_m, "# written by the software settings page: Token Factory spending cap of this site (can only be below the global configs/models.yaml).\n"
                           + yaml.safe_dump({"budget": {"usd_limit": v}})))
    if dry_run:
        return cur
    for f, content in writes:
        if content is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(content, encoding="utf-8")
    reload_force_table()
    return read_settings(site)


def _valid_key(key: str) -> str:
    key = str(key).strip()
    if not key or len(key) > 512 or any(c.isspace() for c in key) or not key.isprintable():
        raise ValueError("Invalid API key")
    return key


def _api_key_set() -> bool:
    """Whether the Token Factory key exists (presence only, never echoed; the key file is read from the user config directory)."""
    from ..common.config import load_secrets
    load_secrets()
    if os.environ.get("NEBIUS_API_KEY"):
        return True
    try:
        return any(ln.startswith("NEBIUS_API_KEY=") and ln.split("=", 1)[1].strip()
                   for ln in secrets_file().read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


def save_api_key(key: str, path: Optional[Path] = None) -> None:
    """Wizard Models step / settings page: write NEBIUS_API_KEY into secrets.env in the user config directory (0600), replacing the old value; not echoed, not logged.
    Atomic write: write a temporary file (0600) in the same directory, then rename, so a crash halfway does not lose the other keys in the same file."""
    path = path or secrets_file()
    key = _valid_key(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    lines = [ln for ln in lines if not ln.startswith("NEBIUS_API_KEY=")] + [f"NEBIUS_API_KEY={key}"]
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    os.chmod(path, 0o600)
    os.environ["NEBIUS_API_KEY"] = key


def verify_then_save_key(key: str, allow_unverified: bool = False) -> Optional[str]:
    """Verify the candidate key in memory first and write it only if it passes; on failure the old key is kept. Returns None = saved, otherwise the reason for the operator (nothing was written).
    allow_unverified: only for recorded-video replays (--deploy-record, where the check results are stored): when verification is impossible (cannot connect) the key is still saved, but a key that is explicitly rejected is never written."""
    from .deploy_checks import probe_api_key
    ok, why = probe_api_key(key)
    if ok is False:
        return "Token Factory rejected the API key. The previous key was kept. Check the key and try again."
    if ok is None and not allow_unverified:
        return f"The key could not be checked ({why}). Nothing was saved. Check the internet connection and try again."
    save_api_key(key)
    return None


def service_status(deploy_record: Optional[Path], sim_robot: bool = False, link: Optional[dict[str, Any]] = None, robot: str = "") -> dict[str, Any]:
    """Connection status of each service on the settings page: if a saved deploy check exists use that real check; otherwise only look at whether the local key / token files exist (no network, no echo).
    Simulated deployment (desktop edition): the robot computer is the built-in local simulation, and the skill server item checks whether MuJoCo can be imported (the Robot connected badge in the title bar follows it)."""
    if deploy_record is not None and deploy_record.is_file():
        rec = json.loads(deploy_record.read_text(encoding="utf-8"))
        return {"source": "check", "ts": rec.get("ts"), "checks": {c["id"]: {"ok": bool(c.get("ok")), "detail": c.get("detail", "")}
                                                                   for c in rec.get("checks", [])}}
    from ..common.config import load_secrets
    from ..robot.server import read_token
    load_secrets()
    try:
        tok = read_token() is not None
    except PermissionError:
        tok = False
    checks = {"tokenfactory": {"ok": bool(os.environ.get("NEBIUS_API_KEY")),
                               "detail": "API key OK" if os.environ.get("NEBIUS_API_KEY") else "API key MISSING"},
              "token": {"ok": tok, "detail": "shared token OK" if tok else "shared token MISSING"}}
    if sim_robot:
        checks["skill_server"] = _sim_overall(robot)
    elif link is not None:                                    # paired robot computer: the title bar follows the live health check of the skill server
        checks["skill_server"] = {"ok": bool(link.get("ok")), "detail": link.get("detail", "")}
        if link.get("camera") is not None:
            checks["camera"] = {"ok": bool(link["camera"]), "detail": "camera service reachable" if link["camera"] else "camera service not reachable"}
    return {"source": "local", "ts": None, "checks": checks}


def notify_staff(url: str, payload: dict[str, Any]) -> None:
    """Send a webhook on human handoff (Settings > Notifications; not sent if no address is set). Background thread; a failure is only logged and does not affect the task."""
    import httpx

    def _post() -> None:
        try:
            httpx.post(url, json=payload, timeout=5.0)
        except Exception as e:  # noqa: BLE001
            print(f"[operator] staff-check webhook failed: {type(e).__name__}")
    threading.Thread(target=_post, daemon=True).start()


# ---------------------------------------------------------------- force database screening (the Jobs page animation "look up by stain x surface in the site database")
def force_lookup(site: str, stain: str, surface: str) -> dict[str, Any]:
    """Return the screening process: all entries → filter by surface first → then hit by stain. Only entry names and the force of the hit are given, not the values of the whole table (the video does not show the force table)."""
    ft = load_yaml("force_table.yaml", site=site)
    stains = {k: v.get("label", k) for k, v in ft["stains"].items()}
    surfs = {k: v.get("label", k) for k, v in ft["surfaces"].items()}
    entries = [{"stain": s, "surface": f, "label": f"{stains[s]} · {surfs[f]}"} for s in ft["forces"] for f in ft["surfaces"]]
    on_surface = [e for e in entries if e["surface"] == surface]
    key = stain if stain in ft["forces"] else "wet_spill"
    hit = next(e for e in on_surface if e["stain"] == key)
    return {"entries": entries, "on_surface": on_surface, "hit": {**hit, "force_n": float(ft["forces"][key][surface])},
            "unknown_fallback": key != stain}


# ---------------------------------------------------------------- logs, budget
def call_failure(r: dict[str, Any]) -> Optional[str]:
    """A failed model call (401, cannot connect, over budget, missing key) shows "failed · 401" in the latency column of Logs; a successful call returns None."""
    err = r.get("error")
    if not err:
        return None
    status, kind, e = r.get("http_status"), str(r.get("error_kind") or ""), str(err)
    if isinstance(status, int):
        return f"failed · {status}"
    m = re.search(r"Error code: (\d{3})", e)
    if m:
        return f"failed · {m.group(1)}"
    if kind == "budget" or "BudgetExceeded" in e:
        return "failed · over budget"
    if kind == "no_key" or "MissingApiKey" in e or "missing environment variable" in e:
        return "failed · no key"
    if "Connection" in e or "Timeout" in e or kind == "unreachable":
        return "failed · no connection"
    return "failed"


def read_calls(path: Optional[Path], limit: int = 400, model: Optional[str] = None, run_id: Optional[str] = None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if model and model.lower() not in str(r.get("model", "")).lower():
            continue
        if run_id and run_id not in json.dumps(r.get("meta") or r.get("tag") or ""):
            pass
        rows.append({"ts": r.get("ts") or r.get("time"), "role": r.get("role"), "model": r.get("model"), "backend": r.get("backend"),
                     "latency_s": r.get("latency_s"), "tokens_in": r.get("prompt_tokens") or (r.get("usage") or {}).get("prompt_tokens"),
                     "tokens_out": r.get("completion_tokens") or (r.get("usage") or {}).get("completion_tokens"),
                     "cost_usd": r.get("cost_usd"), "ok": r.get("schema_ok", r.get("ok")), "request_id": r.get("request_id"), "tag": r.get("tag"),
                     "failed": call_failure(r)})
    return rows[-limit:]


def read_ledger(path: Optional[Path]) -> Optional[float]:
    try:
        return float(json.loads(Path(path).read_text(encoding="utf-8"))["total_usd"]) if path else None
    except Exception:  # noqa: BLE001
        return None


def scan_interrupted(runs_dir: Path) -> list[Path]:
    """Scan at startup: a run directory of a job started by the software (reference_meta.json has a site) that has neither summary.json nor task_end = an interrupted job."""
    out: list[tuple[float, Path]] = []
    try:
        dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    except OSError:
        return []
    for d in dirs:
        ev, meta = d / "events.jsonl", d / "reference_meta.json"
        if not ev.is_file() or (d / "summary.json").is_file() or not meta.is_file():
            continue
        try:
            if not json.loads(meta.read_text(encoding="utf-8")).get("site"):
                continue
            tail = ev.read_text(encoding="utf-8").strip().splitlines()[-1:]
            if tail and json.loads(tail[0]).get("kind") == "task_end":
                continue
            out.append((ev.stat().st_mtime, d))
        except (OSError, ValueError, AttributeError):
            continue
    return [d for _, d in sorted(out)]


# ---------------------------------------------------------------- application
class OperatorState:
    def __init__(self, site: str, job: Optional[Take] = None, robot_takes: Optional[dict[str, Take]] = None,
                 deploy_record: Optional[Path] = None, calls_log: Optional[Path] = None, ledger: Optional[Path] = None,
                 report_runs: Optional[list[Path]] = None, policy: str = "rules", verifier: str = "cv", source: str = "topdown",
                 runs_dir: Optional[Path] = None, client: Any = None, robot_host: Optional[str] = None, sim_robot: bool = False,
                 pace: float = 0.0, skill_url: Optional[str] = None, camera_url: Optional[str] = None):
        self.site = site
        self.pace = float(pace)                           # simulation pace of live tasks (0 = run as fast as possible; desktop edition 3 = triple real time, with smooth motion in the camera)
        self.sim_robot = sim_robot                        # desktop-edition simulated deployment: robot computer = built-in local MuJoCo simulation (tasks run in simulation anyway)
        self.job = job
        self.robot_takes = robot_takes or {}
        self.deploy_record = deploy_record
        self.calls_log = calls_log or (runs_root() / "llm_calls.jsonl")
        self.ledger = ledger or Path("~/.local/state/cleanproof/spend.json").expanduser()
        self.report_runs = list(report_runs or [])
        self.policy, self.verifier, self.source = policy, verifier, source
        self.runs_dir = runs_dir or runs_root()
        # every job started from the software (live mode): recorded in a manifest under the run directory, listed by Reports / Today (kept across job changes and software restarts)
        self.history_file = self.runs_dir / "software_jobs.txt"
        self.live_history: list[Path] = []
        if job is None and self.history_file.is_file():                 # not read when recording the video (--job): the visuals use only the recorded jobs
            self.live_history = [self.runs_dir / n for n in self.history_file.read_text(encoding="utf-8").split() if (self.runs_dir / n).is_dir()]
        # jobs left unfinished when the software was killed / crashed / lost power last time (events but no summary): Reports marks them Interrupted, and a notice is shown once when the next job starts
        self.interrupted: list[Path] = scan_interrupted(self.runs_dir) if job is None else []
        for d in self.interrupted:
            if d not in self.live_history:
                self.live_history.append(d)
        acked = set(read_app().get("interrupted_ack") or [])
        self.interrupt_pending = [d.name for d in self.interrupted if d.name not in acked]
        self.client = client
        self.live: Any = None
        self.lock = threading.Lock()
        self.previews: dict[str, bytes] = {}              # camera frames before job creation in live mode (cached per site, see preview_frame)
        self.notified: set[str] = set()                   # runs for which the human-handoff webhook has been sent
        self.robot_host = robot_host                      # robot computer hostname (recorded at pairing; can be set with --robot-host)
        app = read_app()                                  # machine options saved on the settings page take precedence over launch arguments (invalid saved values are ignored)
        if app.get("policy") in ("nemotron", "rules"):
            self.policy = str(app["policy"])
        if app.get("verifier") in ("vlm", "cv"):
            self.verifier = str(app["verifier"])
        for k in ("skill_url", "camera_url"):
            if app.get(k):
                try:
                    app[k] = _check_url(app[k], self.host())
                except ValueError:
                    app[k] = None
        # robot computer address: use the one given explicitly on the command line (deploy script / end-to-end self-test), otherwise the one saved on the settings page, otherwise the default tunnel ports
        self._cli_urls = (skill_url, camera_url)
        self.skill_url = str(skill_url or app.get("skill_url") or DEFAULT_SKILL_URL)
        self.camera_url = str(camera_url or app.get("camera_url") or DEFAULT_CAMERA_URL)
        self._link: tuple[float, Optional[dict[str, Any]]] = (0.0, None)
        self._link_lock = threading.Lock()
        self._remote_frame: tuple[float, Optional[bytes]] = (0.0, None)
        self._remote_frame_lock = threading.Lock()
        self.calib_lock = threading.Lock()                # calibration does one step at a time (the on-site calibration steps write the same calib_points.json)
        self._client_stale = False                        # the API key was changed, the model client needs rebuilding
        self._last_backend: Optional[str] = None          # backend reported by the robot computer at the last successful connection (used to decide the calibration mode while it is temporarily unreachable)

    @property
    def robot_mode(self) -> str:
        """sim = the robot computer is the built-in local simulation (--sim-robot); paired = the paired robot computer is reached through the tunnel."""
        return "sim" if self.sim_robot else "paired"

    def tunnel_ports(self) -> list[Optional[int]]:
        from urllib.parse import urlparse
        out = []
        for u in (self.skill_url, self.camera_url):
            try:
                p = urlparse(u)
                out.append(p.port or (443 if p.scheme == "https" else 80))
            except ValueError:
                out.append(None)
        return out

    def remote_frame(self, max_age_s: float = 0.2) -> Optional[bytes]:
        """Camera frame in paired mode when no job is running: what the robot computer sees right now, i.e. the skill server display viewpoint /frame/view for a simulation backend,
        falling back to /frame/fixed; for a real robot the camera service /frame/fixed. Cached briefly, so page polling does not pile requests onto the robot computer."""
        with self._remote_frame_lock:
            t, jpg = self._remote_frame
            if jpg is not None and time.time() - t < max_age_s:
                return jpg
            link = self.robot_link()
            if not link or not link.get("ok"):
                return None
            import httpx

            from ..robot.server import auth_headers
            # with a simulation backend try the display viewpoint /frame/view first and fall back to the fixed camera only on 404 (this platform has no display viewpoint); on timeout / error keep the previous frame,
            # so the top view never flashes before job creation (it never appears in the video)
            urls = ([f"{self.skill_url.rstrip('/')}/frame/view", f"{self.skill_url.rstrip('/')}/frame/fixed"] if link.get("simulated")
                    else [f"{self.camera_url.rstrip('/')}/frame/fixed"])
            for u in urls:
                try:
                    r = httpx.get(u, timeout=5.0, headers=auth_headers())
                except Exception:  # noqa: BLE001
                    return jpg
                if r.status_code == 200 and r.content:
                    self._remote_frame = (time.time(), r.content)
                    return r.content
                if r.status_code != 404:
                    return jpg
            return jpg

    def key_changed(self) -> None:
        """A new API key was saved: the model client is rebuilt to use it (the running job finishes with the old client first), no software restart needed."""
        if self.live is not None and self.live.running:
            self._client_stale = True
        else:
            self.client, self._client_stale = None, False

    def get_client(self) -> Any:
        """The model client is built only when Nemotron / Cosmos is actually needed (the settings page can switch planning or inspection mode at runtime)."""
        if self._client_stale and not (self.live is not None and self.live.running):
            self.client, self._client_stale = None, False
        if self.client is None and (self.policy == "nemotron" or self.verifier == "vlm"):
            from ..common.llm_client import LLMClient
            self.client = LLMClient()
        return self.client

    def robot_link(self, max_age_s: float = 3.0) -> Optional[dict[str, Any]]:
        """Whether the paired robot computer is reachable right now (skill server /health + camera service /health, through the tunnel; the result is cached for a few seconds).
        A simulated deployment returns None; so does video recording (--job, replaying an already recorded simulated job): no network probing, and the visuals are determined solely by the recorded job."""
        if self.sim_robot or self.job is not None:
            return None
        t, v = self._link
        if v is not None and time.time() - t < max_age_s:
            return v
        with self._link_lock:                              # several page endpoints ask at once: probe only once
            t, v = self._link
            if v is not None and time.time() - t < max_age_s:
                return v
            return self._probe_link()

    def _probe_link(self) -> dict[str, Any]:
        import httpx

        from ..robot.server import auth_headers
        out: dict[str, Any] = {"ok": False, "detail": "skill server not reachable", "backend": None, "mode": None, "label": None,
                               "host": self.host(), "simulated": False, "camera": False, "calibrated": None, "site": None,
                               "busy": None, "stopped": None, "fault": None,
                               "skill_url": self.skill_url, "camera_url": self.camera_url}
        try:
            h = httpx.get(self.skill_url.rstrip("/") + "/health", timeout=1.5, headers=auth_headers())
            if h.status_code == 401:
                out["detail"] = "skill server refused the shared token"
            else:
                h.raise_for_status()
                hj = h.json()
                from ..agent.remote_job import SIM_BACKENDS
                out.update(ok=True, backend=hj.get("backend"), mode=hj.get("mode"), label=hj.get("label"), host=hj.get("host") or self.host(),
                           simulated=hj.get("backend") in SIM_BACKENDS, calibrated=hj.get("calibrated") is not False,
                           site=Path(str(hj.get("site") or "")).name or None,
                           busy=hj.get("busy") or None, stopped=hj.get("stopped") or None, fault=hj.get("fault") or None,
                           detail=f"{hj.get('label')} · mode {hj.get('mode')} · connected" + (" · token OK" if auth_headers() else ""))
        except PermissionError:
            out["detail"] = "shared token file is readable by other users (needs mode 0600)"
        except Exception:  # noqa: BLE001
            pass
        if out["ok"]:
            self._last_backend = str(out.get("backend") or "") or self._last_backend
        if out["simulated"]:                                  # a simulation backend has no separate camera service: the skill server supplies the frames itself
            out["camera"] = True
        else:
            try:
                out["camera"] = httpx.get(self.camera_url.rstrip("/") + "/health", timeout=1.5, headers=auth_headers()).status_code == 200
            except Exception:  # noqa: BLE001
                out["camera"] = False
        self._link = (time.time(), out)
        return out

    def remember(self, run_dir: Path) -> None:
        """Record one (live) job in the manifest: only the run directory name, no data is copied."""
        if self.job is not None or run_dir.parent.resolve() != self.runs_dir.resolve() or any(d.name == run_dir.name for d in self.live_history):
            return
        self.live_history.append(run_dir)
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            self.history_file.write_text("".join(d.name + "\n" for d in self.live_history), encoding="utf-8")
        except OSError:
            pass

    def host(self) -> Optional[str]:
        return self.robot_host or self.paired_host()

    @staticmethod
    def paired_host() -> Optional[str]:
        """Robot computer hostname recorded at pairing (install_edge.sh --pair writes <config directory>/robot_host)."""
        f = robot_host_file()
        try:
            return f.read_text(encoding="utf-8").strip() or None if f.is_file() else None
        except OSError:
            return None


def _run_meta(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / "reference_meta.json").read_text(encoding="utf-8")) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def run_platform(d: Path, default: str = "spot_arm") -> str:
    """Robot platform of this job: robot.platform in reference_meta.json (written by both the local simulation and the paired robot computer)."""
    m = _run_meta(d)
    return str((m.get("robot") or {}).get("platform") or default) if isinstance(m.get("robot"), dict) else default


def run_site(d: Path) -> Optional[str]:
    """The site this job belongs to: look at reference_meta.json first (the paired robot computer writes site; local live jobs record operator_site in the scene spec),
    then infer from the run directory name (live-<site>-<backend>-<mode>-<time> / sim-<scenario>-...)."""
    m = _run_meta(d)
    known = {s["id"] for s in list_sites()}
    sc = m.get("scenario") if isinstance(m.get("scenario"), dict) else {}
    for cand in (m.get("site"), sc.get("operator_site")):
        if cand and str(cand) in known:
            return str(cand)
    parts = d.name.split("-")
    if len(parts) > 1:
        if parts[0] == "live" and parts[1] in known:
            return parts[1]
        sid = site_of_scenario(parts[1])
        if sid:
            return sid
    if sc.get("site") and str(sc["site"]) in known:
        return str(sc["site"])
    return None


_REPORT_LOCK = threading.Lock()


def _load_run_isolated(rd: Path) -> Any:
    """Read the report data of a job without touching the global site (CLEANPROOF_SITE): report.rundata.load_run calls use_site for the site the run registered,
    and the job currently running (another thread) reads its config by the global site and must not be switched away by it. Here its use_site is made a no-op in this thread only,
    and the zone grid is read by the site of this job itself (local simulated job: the site registered for the scene; paired robot computer: the site in reference_meta)."""
    import cleanproof.common.config as cfg
    import cleanproof.report.rundata as rdm
    m = _run_meta(rd)
    sc = m.get("scenario") if isinstance(m.get("scenario"), dict) else None
    grid_site = str((sc or {}).get("site") or "") if sc is not None else str(m.get("site") or "")
    if grid_site and not (SITE_DIR / grid_site / "site.yaml").is_file():
        grid_site = ""
    me = threading.get_ident()
    with _REPORT_LOCK:
        orig_use, orig_load = cfg.use_site, rdm.load_yaml

        def use_site_other_threads(site: Optional[str]) -> None:
            if threading.get_ident() != me:
                orig_use(site)

        def load_yaml_for_run(name: Any, site: Optional[str] = None) -> dict[str, Any]:
            if threading.get_ident() == me and site is None:
                site = grid_site
            return orig_load(name, site=site)
        cfg.use_site, rdm.load_yaml = use_site_other_threads, load_yaml_for_run
        try:
            return rdm.load_run(rd)
        finally:
            cfg.use_site, rdm.load_yaml = orig_use, orig_load


def _requests_file(run_dir: Path) -> Path:
    return run_dir / "requests.json"


def read_requests(run_dir: Path) -> list[dict[str, Any]]:
    s = run_dir / "summary.json"
    if not s.is_file():
        return []
    reqs = json.loads(s.read_text(encoding="utf-8")).get("human_requests") or []
    res = json.loads(_requests_file(run_dir).read_text(encoding="utf-8")) if _requests_file(run_dir).is_file() else {}
    out = []
    for i, r in enumerate(reqs):
        key = str(i)
        out.append({"id": key, "step": r.get("step"), "zones": r.get("zones") or [], "reason": r.get("reason", ""),
                    "resolved": res.get(key)})
    return out


def resolve_request(run_dir: Path, req_id: str, note: str, by: str = "operator") -> dict[str, Any]:
    reqs = {r["id"] for r in read_requests(run_dir)}
    if req_id not in reqs:
        raise KeyError(req_id)
    f = _requests_file(run_dir)
    res = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    res[req_id] = {"note": note[:300], "by": by, "ts": datetime.now(timezone.utc).isoformat()}
    f.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return res[req_id]


def create_operator_app(st: OperatorState, allowed_hosts: tuple[str, ...] = ()):
    from fastapi import Body, FastAPI, Header, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

    from ..robot.platforms import load_platforms
    app = FastAPI(title="CleanProof", docs_url=None, redoc_url=None)
    app.state.op = st
    from fastapi.exceptions import RequestValidationError

    # ---- cross-site protection: a web page cannot click approve / estop for the operator, and DNS rebinding cannot get in ----
    # Host is accepted only for this machine (127.0.0.1 / localhost / ::1) and the host names configured at startup; testserver is the fixed Host of the test client, not a registrable public domain.
    host_ok = {"127.0.0.1", "localhost", "::1", "testserver"} | {h.lower() for h in allowed_hosts}

    def _host_name(h: str) -> str:
        h = (h or "").strip().lower()
        if h.startswith("["):                               # [::1]:8092
            return h[1:].split("]")[0]
        return h.rsplit(":", 1)[0] if h.count(":") == 1 else h

    @app.middleware("http")
    async def cross_site_guard(request: Any, call_next: Any) -> Any:
        host = request.headers.get("host", "")
        if _host_name(host) not in host_ok:
            return JSONResponse({"detail": "Unexpected Host header. Open CleanProof at http://127.0.0.1:<port>/ "
                                            "(or start it with --allow-host for another name)."}, status_code=403)
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype != "application/json" and not request.headers.get("x-cleanproof"):
                return JSONResponse({"detail": "Blocked: state-changing requests must be JSON or carry the X-CleanProof header."}, status_code=403)
            origin = request.headers.get("origin")
            if origin is not None and origin.split("://", 1)[-1].lower() != host.lower():       # a browser request that carries Origin must be same-origin
                return JSONResponse({"detail": "Blocked: cross-site request."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ConfigError)
    async def bad_config(request: Any, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"detail": "Configuration problem: " + str(exc), "config_errors": exc.problems}, status_code=422)

    @app.exception_handler(yaml.YAMLError)
    async def bad_yaml(request: Any, exc: yaml.YAMLError) -> JSONResponse:
        return JSONResponse({"detail": "A configuration file could not be read. Check the site files for typos."}, status_code=422)

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Any, exc: RequestValidationError) -> JSONResponse:
        """Malformed request: only say which field is wrong, never echo the request content (which may contain an API key)."""
        where = sorted({".".join(str(x) for x in e.get("loc", ()) if x != "body") or "body" for e in exc.errors()})
        return JSONResponse({"detail": "Invalid request" + (f" ({', '.join(where[:5])})" if where else "")}, status_code=422)
    _reports: dict[str, str] = {}
    _top: dict[tuple[str, int], bytes] = {}                # (site, reference version) → PNG

    @app.get("/", response_class=HTMLResponse)
    def page() -> str:
        return (STATIC / "operator.html").read_text(encoding="utf-8")

    # ---- overview ----
    @app.get("/api/meta")
    def meta() -> JSONResponse:
        errs = config_errors()
        if any(e["site"] == st.site for e in errs):          # the current site's config is broken: return only the errors, the UI shows a banner (no blank page)
            return JSONResponse({"site": st.site, "sites": list_sites(), "robots": [], "supported": [], "job": None, "mode": "config_error",
                                 "config_errors": errs, "setup_done": True, "spent_usd": None, "runtime": {"policy": st.policy, "verifier": st.verifier}})
        sites = list_sites()
        specs = load_platforms()
        robots = []
        for s in sites:
            k = s["robot"]
            if k in specs:
                tk = st.robot_takes.get(k)
                robots.append({"key": k, "label": specs[k].label, "kind": specs[k].kind, "site": s["id"], "site_name": s["name"],
                               "has_feed": tk is not None, "t0": tk.t0 if tk else None, "t1": tk.t1 if tk else None,
                               "last_run": tk.run_dir.name if tk else None,
                               "last_status": report_row(tk.run_dir, specs[k].label)["status"] if tk else None})
        supported = [{"key": k, "label": s.label, "kind": s.kind, "dof": s.dof,
                      "group": "Mobile manipulators" if "mobile" in s.kind.lower() else "Arms",
                      "photo": (PHOTO_DIR / f"{k}.png").is_file(),
                      "adapter": "RTDE" if k.startswith("ur") else "software adapter"} for k, s in specs.items()]
        job = None
        if st.job is not None:
            job = {"run_id": st.job.run_dir.name, "t0": st.job.t0, "t1": st.job.t1, "scenario": st.job.meta.get("scenario"),
                   "site": site_of_scenario(str(st.job.meta.get("scenario"))) or st.site,
                   "robot": (st.job.meta.get("robot") or {}).get("platform", "spot_arm"), "zones": st.job.targets,
                   "events": st.job.event_times()}
        from .deploy_checks import _git_rev
        return JSONResponse({"site": st.site, "sites": sites, "robots": robots, "supported": supported, "job": job,
                             "mode": "recorded" if st.job is not None else "live", "spent_usd": read_ledger(st.ledger),
                             "setup_done": setup_file().is_file(), "site_types": SITE_TYPES, "site_type_stored": SITE_TYPE_STORED, "version": _git_rev(),
                             "robot_host": st.host(), "sim_robot": st.sim_robot, "robot_link": st.robot_link(),
                             "robot_mode": st.robot_mode, "app_version": APP_VERSION, "tunnel_ports": st.tunnel_ports(),
                             "sim_templates": sim_templates() if st.sim_robot else [],
                             "config_errors": errs, "runtime": {"policy": st.policy, "verifier": st.verifier}})

    @app.get("/api/robot/link")
    def robot_link_status() -> JSONResponse:
        """The Robot connected badge in the title bar (the page asks every 3 s): a simulated deployment checks whether the local simulation can start, paired mode uses the live health check of the robot computer."""
        link = st.robot_link()
        robot = next((x["robot"] for x in list_sites() if x["id"] == st.site), "")
        sk = _sim_overall(robot) if st.sim_robot else {"ok": bool(link and link.get("ok")), "detail": (link or {}).get("detail", "")}
        return JSONResponse({"mode": st.robot_mode, "link": link, "skill_server": sk})

    # ---- first-run wizard ----
    def _checks(ids: tuple[str, ...], fresh: Any) -> list[dict[str, Any]]:
        """If a saved deploy check exists (--deploy-record), return these items from that real check; otherwise check on the spot."""
        if st.deploy_record is not None and st.deploy_record.is_file():
            rec = json.loads(st.deploy_record.read_text(encoding="utf-8"))
            return [c for c in rec["checks"] if c["id"] in ids]
        return fresh()

    @app.post("/api/setup/connect")
    def setup_connect(body: dict[str, Any] = Body(default={})) -> JSONResponse:
        robot = str(body.get("robot") or next((x["robot"] for x in list_sites() if x["id"] == st.site), "spot_arm"))

        def fresh() -> list[dict[str, Any]]:
            if st.sim_robot:                                  # simulated deployment: really check that the local simulation can start (MuJoCo, platform adapter, camera rendering)
                return sim_robot_checks(robot)
            return robot_computer_checks(st.skill_url, st.camera_url, robot)
        cs = _checks(("skill_server", "camera", "adapter"), fresh)
        return JSONResponse({"ok": all(c.get("ok") for c in cs), "checks": cs})

    @app.post("/api/setup/models")
    def setup_models(body: dict[str, Any] = Body(default={})) -> JSONResponse:
        from .deploy_checks import _timed, check_cosmos, check_orchestrator, check_tokenfactory
        refused = None
        if body.get("api_key"):
            try:
                key = _valid_key(str(body["api_key"]))
            except ValueError as e:
                raise HTTPException(400, _cap(e)) from e
            refused = verify_then_save_key(key, allow_unverified=st.deploy_record is not None)     # verify first, then write: a mistyped key does not overwrite a valid existing one
            if refused is None:
                st.key_changed()
        def fresh() -> list[dict[str, Any]]:
            # same order as the saved deploy check and /api/deploy/checks (the Services rows in the video are ticked in this order)
            cs_ = [{"id": "orchestrator", "title": "CleanProof orchestrator", **_timed(check_orchestrator)},
                   {"id": "cosmos", "title": "Cosmos 3 inference server", **_timed(check_cosmos)},
                   {"id": "tokenfactory", "title": "Nebius Token Factory", **_timed(check_tokenfactory)}]
            key_set = _api_key_set()
            for c in cs_:
                if c["ok"]:
                    continue
                if c["id"] == "cosmos":
                    # Cosmos is not needed by the current settings: not a failure. If it is needed (verifier = vlm) but unreachable, the wizard is not blocked either: when a Cosmos call fails in the orchestration loop
                    # each zone is judged by classical vision (perception/verifier.py VLMVerifier.inspect → cv_fallback), and this is stated honestly
                    c.update(ok=True, fallback=True, detail="Not running here · classical vision check is used" if st.verifier != "vlm"
                             else "Not reachable · zone checks fall back to classical vision")
                elif c["id"] == "tokenfactory":
                    # no key set: fall back to rule-based planning. Key set but rejected / unreachable: state the reason as is; when planning uses rules the wizard is not blocked (amber), when using Nemotron it counts as a failure
                    if not key_set:
                        if st.policy != "nemotron":
                            c.update(ok=True, fallback=True, detail="Not configured · rules planner is used")
                    else:
                        c["detail"] = _tf_reason(str(c.get("detail") or ""))
                        if st.policy != "nemotron":
                            c.update(ok=True, fallback=True, detail=c["detail"] + " · rules planner is used")
            return cs_
        cs = _checks(("tokenfactory", "cosmos", "orchestrator"), fresh)
        if refused:                                                 # the candidate key did not pass: Token Factory reports it in one line, the old key stays
            for c in cs:
                if c["id"] == "tokenfactory":
                    c.update(ok=False, fallback=False, detail=refused)
        return JSONResponse({"ok": all(c.get("ok") for c in cs), "checks": cs, "saved": refused is None})

    def _calib_how() -> str:
        """auto = the robot computer is a simulation (built-in or a simulation backend) → one-click calibration; guided = real robot, done step by step on site."""
        if st.sim_robot or st.job is not None:                 # video recording replays a job from simulation
            return "auto"
        link = st.robot_link()
        if link and link.get("ok"):
            return "auto" if link.get("simulated") else "guided"
        from ..agent.remote_job import SIM_BACKENDS          # temporarily unreachable: use the backend from the last successful connection; if never connected, honestly say it is unknown
        if st._last_backend is None:
            return "unknown"
        return "auto" if st._last_backend in SIM_BACKENDS else "guided"

    def _site_payload(sid: str) -> dict[str, Any]:
        ensure_site_ok(sid)
        s_ = next((x for x in list_sites() if x["id"] == sid), None)
        if s_ is None:
            raise HTTPException(404, "Unknown site")
        g = s_["grid"]
        d = SITE_DIR / sid
        calibrated = bool(s_["calibrated"])
        link = st.robot_link()
        real = bool(link and link.get("ok") and not link.get("simulated"))
        if calibrated and real and link.get("calibrated") is False and link.get("site") in (None, sid):
            calibrated = False                                 # this site is not yet calibrated / synced on the robot computer: honestly counted as uncalibrated
        how = _calib_how()
        needs_sync = False
        if calibrated and not st.sim_robot and st.job is None and calibration.load_points(d).get("app"):   # paired live mode only
            # a site calibrated with in-app guidance: the reference image is captured but not yet synced to the robot computer (page closed midway, etc.): not counted as calibrated, and the guide panel returns to the Sync step.
            # if the robot computer reports this very site and it is calibrated (for example synced elsewhere), it also counts as synced.
            steps = {x["id"]: x["done"] for x in calibration.guided_state(sid, d)["steps"]}
            synced = steps["sync"] or (real and link.get("site") == sid and link.get("calibrated") is True)
            if steps["reference"] and not synced:
                calibrated, needs_sync, how = False, True, "guided"
        out: dict[str, Any] = {"site": sid, "name": s_["name"], "type": s_["type"], "surface": s_["surface_label"],
                               "calibrated": calibrated, "zones": g["rows"] * g["cols"], "grid": g,
                               "reference": None, "calibrated_utc": None, "how": how, "needs_sync": needs_sync}
        if calibrated:
            rec = calibration.read_record(d)
            day = str(rec.get("calibrated_utc") or "")[:10]
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                day = datetime.fromtimestamp((d / "zones.yaml").stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")
            out.update(reference=f"/api/sites/{sid}/reference.png?v={_reference_version(sid)}", calibrated_utc=day)
        return out

    @app.get("/api/setup/site")
    def setup_site(site: Optional[str] = Query(None)) -> JSONResponse:
        """Wizard Site step: the calibration result of this site (zone grid covering the whole tabletop, clean reference; people do not draw zones).
        If never calibrated, honestly return calibrated=false; how says how this robot computer is calibrated (auto one-click / guided on-site)."""
        return JSONResponse(_site_payload(str(site or st.site)))

    @app.post("/api/setup/calibrate")
    def setup_calibrate(body: dict[str, Any] = Body(default={})) -> JSONResponse:
        """One-click calibration (when the robot computer is a simulation): zone grid and tabletop height come from the simulated scene, and the clean reference from the top view before stains are placed (cleanproof/ui/calibration.py).
        A real robot returns 409 guided (use on-site guided calibration /api/calibrate/*)."""
        sid = str(body.get("site") or st.site)
        s_ = next((x for x in list_sites() if x["id"] == sid), None)
        if s_ is None:
            raise HTTPException(404, "Unknown site")
        if st.live is not None and st.live.running:
            raise HTTPException(409, "A job is running; calibrate when it has finished")
        d = SITE_DIR / sid
        try:
            if st.sim_robot:
                if not s_.get("job_scenario"):
                    raise HTTPException(400, "This site has no simulated scene to calibrate from")
                with st.lock:
                    zones, ref, sim_site = calibration.capture_builtin_sim(str(s_["job_scenario"]))
                    calibration.apply_auto(d, zones, ref, "built-in simulation", s_["surface"], compare_look=False,
                                           sim_site=sim_site, scenario=s_["job_scenario"])
            else:
                link = st.robot_link(max_age_s=0.0)
                if not link or not link.get("ok"):
                    raise HTTPException(400, "The robot computer is not reachable: " + str((link or {}).get("detail") or "check the pairing"))
                if not link.get("simulated"):
                    raise HTTPException(409, "guided")
                with st.calib_lock:
                    zones, ref = calibration.capture_robot_sim(st.skill_url)
                    calibration.apply_auto(d, zones, ref, f"robot computer simulation · {link.get('label') or link.get('backend')}",
                                           s_["surface"], compare_look=True, backend=link.get("backend"), host=link.get("host"))
        except calibration.CalibrationError as e:
            raise HTTPException(400, _cap(e)) from e
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001  the simulation could not be built / the robot computer disconnected midway: give the operator an understandable reason
            raise HTTPException(400, f"Calibration failed ({type(e).__name__}); check the robot computer and try again") from e
        _forget_site(sid)
        return JSONResponse(_site_payload(sid))

    # ---- guided on-site calibration (real robot) ----
    def _guided_site(sid: Optional[str]) -> tuple[str, dict[str, Any]]:
        sid = str(sid or st.site)
        s_ = next((x for x in list_sites() if x["id"] == sid), None)
        if s_ is None:
            raise HTTPException(404, "Unknown site")
        return sid, s_

    @app.get("/api/calibrate/state")
    def calibrate_state(site: Optional[str] = Query(None)) -> JSONResponse:
        sid, _ = _guided_site(site)
        return JSONResponse(calibration.guided_state(sid, SITE_DIR / sid))

    @app.get("/api/calibrate/markers.png")
    def calibrate_markers(size_mm: float = Query(40.0, ge=20.0, le=80.0)) -> Response:
        return Response(calibration.markers_png(size_mm), media_type="image/png",
                        headers={"Content-Disposition": 'inline; filename="cleanproof_markers.png"'})

    @app.post("/api/calibrate/step")
    def calibrate_step(body: dict[str, Any] = Body(...)) -> JSONResponse:
        sid, s_ = _guided_site(body.get("site"))
        step = str(body.get("step") or "")
        if st.live is not None and st.live.running:
            raise HTTPException(409, "A job is running; calibrate when it has finished")
        if not st.calib_lock.acquire(timeout=1.0):
            raise HTTPException(409, "Another calibration step is still running")
        try:
            if step == "sync":                                 # syncing restarts the skill server: it must not restart while the robot is still executing a skill
                link = st.robot_link(max_age_s=0.0)
                if link and link.get("busy"):
                    raise HTTPException(409, "The robot is still running a skill; wait for it to finish or press E-stop")
            out = calibration.guided_step(sid, SITE_DIR / sid, step, dict(body), st.skill_url, st.camera_url, host=st.host(),
                                          robot=str(s_.get("robot") or ""))
        except calibration.CalibrationError as e:
            raise HTTPException(400, _cap(e)) from e
        finally:
            st.calib_lock.release()
        if step in ("zones", "reference", "sync"):
            _forget_site(sid)
        if step == "sync":                                    # the services on the robot computer just restarted: re-probe the title bar, connection checks and frames
            st._link = (0.0, None)
            st._remote_frame = (0.0, None)
        return JSONResponse(out)

    def _forget_site(sid: str) -> None:
        """Calibration changed the zones or the reference image: drop the cached frames of this site (reference image, pre-job preview)."""
        st.previews.pop(sid, None)
        for k in [k for k in _top if k[0] == sid]:
            _top.pop(k, None)

    @app.post("/api/sites")
    def new_site(body: dict[str, Any] = Body(...)) -> JSONResponse:
        """Create a new site (Sites page / wizard): write configs/site/<slug>/site.yaml; calibration is done afterwards in the wizard Site step."""
        try:
            s_ = create_site(str(body.get("name") or ""), str(body.get("type") or ""), str(body.get("robot") or ""), st.sim_robot,
                             template=body.get("template") or None, current=st.site, surface=body.get("surface") or None)
        except (ValueError, KeyError) as e:
            raise HTTPException(400, _cap(e)) from e
        return JSONResponse({"site": s_})

    @app.post("/api/setup/finish")
    def setup_finish(body: dict[str, Any] = Body(default={})) -> JSONResponse:
        if st.live is not None and st.live.running:            # changing the site switches the global site config: the job currently running must not be affected
            raise HTTPException(409, "A job is running; finish setup after it ends")
        rec = {"robot": str(body.get("robot") or ""), "site": str(body.get("site") or st.site), "site_type": str(body.get("site_type") or ""),
               "site_name": str(body.get("site_name") or ""), "finished_utc": datetime.now(timezone.utc).isoformat()}
        try:
            update_site(rec["site"], {"robot": rec["robot"], "type": rec["site_type"], "name": rec["site_name"]})
        except (KeyError, ValueError) as e:
            raise HTTPException(400, _cap(e)) from e
        f = setup_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        if st.job is None and rec["site"] != st.site:          # live mode: the site chosen in the wizard is the current site of this computer (from setup.json after a restart)
            st.site = rec["site"]
            try:
                use_site(st.site)
            except FileNotFoundError:
                pass
        return JSONResponse({"ok": True, **rec})

    # ---- deploy ----
    @app.get("/api/deploy/checks")
    def deploy_checks(recorded: int = Query(0)) -> JSONResponse:
        if recorded:
            if st.deploy_record is None or not st.deploy_record.is_file():
                raise HTTPException(404, "No recorded deployment check")
            return JSONResponse(json.loads(st.deploy_record.read_text(encoding="utf-8")))
        from .deploy_checks import _git_rev, _timed, check_cosmos, check_orchestrator, check_tokenfactory, run_checks
        site = st.site
        robot = next((s["robot"] for s in list_sites() if s["id"] == site), "spot_arm")
        if not st.sim_robot and not _is_sim_link(st):
            return JSONResponse(run_checks(st.skill_url, st.camera_url, platform=robot))
        cs = [{"id": "orchestrator", "title": "CleanProof orchestrator", **_timed(check_orchestrator)},
              {"id": "cosmos", "title": "Cosmos 3 inference server", **_timed(check_cosmos)},
              {"id": "tokenfactory", "title": "Nebius Token Factory", **_timed(check_tokenfactory)}]
        cs += sim_robot_checks(robot) if st.sim_robot else robot_computer_checks(st.skill_url, st.camera_url, robot)
        return JSONResponse({"site": site, "ts": datetime.now(timezone.utc).isoformat(), "git": _git_rev(),
                             "ok": all(c.get("ok") for c in cs), "checks": cs})

    # ---- sites ----
    @app.get("/api/sites/{site}/reference.png")
    def site_reference(site: str, if_none_match: Optional[str] = Header(None)) -> Response:
        """Site clean reference image (top view, aligned pixel by pixel with the zone grid): the one captured at calibration; simulated sites that never captured one are rendered from the tabletop material.
        Cached by file version (?v= changes after calibration); an address without ?v= revalidates the version with the server each time (ETag), so the page also gets the new image after recalibration."""
        s = next((x for x in list_sites() if x["id"] == site), None)
        if s is None:
            raise HTTPException(404, "Unknown site")
        key = (site, _reference_version(site))
        etag = f'"{site}-{key[1]}"'
        hdr = {"ETag": etag, "Cache-Control": "no-cache"}
        if if_none_match == etag:
            return Response(status_code=304, headers=hdr)
        if key not in _top:
            import io

            import numpy as np
            from PIL import Image

            from ..common.zones import ZoneGrid
            from ..sim.dirt import DirtMap
            ref = SITE_DIR / site / "reference.png"
            if ref.is_file():
                _top[key] = ref.read_bytes()
            elif _has_calib_points(SITE_DIR / site):                # a site calibrated on site that has not captured its clean reference yet: do not pass off a render as one
                raise HTTPException(404, "No clean reference yet")
            else:
                g = ZoneGrid(load_yaml("zones.yaml", site=site))
                dm = DirtMap(g, load_yaml("sim.yaml", site=site)["dirt"], surface=s["surface"] or "sealed_wood")
                rgb = np.clip(dm.rgb(), 0, 255).astype(np.uint8)
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="PNG")
                _top[key] = buf.getvalue()
        return Response(_top[key], media_type="image/png", headers=hdr)

    @app.get("/api/task/parse")
    def task_parse(text: str = Query("", max_length=400), site: Optional[str] = Query(None)) -> JSONResponse:
        """The "Zones in this job" line of the Jobs page: for the task box, parsed with the same server-side rule (exclusions, ranges, phrasings that are not recognised)."""
        try:
            g = read_settings(site or st.site)["grid"]
        except (KeyError, OSError, ValueError, yaml.YAMLError) as e:
            raise HTTPException(400, _cap(e)) from e
        return JSONResponse(parse_task(text, g["rows"], g["cols"]))

    # ---- settings ----
    def _settings_payload(site: str) -> dict[str, Any]:
        ensure_site_ok(site)
        try:
            out = read_settings(site)
        except KeyError as e:
            raise HTTPException(404, _cap(e)) from e
        from .deploy_checks import _git_rev
        robot = next((x["robot"] for x in list_sites() if x["id"] == site), "")
        specs = load_platforms()
        out["status"] = service_status(st.deploy_record, st.sim_robot, st.robot_link(), robot)
        out["robot"] = {"key": robot, "label": specs[robot].label if robot in specs else robot,
                        "adapter": "RTDE" if robot.startswith("ur") else "software adapter", "host": st.host(),
                        "skill_url": st.skill_url, "camera_url": st.camera_url, "sim_robot": st.sim_robot, "mode": st.robot_mode}
        out["runtime"] = {"policy": st.policy, "verifier": st.verifier, "api_key_set": _api_key_set()}
        out["data"] = {"calls_log": str(st.calls_log.relative_to(REPO_ROOT)) if st.calls_log.is_relative_to(REPO_ROOT) else st.calls_log.name,
                       "runs_dir": str(st.runs_dir.relative_to(REPO_ROOT)) if st.runs_dir.is_relative_to(REPO_ROOT) else st.runs_dir.name,
                       "reports": len(_report_dirs())}
        out["about"] = {"version": _git_rev(), "release": APP_VERSION, "license": "Apache-2.0"}
        # On a real robot (ur3e backend) there is also a separately accepted limit: a plan above it is refused by the robot computer and handed to a person (robot.yaml force_limits.real_validated_max_n)
        lk = st.robot_link() if not st.sim_robot and st.job is None else None
        backend = (lk or {}).get("backend") or st._last_backend
        out["real_limit"] = {"active": backend == "ur3e", "max_n": float((load_yaml("robot.yaml").get("force_limits") or {}).get("real_validated_max_n", 5.0))}
        return out

    @app.get("/api/settings")
    def get_settings(site: Optional[str] = Query(None)) -> JSONResponse:
        return JSONResponse(_settings_payload(site or st.site))

    @app.put("/api/settings")
    def put_settings(body: dict[str, Any] = Body(...), site: Optional[str] = Query(None)) -> JSONResponse:
        """Site-level settings are written to configs/site/<site>/; machine-level ones (planning / inspection mode, robot computer address, API key) go to the user config directory.
        All take effect from the next job. Everything is validated before writing, and one invalid item rejects the whole request (nothing is written); returns the same full settings as GET. Keys are not echoed."""
        sid = site or st.site
        body = dict(body)
        app_upd: dict[str, Any] = {}
        raw_key = body.pop("api_key", None)
        rt = body.pop("runtime", None) or {}
        link = body.pop("robot_link", None) or {}
        try:
            key = None
            if raw_key is not None and str(raw_key).strip():
                key = _valid_key(str(raw_key))
            if not isinstance(rt, dict) or not isinstance(link, dict):
                raise ValueError("Runtime and robot_link must be objects")
            bad = sorted(set(rt) - {"policy", "verifier"}) + sorted(set(link) - {"skill_url", "camera_url"})
            if bad:
                raise ValueError(f"Unknown setting {bad[0]!r}")
            if "policy" in rt:
                if rt["policy"] not in ("nemotron", "rules"):
                    raise ValueError("Planner must be nemotron or rules")
                if rt["policy"] == "nemotron" and key is None and not _api_key_set():
                    raise ValueError("Set the Nebius Token Factory API key before switching the planner to Nemotron")
                app_upd["policy"] = rt["policy"]
            if "verifier" in rt:
                if rt["verifier"] not in ("vlm", "cv"):
                    raise ValueError("Zone check must be vlm (Cosmos 3) or cv (classical vision)")
                app_upd["verifier"] = rt["verifier"]
            for k in ("skill_url", "camera_url"):
                if k in link:                                   # empty = back to the default (tunnel port)
                    app_upd[k] = _check_url(link[k], st.host()) if str(link[k] or "").strip() else None
            if body:
                write_settings(sid, body, dry_run=True)          # site settings are validated first as well
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, _cap(e)) from e
        if key is not None:
            refused = verify_then_save_key(key)                    # verify first, then write; the site settings were validated before this, and when the key does not pass the other settings are not written
            if refused:
                raise HTTPException(400, refused)
            st.key_changed()
        if body:
            write_settings(sid, body)
        if app_upd:
            write_app(app_upd)
            st.policy = app_upd.get("policy", st.policy)
            st.verifier = app_upd.get("verifier", st.verifier)
            urls = {k: (app_upd[k] or st._cli_urls[n] or (DEFAULT_SKILL_URL, DEFAULT_CAMERA_URL)[n]) if k in app_upd else getattr(st, k)
                    for n, k in enumerate(("skill_url", "camera_url"))}
            if urls["skill_url"] != st.skill_url or urls["camera_url"] != st.camera_url:
                st.skill_url, st.camera_url = urls["skill_url"], urls["camera_url"]
                st._link = (0.0, None)
                st._remote_frame = (0.0, None)
        return JSONResponse(_settings_payload(sid))

    @app.post("/api/settings/test-webhook")
    def test_webhook(site: Optional[str] = Query(None)) -> JSONResponse:
        import httpx
        url = read_settings(site or st.site)["notifications"]["staff_check_webhook"]
        if not url:
            raise HTTPException(409, "No webhook configured")
        try:
            r = httpx.post(url, json={"event": "test", "site": st.site}, timeout=5.0)
            return JSONResponse({"ok": r.status_code < 400, "status_code": r.status_code})
        except httpx.HTTPError as e:
            return JSONResponse({"ok": False, "error": type(e).__name__})

    @app.get("/api/forcedb/lookup")
    def lookup(stain: str = Query(...), surface: str = Query(...), site: Optional[str] = Query(None)) -> JSONResponse:
        return JSONResponse(force_lookup(site or st.site, stain, surface))

    # ---- logs ----
    @app.get("/api/logs/models")
    def logs_models(model: Optional[str] = Query(None), limit: int = Query(200, ge=1, le=2000)) -> JSONResponse:
        rows = read_calls(st.calls_log, limit=limit, model=model)
        return JSONResponse({"calls": rows, "spent_usd": read_ledger(st.ledger)})

    # ---- task: the recorded job (video) or live on this machine ----
    @app.get("/api/jobs/state")
    def job_state(t: float = Query(0.0), nev: Optional[int] = Query(None)) -> JSONResponse:
        if st.job is not None:
            s = st.job.state(t, nev)
            s["requests"] = read_requests(st.job.run_dir) if s["status"] != "running" else []
            return JSONResponse(s)
        if st.live is None:
            return JSONResponse({"status": "idle"})
        return JSONResponse(_live_state())

    def _live_state() -> dict[str, Any]:
        s = st.live
        status = s.status()["live"]
        stt = "running" if status["running"] else (status["final_status"] or ("error" if status["error"] else "idle"))   # the simulation thread failed: show it honestly
        out: dict[str, Any] = {"status": stt, "live": status,
                               "zones": {}, "log": [], "wipes": 0, "checks": 0, "cost_usd": 0.0}   # complete even right after job creation, before any event is written
        if s.run_dir is not None and (Path(s.run_dir) / "events.jsonl").is_file():
            from .fleet import summarize_events
            evs = []
            for ln in (Path(s.run_dir) / "events.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    evs.append(json.loads(ln))
                except json.JSONDecodeError:                   # the last line that is still being written
                    break
            start = next((e for e in evs if e["kind"] == "task_start"), {})
            sm = summarize_events(evs, list(start.get("targets") or (start.get("task") or {}).get("zones") or []),
                                  (Path(s.run_dir) / "audit.md").is_file())
            out.update({k: v for k, v in sm.items() if k != "status"})
            out["task"] = {"instruction": (start.get("task") or {}).get("instruction")}
        if s.run_dir is not None:
            out["requests"] = read_requests(Path(s.run_dir)) if status["finished"] else []
            out["run_id"] = Path(s.run_dir).name
            url = read_settings(st.site)["notifications"]["staff_check_webhook"]
            if out["requests"] and url and out["run_id"] not in st.notified:     # human handoff: send the webhook once, per settings
                st.notified.add(out["run_id"])
                notify_staff(url, {"event": "staff_check", "site": st.site, "job": out["run_id"],
                                   "requests": [{"zones": r["zones"], "reason": r["reason"]} for r in out["requests"]]})
        return out

    @app.get("/api/jobs/frame")
    def job_frame(t: float = Query(0.0), robot: Optional[str] = Query(None), site: Optional[str] = Query(None)) -> Any:
        tk = st.robot_takes.get(robot) if robot else st.job
        if tk is not None:
            return FileResponse(tk.frame_path(t), media_type="image/jpeg")
        if not st.sim_robot and st.job is None and not robot and (st.live is None or not st.live.running):
            jpg = st.remote_frame()                            # paired mode, no job running: the frame the robot computer sees right now (no local simulation is built)
            if jpg is not None:
                return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
            if st.live is None or st.live.frame() is None:
                raise HTTPException(404, "No camera frame")
        if st.sim_robot and site is not None and st.job is None and (st.live is None or not st.live.running):   # the page wants the pre-job preview (including after the site was changed once the previous job ended)
            jpg = preview_frame(st, site)
            if jpg is not None:
                return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        if st.live is not None:
            jpg = st.live.frame()
            if jpg is not None:
                return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        elif st.job is None and not robot and st.sim_robot:  # live mode, no job yet: the camera sees the tabletop as it will be at job creation (this site job template, stains already placed)
            jpg = preview_frame(st, site or st.site)
            if jpg is not None:
                return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
        raise HTTPException(404, "No camera frame")

    @app.get("/api/robots/state")
    def robot_state(robot: str = Query(...), t: float = Query(0.0)) -> JSONResponse:
        tk = st.robot_takes.get(robot)
        if tk is None:
            raise HTTPException(404, "No recent job for this robot")
        return JSONResponse(tk.state(t))

    @app.post("/api/jobs/start")
    def job_start(body: dict[str, Any] = Body(default={})) -> JSONResponse:
        from .session import LiveSession, RemoteSession
        if st.job is not None:
            raise HTTPException(409, "This instance shows a recorded job; start the software without --job to run live")
        with st.lock:
            if st.live is not None and st.live.running:
                raise HTTPException(409, "A job is already running")
            notice = None
            if st.interrupt_pending:                            # a job the software did not finish last time: say so once when a new job starts (Reports marks it Interrupted), without blocking the job
                names, st.interrupt_pending = st.interrupt_pending, []
                write_app({"interrupted_ack": sorted(set(read_app().get("interrupted_ack") or []) | set(names))[-200:]})
                notice = ("The previous job was interrupted (CleanProof closed before it finished). Check that the robot is idle and the surface "
                          "is as expected. Its record is in Reports, marked Interrupted.")
            site = str(body.get("site") or st.site)
            ensure_site_ok(site)                                # rows: 0, a job_scenario that does not exist, a negative number in the force table: stop before the job starts and name the file and line
            s = next((x for x in list_sites() if x["id"] == site), None)
            if s is None:
                raise HTTPException(400, f"Unknown site {site!r}")
            if st.sim_robot and not s.get("job_scenario"):     # built-in simulation: this site needs a simulation scene (in paired mode the tabletop is the real one on site)
                raise HTTPException(400, f"Site {site!r} has no job template")
            cfg = read_settings(site)
            approval = cfg["require_plan_approval"]
            jo = job_options(site)                             # the Jobs group on the settings page: takes effect from this job
            instruction = str(body.get("instruction") or "").strip()[:200]
            no_wipe = {str(z).upper() for z in (jo["no_wipe_zones"] or [])}
            parsed = parse_task(instruction, cfg["grid"]["rows"], cfg["grid"]["cols"])
            if parsed["mode"] != "all" and not parsed["zones"]:
                raise HTTPException(400, "Every zone is left out, so there is nothing to clean. Check the task.")
            if parsed["confirm"] and not body.get("confirm_scope"):      # a negation without a recognised zone: do not silently wipe everything, ask for confirmation before the job starts
                return JSONResponse({"detail": parsed["notes"][0], "needs_confirm": True}, status_code=409)
            asked = parsed["zones"]
            named = [z for z in asked if z not in no_wipe]       # drop no-wipe zones from the named zones
            if asked and not named:
                raise HTTPException(400, f"{', '.join(asked)} {'is a no-wipe zone' if len(asked) == 1 else 'are no-wipe zones'} at this site")
            if st.policy == "nemotron" or st.verifier == "vlm" or st._client_stale:
                try:
                    st.get_client()
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(400, f"Cannot reach the models ({type(e).__name__}); check the API key in Settings") from e
            if st.live is not None and st.live.run_dir is not None:
                st.remember(Path(st.live.run_dir))
            if not st.sim_robot:                               # paired robot computer: skills run on the robot side and the tabletop is the real one on site
                link = st.robot_link(max_age_s=0.0)
                if not link or not link.get("ok"):
                    raise HTTPException(400, "The robot computer is not reachable: " + str((link or {}).get("detail") or "check the pairing"))
                if link.get("stopped"):                        # a job would only end with a software error anyway: say clearly first what to do
                    raise HTTPException(400, "The robot computer is stopped (E-stop). Reset it on site before the next job.")
                if link.get("fault"):
                    short = str(link["fault"]).strip().splitlines()[0][:80] if str(link["fault"]).strip() else "unknown"
                    raise HTTPException(400, f"The robot computer reports a fault ({short}). Reset it on site before the next job.")
                if not link.get("simulated") and link.get("site") and link["site"] != site:
                    other = next((x["name"] for x in list_sites() if x["id"] == link["site"]), link["site"])
                    raise HTTPException(400, f"The robot computer is set up for {other}. Sync this site to it from Setup › Site, "
                                             f"or pick {other}.")
                if not link.get("simulated") and link.get("calibrated") is False:
                    raise HTTPException(400, "The site is not calibrated on the robot computer yet. Calibrate it in the Site step "
                                             "and send it to the robot computer, then start the job.")
                task = {"instruction": instruction or "Clean the whole surface.", "surface": s.get("surface") or "sealed_wood",
                        "zones": named, "forbidden_zones": sorted(no_wipe),
                        "max_passes_per_zone": int(jo["max_passes_per_zone"]), "max_steps": int(jo["max_steps"])}
                kw: dict[str, Any] = {}
                import inspect
                if "robot" in inspect.signature(RemoteSession).parameters:
                    kw["robot"] = str(s.get("robot") or "")
                st.live = RemoteSession(st.skill_url, st.camera_url, site, task, st.policy, st.verifier, st.runs_dir,
                                        require_approval=approval, client=st.client,
                                        audit=bool(jo["audit"]) and st.client is not None and st.policy == "nemotron" and _api_key_set(), recheck_all=bool(jo["recheck_all"]), **kw)
                st.live.start()
                return JSONResponse({**_live_state(), "notice": notice})
            from ..sim.scenarios import load_scenario
            spec = load_scenario(s["job_scenario"])            # job template of this site (tabletop and stains in simulation); the instruction is what the operator typed
            task = {**spec["task"], "max_passes_per_zone": int(jo["max_passes_per_zone"]), "max_steps": int(jo["max_steps"])}
            if no_wipe:
                task["forbidden_zones"] = sorted(set(task.get("forbidden_zones") or []) | no_wipe)
            if instruction:
                task["instruction"] = instruction
            if named:
                keep = [z for z in named if z not in set(task.get("forbidden_zones") or [])]
                if not keep:
                    raise HTTPException(400, f"{', '.join(named)}: no-wipe zone at this site")
                task["zones"] = keep
            spec = {**spec, "task": task, "operator_site": site}  # reports show the site this job belongs to (several sites can share one simulation scene)
            st.live = LiveSession(s["job_scenario"], st.policy, st.verifier, st.source, st.runs_dir,
                                  require_approval=approval, pace=float(body.get("pace", st.pace)), client=st.client, spec=spec,
                                  audit=bool(jo["audit"]) and st.client is not None and st.policy == "nemotron" and _api_key_set(), recheck_all=bool(jo["recheck_all"]))
            st.live.start()
        return JSONResponse({**_live_state(), "notice": notice})

    def _ctl(name: str) -> JSONResponse:
        if st.live is None:
            raise HTTPException(409, "No live job")
        fn = getattr(st.live, name)
        return JSONResponse({"ok": bool(fn()), **_live_state()})

    @app.post("/api/jobs/approve")
    def job_approve() -> JSONResponse:
        return _ctl("approve")

    @app.post("/api/jobs/reject")
    def job_reject() -> JSONResponse:
        return _ctl("reject")

    @app.post("/api/jobs/pause")
    def job_pause() -> JSONResponse:
        return _ctl("pause")

    @app.post("/api/jobs/resume")
    def job_resume() -> JSONResponse:
        return _ctl("resume")

    @app.post("/api/jobs/stop")
    def job_stop() -> JSONResponse:
        return _ctl("stop")

    @app.post("/api/jobs/estop")
    def job_estop() -> JSONResponse:
        if st.live is not None and st.live.running:
            r = _ctl("estop")
            st._link = (0.0, None)                           # the e-stop was delivered to the skill server synchronously: the next connection check sees stopped directly (title bar and Start follow immediately)
            return r
        if st.sim_robot or st.job is not None:                 # built-in simulation / video recording: with no job running there is no robot to stop
            raise HTTPException(409, "No job is running")
        # paired, no job running: still deliver the software stop to the robot computer (skill server POST /stop, new connection, 3 s timeout).
        # a job that has already ended is not re-recorded as an e-stop.
        import httpx

        from ..robot.server import auth_headers
        delivered = False
        try:
            with httpx.Client(timeout=3.0) as cl:
                delivered = cl.post(st.skill_url.rstrip("/") + "/stop", json={"reason": "operator E-STOP (console, no job running)"},
                                    headers=auth_headers()).status_code == 200
        except Exception:  # noqa: BLE001
            delivered = False
        if not delivered:
            return JSONResponse({"ok": False, "estop_delivered": False,
                                 "detail": "The E-stop did not reach the robot computer. Use the physical E-stop on site."}, status_code=502)
        st._link = (0.0, None)
        return JSONResponse({"ok": True, "estop_delivered": True})

    # ---- requests handed to human review ----
    def _job_run_dir() -> Path:
        if st.job is not None:
            return st.job.run_dir
        if st.live is not None and st.live.run_dir is not None:
            return Path(st.live.run_dir)
        raise HTTPException(404, "No job")

    @app.get("/api/requests")
    def requests_list() -> JSONResponse:
        return JSONResponse({"requests": read_requests(_job_run_dir())})

    @app.post("/api/requests/resolve")
    def requests_resolve(body: dict[str, Any] = Body(...)) -> JSONResponse:
        try:
            by = re.sub(r"\s+", " ", str(body.get("by") or "")).strip()[:60] or "operator"        # who handled it (audit trail); empty = operator
            r = resolve_request(_job_run_dir(), str(body.get("id")), str(body.get("note") or "").strip(), by)   # the note may be empty
        except KeyError as e:
            raise HTTPException(404, f"No request {e}") from e
        return JSONResponse({"ok": True, "resolved": r, "requests": read_requests(_job_run_dir())})

    # ---- reports ----
    def _report_dirs() -> list[Path]:
        if st.live is not None and st.live.run_dir is not None and st.live.finished:
            st.remember(Path(st.live.run_dir))
        ds = list(st.report_runs) + (list(st.live_history) if st.job is None else [])
        if st.job is not None:
            ds.append(st.job.run_dir)
        ds += [tk.run_dir for tk in st.robot_takes.values()]
        if st.live is not None and st.live.run_dir is not None and st.live.finished:
            ds.append(Path(st.live.run_dir))
        seen, out = set(), []
        for d in ds:
            if d.name not in seen and (d / "events.jsonl").is_file():
                seen.add(d.name)
                out.append(d)
        return out

    @app.get("/api/reports")
    def reports() -> JSONResponse:
        specs = load_platforms()
        rows = []
        for d in _report_dirs():
            plat = run_platform(d)
            r = report_row(d, specs[plat].label if plat in specs else plat)
            if r["status"] == "running" and not (st.live is not None and st.live.running and st.live.run_dir is not None
                                                 and Path(st.live.run_dir).name == d.name):
                r["status"] = "interrupted" if d in st.interrupted else "error"      # no task_end and not the running job: the software was interrupted last time, or the simulation thread failed
            sid = run_site(d)
            r["site"] = next((s["name"] for s in list_sites() if s["id"] == sid), r["site"])
            r["site_id"] = sid
            rows.append(r)
        rows.sort(key=lambda r: r["ended_utc"], reverse=True)
        return JSONResponse({"reports": rows})

    @app.get("/api/reports/{run_id}", response_class=HTMLResponse)
    def report(run_id: str) -> HTMLResponse:
        rd = next((d for d in _report_dirs() if d.name == run_id), None)
        if rd is None:
            raise HTTPException(404, "No such report")
        res = json.loads(_requests_file(rd).read_text(encoding="utf-8")) if _requests_file(rd).is_file() else {}
        interrupted = rd in st.interrupted
        key = run_id + "|" + json.dumps(res, sort_keys=True) + ("|i" if interrupted else "")
        if key not in _reports:
            from ..report.html import render_product_report
            plat = run_platform(rd)
            sid = run_site(rd)
            sname = next((s["name"] for s in list_sites() if s["id"] == sid), "")
            specs = load_platforms()
            _reports[key] = render_product_report(_load_run_isolated(rd), sname, specs[plat].label if plat in specs else plat, res,
                                                 interrupted=interrupted)
        return HTMLResponse(_reports[key])

    # ---- static resources ----
    @app.get("/robots/{key}.png")
    def robot_png(key: str) -> FileResponse:
        d = (REPO_ROOT / "assets" / "robots").resolve()
        f = (d / f"{key}.png").resolve()
        if d not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f, media_type="image/png")

    @app.get("/robots/photo/{key}.png")
    def robot_photo(key: str) -> FileResponse:
        """Image for the Robots page: use the licensed original (cut out on white) if there is one, otherwise the white-background render (assets/robots_photos/SOURCES.md)."""
        for d in (PHOTO_DIR.resolve(), (REPO_ROOT / "assets" / "robots").resolve()):
            f = (d / f"{key}.png").resolve()
            if d in f.parents and f.is_file():
                return FileResponse(f, media_type="image/png")
        raise HTTPException(404)

    @app.get("/brand/{name}.png")
    def brand_png(name: str) -> FileResponse:
        d = (REPO_ROOT / "assets" / "brand_icons").resolve()
        f = (d / f"{name}.png").resolve()
        if d not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f, media_type="image/png")

    @app.get("/brand/{name}.svg")
    def brand_svg(name: str) -> FileResponse:
        """Vector icon (used as the app icon in the title bar: sharp at any scale; lion badge)."""
        d = (REPO_ROOT / "assets" / "brand_icons").resolve()
        f = (d / f"{name}.svg").resolve()
        if d not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f, media_type="image/svg+xml")

    @app.get("/fonts/{name}/{file}")
    def font(name: str, file: str) -> FileResponse:
        f = (FONT_DIR / name / file).resolve()
        if FONT_DIR.resolve() not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f)

    return app


def preview_frame(st: "OperatorState", site: str) -> Optional[bytes]:
    """Camera frame before job creation in live mode: build the same simulation scene from this site job template (same spec as /api/jobs/start) and shoot one frame from the same display viewpoint used after job creation
    (the built-in UR3e platform uses the top-view camera).
    This is the tabletop the camera will see at the moment of job creation (stains already placed), not an illustration; cached per site and built only once."""
    if site in st.previews:
        return st.previews[site]
    s = next((x for x in list_sites() if x["id"] == site), None)
    if s is None or not s.get("job_scenario"):
        return None
    with st.lock:
        if site in st.previews:
            return st.previews[site]
        from ..common.config import current_site, use_site
        prev_site = current_site()
        try:
            from ..perception.sources import SimCameraSource, encode_jpeg
            from ..sim.scenarios import build, load_scenario
            sc = build(s["job_scenario"], spec=load_scenario(s["job_scenario"]))
            src = SimCameraSource(sc.robot, sc.grid, mode="topdown")
            sc.executor.survey()
            sc.add_stains()
            jpg = None
            from ..robot.platforms import get_platform
            platform = (sc.meta or {}).get("platform")
            if platform and not get_platform(platform).builtin:      # same viewpoint as the camera after job creation (session.LiveSession._attach_view)
                try:
                    from ..robot.platform_render import LiveView
                    view = LiveView(platform, (int(sc.robot.dirt.nx), int(sc.robot.dirt.ny)), size=(1280, 720))
                    try:
                        jpg = encode_jpeg(view.render(sc.robot.get_q(), view.dirt_rgb(sc.robot.dirt)), 85)
                    finally:
                        view.close()
                except Exception:  # noqa: BLE001  the display viewpoint failed: use the top view
                    jpg = None
            if jpg is None:
                jpg = encode_jpeg(src.rectified(), 85)
        except Exception:  # noqa: BLE001  if no frame can be obtained leave the page empty, which does not affect job creation
            jpg = None
        finally:
            use_site(prev_site or None)                       # build() switches the global site: switch back after shooting
        if jpg is not None:
            st.previews[site] = jpg
        return jpg


def _reference_version(site: str) -> int:
    """Reference image version (the page ?v= busts the cache): use the modification time of the captured reference.png if there is one, otherwise that of zones.yaml (the render also changes when the grid changes)."""
    d = SITE_DIR / site
    for f in (d / "reference.png", d / "zones.yaml", d / "site.yaml"):
        if f.is_file():
            return int(f.stat().st_mtime_ns // 1_000_000)
    return 0


def _sim_skill_check() -> dict[str, Any]:
    """Simulated deployment: robot computer = built-in local simulation, and the skill server item checks whether MuJoCo can be imported."""
    try:
        import mujoco
        return {"ok": True, "detail": f"built-in simulation · MuJoCo {mujoco.__version__}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": f"built-in simulation unavailable · {type(e).__name__}"}


def _sim_adapter_check(robot: str) -> dict[str, Any]:
    """Lightweight robot adapter check for the title bar (no scenario assembled): the platform is registered and, for a non-built-in platform, the model file (MuJoCo Menagerie) can be found."""
    from ..robot.platforms import load_platforms, menagerie_root
    spec = load_platforms().get(robot)
    if spec is None:
        return {"ok": False, "detail": f"no adapter for {robot or 'this robot'}"}
    if not spec.builtin:
        try:
            menagerie_root()
        except FileNotFoundError:
            return {"ok": False, "detail": "Robot model files not found · set CLEANPROOF_MENAGERIE to the MuJoCo Menagerie folder"}
    return {"ok": True, "detail": f"{spec.label} · ready"}


def _sim_overall(robot: str) -> dict[str, Any]:
    """Title-bar status of a simulation deployment = the simulation can start + the robot adapter is usable (a light version of the three checks in the wizard); if any one fails, the title bar does not show a green Robot connected."""
    sk = _sim_skill_check()
    if not sk["ok"]:
        return sk
    ad = _sim_adapter_check(robot)
    return sk if ad["ok"] else ad


def _is_sim_link(st: "OperatorState") -> bool:
    link = st.robot_link()
    return bool(link and link.get("simulated"))


def robot_computer_checks(skill_url: str, camera_url: str, robot: str) -> list[dict[str, Any]]:
    """Three checks of the paired robot computer (through the encrypted tunnel): skill server, camera service, robot. When the robot computer runs a simulation backend,
    camera frames are supplied by the skill server itself (/frame/*), and the camera item checks whether a frame can be fetched from there."""
    from ..agent.remote_job import SIM_BACKENDS
    from ..robot.platforms import load_platforms
    from .deploy_checks import _timed, check_adapter, check_camera, check_sim_camera, check_skill_server
    sk = {"id": "skill_server", "title": "Skill server", **_timed(lambda: check_skill_server(skill_url))}
    sim = ((sk.get("facts") or {}).get("backend") in SIM_BACKENDS)
    # a simulation backend has no camera service: count only the fixed camera /frame/fixed of the skill server (/frame/view is a display render, not a camera; in the video it reads "1 camera stream live")
    cam = {"id": "camera", "title": "Camera service", **_timed((lambda: check_sim_camera(skill_url)) if sim else (lambda: check_camera(camera_url)))}
    uncal = _only_uncalibrated(sk, skill_url)
    if uncal:
        # wizard: the real-robot site is not calibrated yet (skill server starts only in shadow mode and all skills are rejected); this is exactly what the next Site step does,
        # so the Connect step lets it through as "connected, awaiting calibration" (fallback: the page shows an amber check and the user can continue), otherwise a new robot computer would get stuck at Connect.
        # the deploy check page (/api/deploy/checks) still honestly reports not ready; job creation is always refused before calibration and sync (/api/jobs/start).
        sk = {**sk, "ok": True, "fallback": True,
              "detail": ("connected · token OK" if "token OK" in str(sk.get("detail")) else "connected") + " · site not calibrated yet"}
    ad = {"id": "adapter", "title": "Robot", **_timed(lambda: check_adapter(sk, robot))}
    specs = load_platforms()
    there = str((sk.get("facts") or {}).get("platform") or "")
    if there and there != robot:                               # the robot connected to the robot computer is not the one chosen in the wizard
        lab = lambda k: specs[k].label if k in specs else k  # noqa: E731
        ad.update(ok=False, fallback=False, detail=f"{lab(there)} on the robot computer, not {lab(robot)}")
    elif uncal and ad.get("ok"):
        ad.update(fallback=True, detail=f"{(ad.get('facts') or {}).get('label') or robot} · connected · calibrate the site in the next steps")
    return [sk, cam, ad]


def _only_uncalibrated(sk: dict[str, Any], skill_url: str) -> bool:
    """The skill server is reachable and the token is right, and the only reason it is not ready is that the site is uncalibrated (no e-stop, fault or other startup problem)."""
    f = sk.get("facts") or {}
    if sk.get("ok") or f.get("calibrated") is not False or f.get("stopped"):
        return False
    import httpx

    from ..robot.server import auth_headers
    try:
        h = httpx.get(skill_url.rstrip("/") + "/health", timeout=5.0, headers=auth_headers()).json()
    except Exception:  # noqa: BLE001
        return False
    others = [p for p in (h.get("startup_problems") or []) if not str(p).startswith("site not calibrated")]
    return h.get("calibrated") is False and not others and not h.get("fault") and not h.get("stopped")


def sim_robot_checks(robot: str) -> list[dict[str, Any]]:
    """Connection checks of the simulated deployment (same names and order as the three real-robot items): MuJoCo can be imported, the platform adapter can be assembled into a scene (really run once), the camera is the simulated top-view camera (reports the configured resolution).
    Failures are returned honestly."""
    from .deploy_checks import _timed

    def sim() -> dict[str, Any]:
        import mujoco
        return {"ok": True, "detail": f"MuJoCo {mujoco.__version__} · simulated robot computer"}

    def adapter() -> dict[str, Any]:
        from ..robot.platforms import build_platform_scene, load_platforms
        spec = load_platforms().get(robot)
        if spec is None:
            return {"ok": False, "detail": f"no adapter for {robot}"}
        if not spec.builtin:
            try:
                build_platform_scene(spec)
            except FileNotFoundError:                     # the UI is in English: give the operator an understandable hint without exposing the raw internal exception
                return {"ok": False, "detail": "Robot model files not found · set CLEANPROOF_MENAGERIE to the MuJoCo Menagerie folder"}
        return {"ok": True, "detail": f"{spec.label} · ready · software adapter (simulation)", "facts": {"platform": robot, "label": spec.label}}

    def camera() -> dict[str, Any]:
        from ..common.config import load_yaml
        r = load_yaml("sim.yaml").get("render") or {}
        return {"ok": True, "detail": f"simulated top-down camera · {r.get('width', 640)}×{r.get('height', 480)}"}
    return [{"id": "skill_server", "title": "Skill server", **_timed(sim)},
            {"id": "camera", "title": "Camera service", **_timed(camera)},
            {"id": "adapter", "title": "Robot", **_timed(adapter)}]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=None, help="site of this machine (configs/site/<site>/; omitted = the site last chosen in the wizard, setup.json, otherwise home_dining)")
    ap.add_argument("--job", default=None, help="for recording the video: recording directory of an already recorded run (containing frames.jsonl, meta.json)")
    ap.add_argument("--robot-take", action="append", default=[], help="Robots page: platform=recording directory (latest job of that robot)")
    ap.add_argument("--report-run", action="append", default=[], help="extra run directory in the report list")
    ap.add_argument("--deploy-record", default=None, help="saved deploy check (cleanproof/ui/deploy_checks.py --out)")
    ap.add_argument("--calls-log", default=None, help="model call log (default runs/llm_calls.jsonl)")
    ap.add_argument("--ledger", default=None, help="Token Factory ledger (default ~/.local/state/cleanproof/spend.json)")
    ap.add_argument("--policy", default="rules", help="planning policy of local live tasks (nemotron goes through Token Factory and incurs charges)")
    ap.add_argument("--verifier", default="cv", help="verifier of local live tasks (vlm = Cosmos 3 on the GPU workstation)")
    ap.add_argument("--source", default="topdown", choices=("topdown", "render"))
    ap.add_argument("--robot-host", default=None, help="robot computer hostname (default: read ~/.config/cleanproof/robot_host, written at pairing)")
    ap.add_argument("--sim-robot", action="store_true", help="simulated deployment (desktop edition when not paired): the robot computer is the built-in local MuJoCo simulation, and the wizard checks the local simulation")
    ap.add_argument("--skill-url", default=None, help=f"skill server on the paired robot computer (through the tunnel; default {DEFAULT_SKILL_URL}, changeable on the settings page)")
    ap.add_argument("--camera-url", default=None, help=f"camera service (default {DEFAULT_CAMERA_URL})")
    ap.add_argument("--pace", type=float, default=0.0, help="simulation pace of live tasks: 0 = run as fast as possible, 1 = real time, 3 = triple real time (desktop edition)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--allow-host", action="append", default=[], help="Host headers allowed besides 127.0.0.1 / localhost (add one when you reach the app by host name or LAN IP)")
    ap.add_argument("--port", type=int, default=8092)
    args = ap.parse_args()
    args.site = args.site or (None if args.job else setup_site_choice()) or "home_dining"    # video recording (--job) does not read setup.json
    host = args.robot_host or OperatorState.paired_host()
    for opt in ("skill_url", "camera_url"):                   # the shared token is sent only to the local tunnel port or the paired robot computer
        if getattr(args, opt):
            try:
                setattr(args, opt, _check_url(getattr(args, opt), host))
            except ValueError as e:
                ap.error(f"--{opt.replace('_', '-')}: {e}")
    use_site(args.site)
    job = Take("job", args.job) if args.job else None
    robot_takes = {}
    for x in args.robot_take:
        k, d = x.split("=", 1)
        robot_takes[k] = Take(k, d)
    if job is not None:                                       # the robot of the recorded job also uses this job frames on the Robots page
        jp = str((job.meta.get("robot") or {}).get("platform") or "spot_arm")
        robot_takes.setdefault(jp, job)
    use_site(args.site)                                       # Take switches to the site registered in the recording while reading it, so switch back to the local site here
    client = None
    if args.policy == "nemotron" or args.verifier == "vlm":
        from ..common.llm_client import LLMClient
        client = LLMClient()
    st = OperatorState(args.site, job, robot_takes, Path(args.deploy_record) if args.deploy_record else None,
                       Path(args.calls_log) if args.calls_log else None, Path(args.ledger) if args.ledger else None,
                       [Path(x) for x in args.report_run], args.policy, args.verifier, args.source, client=client, robot_host=args.robot_host,
                       sim_robot=args.sim_robot, pace=args.pace, skill_url=args.skill_url, camera_url=args.camera_url)
    import uvicorn
    print(f"CleanProof → http://{args.host}:{args.port}/  site {args.site} · {'recorded job ' + job.run_dir.name if job else 'live'}")
    allow = tuple(args.allow_host) + (() if args.host in ("0.0.0.0", "::", "") else (args.host,))
    if args.host in ("0.0.0.0", "::", "") or args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"WARNING: CleanProof is listening on {args.host or 'all interfaces'}. Anyone on that network can reach the buttons "
              "(start, stop, E-stop, settings, API key). There is no login. Use this only on a network you trust, "
              "and pass --allow-host <name-or-ip> for the address you open in the browser.", flush=True)
    uvicorn.run(create_operator_app(st, allow), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
