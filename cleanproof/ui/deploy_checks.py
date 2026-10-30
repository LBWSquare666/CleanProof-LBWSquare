"""Deployment checks (the "Deploy" page of the software): actually check device by device whether services are ready; results can be saved to disk and replayed from the saved file when recording the video.

Devices and checks (all read-only HTTP GETs, no inference requests, no charges):
  - GPU workstation (the on-site GPU computer, called "GPU workstation" in the UI): CleanProof orchestration service (config and version of this process), Cosmos 3 inference service (vLLM, OpenAI-compatible /models endpoint);
  - Cloud: Nebius Token Factory (whether the key exists is reported only as OK / MISSING, the key is never printed; /models lists the planning and audit models);
  - Robot computer: skill server (/health, with the shared token; it is the only process that can send motion to the robot), camera service (/health);
  - Robot: robot state read back through the skill server software adapter (safety mode, startup checks).

  python -m cleanproof.ui.deploy_checks --skill-url http://127.0.0.1:8100 --camera-url http://127.0.0.1:8101 --out runs/deploy/checks.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from ..common.config import REPO_ROOT, current_site, load_secrets, load_yaml

# command that starts the matching service on each device (entry points that really exist in the repo, shown as is on the Deploy page)
COMMANDS = {
    "orchestrator": "python -m cleanproof.ui.operator --site {site}",
    "cosmos": "scripts/server/serve_cosmos.sh",
    "tokenfactory": "NEBIUS_API_KEY in ~/.config/cleanproof/secrets.env",
    "skill_server": "python -m cleanproof.robot.server --backend {backend} --port 8100",
    "camera": "python -m cleanproof.perception.camera_server --port 8101",
    "adapter": "configs/platforms.yaml · {platform}",
}


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _timed(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001  a failed check must be recorded honestly, not raised
        out = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:200]}
    out["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
    return out


def _token_headers() -> dict[str, str]:
    from ..robot.server import auth_headers
    return auth_headers()


def check_orchestrator() -> dict[str, Any]:
    models = load_yaml("models.yaml")["roles"]
    plan_ok = models["decision"]["backend"] == "tokenfactory" and models["audit"]["backend"] == "tokenfactory"
    return {"ok": plan_ok, "detail": f"version {_git_rev() or 'unknown'} · site {current_site() or 'default'}",
            "facts": {"planner": models["decision"]["model"], "auditor": models["audit"]["model"], "verifier": models["verifier"]["model"]}}


def check_cosmos(base_url: Optional[str] = None, timeout: float = 5.0) -> dict[str, Any]:
    cfg = load_yaml("models.yaml")
    b = cfg["backends"][cfg["roles"]["verifier"]["backend"]]
    url = (base_url or os.environ.get("CP_BASE_URL_LOCAL_VLLM") or b["base_url"]).rstrip("/") + "/models"
    r = httpx.get(url, timeout=timeout)
    r.raise_for_status()
    ids = [m.get("id") for m in r.json().get("data", [])]
    want = cfg["roles"]["verifier"]["model"]
    return {"ok": want in ids, "detail": f"{want} served" if want in ids else f"{want} not in {ids}", "facts": {"models": ids}}


def probe_api_key(key: str, timeout: float = 8.0) -> tuple[Optional[bool], str]:
    """Verify a candidate key in memory only (nothing written to disk, no environment variable changed): (True, "") accepted · (False, reason) rejected · (None, reason) could not be verified (not reachable, etc.)."""
    b = load_yaml("models.yaml")["backends"]["tokenfactory"]
    try:
        r = httpx.get(b["base_url"].rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return None, f"not reachable ({type(e).__name__})"
    if r.status_code in (401, 403):
        return False, "rejected"
    if r.status_code >= 400:
        return None, f"HTTP {r.status_code}"
    return True, ""


def check_tokenfactory(timeout: float = 8.0) -> dict[str, Any]:
    load_secrets()
    cfg = load_yaml("models.yaml")
    b = cfg["backends"]["tokenfactory"]
    key = os.environ.get(b["api_key_env"], "")
    if not key:
        return {"ok": False, "detail": "API key MISSING"}
    r = httpx.get(b["base_url"].rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    r.raise_for_status()
    ids = {m.get("id") for m in r.json().get("data", [])}
    need = [cfg["roles"]["decision"]["model"], cfg["roles"]["audit"]["model"]]
    missing = [m for m in need if m not in ids]
    return {"ok": not missing, "detail": "API key OK · " + ("planner and auditor available" if not missing else f"missing {missing}"),
            "facts": {"available": [m for m in need if m in ids]}}


def check_skill_server(url: str, timeout: float = 5.0) -> dict[str, Any]:
    hdr = _token_headers()
    r = httpx.get(url.rstrip("/") + "/health", headers=hdr, timeout=timeout)
    if r.status_code == 401:
        return {"ok": False, "detail": "token rejected"}
    r.raise_for_status()
    h = r.json()
    ok = not h.get("startup_problems") and not h.get("fault") and not h.get("stopped") and h.get("calibrated") is not False
    if h.get("calibrated") is False:                       # real-robot site not calibrated (skill server starts only in shadow mode, all skills are rejected)
        why = " · site not calibrated · calibrate it in the Site step"
    elif h.get("stopped"):
        why = " · stopped (E-stop) · reset it on site"
    elif h.get("fault"):
        why = " · unreset fault · reset it on site"
    elif h.get("startup_problems"):
        why = f" · start-up check failed: {h['startup_problems'][0]}"
    else:
        why = ""
    return {"ok": ok, "detail": ("connected · token OK" if hdr else "connected") + why,
            "facts": {"backend": h.get("backend"), "mode": h.get("mode"), "label": h.get("label"), "git": h.get("git"),
                      "safety_mode": h.get("safety_mode"), "robot_mode": h.get("robot_mode"), "platform": h.get("platform"),
                      "stopped": bool(h.get("stopped")), "calibrated": h.get("calibrated") is not False}}


def check_camera(url: str, timeout: float = 5.0) -> dict[str, Any]:
    r = httpx.get(url.rstrip("/") + "/health", headers=_token_headers(), timeout=timeout)
    r.raise_for_status()
    cams = r.json().get("cameras") or {}
    live = [k for k, v in cams.items() if (v.get("frames") or 0) > 0]
    return {"ok": bool(live), "detail": f"{len(live)} camera stream{'s' if len(live) != 1 else ''} live", "facts": {"cameras": sorted(cams)}}


def check_sim_camera(skill_url: str, timeout: float = 10.0) -> dict[str, Any]:
    """When the robot computer runs a simulation backend there is no camera service: the camera stream is the simulated fixed camera of the skill server (/frame/fixed).
    /frame/view is a display render for the software and does not count as a camera stream."""
    r = httpx.get(skill_url.rstrip("/") + "/frame/fixed", headers=_token_headers(), timeout=timeout)
    live = r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    return {"ok": live, "detail": f"{int(live)} camera stream{'' if live else 's'} live", "facts": {"cameras": ["fixed"] if live else []}}


def check_adapter(skill: dict[str, Any], platform: str) -> dict[str, Any]:
    from ..robot.platforms import load_platforms
    spec = load_platforms().get(platform)
    f = skill.get("facts") or {}
    actual = str(f.get("platform") or "")
    mismatch = bool(actual) and actual != platform          # the robot connected to the robot computer is not the one selected for this site
    ok = bool(skill.get("ok")) and f.get("safety_mode") == 1 and not mismatch
    via = "RTDE" if platform.startswith("ur") and f.get("backend") in ("ursim", "ur3e") else "software adapter"
    label = spec.label if spec else platform
    if ok:
        detail = f"{label} · ready · {via}"
    elif mismatch:
        aspec = load_platforms().get(actual)
        detail = f"{aspec.label if aspec else actual} on the robot computer, not {label}"
    elif f.get("calibrated") is False:
        detail = f"{label} · not ready · site not calibrated"
    elif f.get("stopped"):
        detail = f"{label} · not ready · stopped (E-stop)"
    else:
        detail = "robot not ready"
    return {"ok": ok, "detail": detail,
            "facts": {"platform": platform, "label": spec.label if spec else platform, "via": via}}


def run_checks(skill_url: str = "http://127.0.0.1:8100", camera_url: str = "http://127.0.0.1:8101", platform: str = "spot_arm",
               cosmos_url: Optional[str] = None) -> dict[str, Any]:
    site = current_site() or "default"
    checks: list[dict[str, Any]] = []

    def add(device: str, cid: str, title: str, res: dict[str, Any], **fmt: str) -> dict[str, Any]:
        item = {"device": device, "id": cid, "title": title, "command": COMMANDS[cid].format(site=site, **fmt), **res}
        checks.append(item)
        return item

    add("edge", "orchestrator", "CleanProof orchestrator", _timed(check_orchestrator))
    add("edge", "cosmos", "Cosmos 3 inference server", _timed(lambda: check_cosmos(cosmos_url)))
    add("cloud", "tokenfactory", "Nebius Token Factory", _timed(check_tokenfactory))
    sk = add("robot_computer", "skill_server", "Skill server", _timed(lambda: check_skill_server(skill_url)), backend="<backend>")
    backend = (sk.get("facts") or {}).get("backend") or "<backend>"
    sk["command"] = COMMANDS["skill_server"].format(backend=backend)
    sim = backend in ("mujoco", "rtde_fake")                 # simulation backend: the camera stream is provided by the skill server itself
    add("robot_computer", "camera", "Camera service", _timed(lambda: check_sim_camera(skill_url) if sim else check_camera(camera_url)))
    add("robot", "adapter", "Robot", _timed(lambda: check_adapter(sk, platform)), platform=platform)
    return {"site": site, "ts": datetime.now(timezone.utc).isoformat(), "git": _git_rev(), "host_role": "GPU workstation",
            "ok": all(c.get("ok") for c in checks), "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-url", default="http://127.0.0.1:8100")
    ap.add_argument("--camera-url", default="http://127.0.0.1:8101")
    ap.add_argument("--platform", default="spot_arm")
    ap.add_argument("--cosmos-url", default=None)
    ap.add_argument("--site", default=None, help="CLEANPROOF_SITE (defaults to the environment variable)")
    ap.add_argument("--out", default=None, help="save the results as JSON (replayed by the software Deploy page)")
    args = ap.parse_args()
    if args.site:
        from ..common.config import use_site
        use_site(args.site)
    res = run_checks(args.skill_url, args.camera_url, args.platform, args.cosmos_url)
    for c in res["checks"]:
        print(f"{'OK ' if c.get('ok') else 'BAD'} {c['device']:15s} {c['title']:28s} {c.get('detail', '')}  ({c['latency_ms']} ms)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"→ {args.out}")
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
