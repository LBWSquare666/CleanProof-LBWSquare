"""Client of the skill server: maps the executor interface TaskRunner needs (survey / closeup / wipe / startup_problems) to HTTP requests.

The agent process has no RTDE interface at all; it can only request whitelisted skills through the skill server (project rule: real-robot red line).
Network failures are not retried: the skill may already be running on the robot side, and sending it again is unsafe. Any failure returns status=aborted, and the loop hands over to a human accordingly.
Reasons shown to the operator (the software UI is in English) are always English; the skill server's original text only goes to the log (skill event records, this module's logger).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional, Sequence

import httpx

from ..common.zones import ZoneGrid
from .server import TOKEN_HEADER, read_token
from .skills import SkillOutcome

log = logging.getLogger("cleanproof.robot.remote")

MOVE_TIMEOUT_S = 180.0      # survey / close look (free-space move at dry_run speed ×0.3)
STOP_TIMEOUT_S = 3.0
IDEMPOTENT_SKILLS = ("survey", "inspect_closeup")      # skills that only observe and never touch the surface; they may be retried briefly when the connection drops
IDEMPOTENT_RETRIES, RETRY_BACKOFF_S = 2, 0.5


def refusal_reason(code: int, raw: str, skill: str) -> str:
    """skill server rejected a skill (not 200) → English reason shown to the operator. Distinguished by the original rejection text in server.py; the original text is also logged."""
    raw = str(raw or "")
    if code == 401:
        return "the robot computer rejected this computer's access token (pair the computers again)"
    if code == 409:
        return "the robot computer is busy with another skill"
    if code == 423:
        if "site not calibrated" in raw:
            return "this site is not calibrated on the robot computer yet"
        if "software stop" in raw:
            return "the robot computer is stopped (E-stop); reset it on site"
        if "fault" in raw:
            return "the robot computer has an unreset fault; check the robot and reset it on site"
        if "authorization" in raw or "arm" in raw.lower():
            return "the robot is not armed for motion (or the arming expired); arm it on site"
        return "the robot computer refused motion; check the robot on site"
    if code == 422:
        if "forbidden zone" in raw:
            return "the robot computer does not know some of the no-wipe zones"
        return f"the robot computer rejected the {skill} request as invalid"
    if code >= 500:
        return f"the robot computer failed while running {skill}; check the robot on site"
    return f"the robot computer could not run {skill}"


STARTUP_PROBLEMS = (                          # original text fragments of skills.SkillExecutor.startup_problems → English
    ("robot_mode=", "the robot is not in running mode"),
    ("speed slider", "the teach-pendant speed slider is not at 100%"),
    ("TCP offset", "the tool (TCP) offset on the teach pendant does not match the configured tool"),
    ("Payload", "the payload on the teach pendant does not match the configured tool"),
    ("startup check failed", "the robot start-up check could not run"),
)


def startup_problem_text(p: str) -> str:
    for key, text in STARTUP_PROBLEMS:
        if key in p:
            return text
    return "a robot start-up check failed (details in the robot computer log)"


def readiness_problems(h: dict[str, Any]) -> list[str]:
    """skill server /health → reasons a job cannot start (English, for the operator; empty = can start)."""
    out = []
    if h.get("calibrated") is False:                 # uncalibrated real-robot site (the skill server starts only in shadow, all skills are rejected)
        out.append("this site is not calibrated on the robot computer yet; calibrate it in Setup (Site step), then start the job")
    if h.get("stopped"):
        out.append("the robot computer is stopped (E-stop); reset it on site before the next job")
    if h.get("fault"):
        out.append("the robot computer has an unreset fault; check the robot and reset it on site before the next job")
    if h.get("busy"):
        out.append(f"the robot computer is still running a skill ({h['busy']}); try again when it is idle")
    arm = h.get("arming")
    if arm is not None and h.get("mode") != "shadow" and not arm.get("armed"):
        out.append("the robot is not armed for motion; arm it on site, then start the job")
    for p in h.get("startup_problems") or []:
        if h.get("calibrated") is False and str(p).startswith("site not calibrated"):
            continue                                 # already said above
        log.warning("robot start-up check failed: %s", p)          # the original text goes only to the log
        out.append(f"robot start-up check failed: {startup_problem_text(str(p))}")
    return out


class RemoteRobot:
    """TaskRunner only uses now() and backend_name; the emergency stop is forwarded to /stop."""

    control_dt = 0.004

    def __init__(self, ex: "RemoteSkillExecutor", backend_name: str, label: str):
        self._ex = ex
        self.backend_name = backend_name
        self.label = label

    def now(self) -> float:
        try:
            return float(self._ex.http.get("/now", timeout=5.0).json()["now"])
        except Exception:  # noqa: BLE001
            return float("nan")      # do not mix with the local clock: if the robot time is unknown, record NaN honestly

    def emergency_stop(self, reason: str = "operator stop") -> None:
        self._ex.stop(reason)


class RemoteSkillExecutor:
    def __init__(self, base_url: str, forbidden_zones: Sequence[str] = (), connect_timeout_s: float = 5.0,
                 transport: Any = None):
        self.base_url = base_url.rstrip("/")
        tok = read_token()
        self._headers = {TOKEN_HEADER: tok} if tok else {}
        self._transport = transport
        self.http = httpx.Client(base_url=self.base_url, timeout=connect_timeout_s, transport=transport,
                                 headers=self._headers or None)
        try:
            cfg = self.http.get("/config").raise_for_status().json()
            self.server_cfg = cfg
            self.cfg = cfg["robot"]
            self.grid = ZoneGrid(cfg["zones"])
            self.forbidden = set(forbidden_zones)
            self.fault: Optional[str] = None
            h = self.health()
        except BaseException:
            self.http.close()                        # unreachable / wrong token: do not leave an open connection pool behind
            raise
        self.robot = RemoteRobot(self, f"remote:{h['backend']}", f"{h['label']} [mode={h['mode']}] via {self.base_url}")
        self.mode = h["mode"]

    # ---------- state ----------
    def health(self) -> dict[str, Any]:
        err: Optional[Exception] = None
        for i in range(1 + IDEMPOTENT_RETRIES):          # GET /health is idempotent: retry briefly when the connection drops
            try:
                return self.http.get("/health").raise_for_status().json()
            except httpx.TransportError as e:
                err = e
                if i < IDEMPOTENT_RETRIES:
                    time.sleep(RETRY_BACKOFF_S * (i + 1))
        assert err is not None
        raise err

    def startup_problems(self) -> list[str]:
        try:
            h = self.health()
        except Exception as e:  # noqa: BLE001
            return [f"robot computer not reachable at {self.base_url} ({type(e).__name__})"]
        return readiness_problems(h)

    def stop(self, reason: str = "operator stop") -> bool:
        """Software stop (POST /stop), returns whether it was delivered. Sent over a separate connection: it gets through immediately even while a skill request occupies the main connection (a wipe can take tens of seconds);
        it also works after this object's close(). When it cannot be delivered, say so loudly: the software stop did not take effect, only the physical emergency stop works."""
        try:
            if self._transport is not None:              # injected transport (tests): closing a temporary client would close it too, so use the main connection
                ok = self.http.post("/stop", json={"reason": reason}, timeout=STOP_TIMEOUT_S).status_code == 200
            else:
                with httpx.Client(base_url=self.base_url, headers=self._headers or None,
                                  timeout=httpx.Timeout(STOP_TIMEOUT_S, connect=STOP_TIMEOUT_S)) as c:
                    ok = c.post("/stop", json={"reason": reason}).status_code == 200
        except Exception as e:  # noqa: BLE001
            ok = False
            reason = f"{reason} ({type(e).__name__}: {e})"
        if not ok:
            print(f"\n!!! software stop was not delivered to the skill server: {reason}\n!!! to stop the machine press the physical E-stop / teach-pendant pause now\n", flush=True)
        return ok

    def reset_fault(self) -> None:
        self.fault = None     # a remote reset can only be done by on-site staff via POST /reset (with a name), the agent never does it

    # ---------- skills ----------
    def _skill(self, body: dict[str, Any]) -> SkillOutcome:
        body = {**body, "forbidden_zones": sorted(self.forbidden), "request_id": uuid.uuid4().hex[:12]}
        name = body["skill"]
        t0 = time.monotonic()
        if name in ("wipe_zone", "spot_scrub"):     # no shorter than the server's worst-case time: zone_timeout_s per zone, plus margin for moves
            timeout = len(body.get("zone_ids", [])) * float(self.cfg["skills"]["zone_timeout_s"]) + 2 * MOVE_TIMEOUT_S
        else:
            timeout = MOVE_TIMEOUT_S
        retries = IDEMPOTENT_RETRIES if name in IDEMPOTENT_SKILLS else 0       # contact motion skills such as wiping are never retried
        try:
            for i in range(1 + retries):
                try:
                    r = self.http.post("/skill", json=body, timeout=httpx.Timeout(timeout, connect=5.0))
                    break
                except httpx.TransportError as e:
                    if i >= retries or isinstance(e, httpx.ReadTimeout):    # a read timeout means the skill may still be running: do not resend
                        raise
                    log.warning("lost the connection during %s, retrying (%d/%d)", name, i + 1, retries)
                    time.sleep(RETRY_BACKOFF_S * (i + 1))      # resend with the same request_id; if the server is still running it answers 409, so nothing runs twice
        except Exception as e:  # noqa: BLE001
            return SkillOutcome(name, "aborted", f"lost the connection to the robot computer during {name} ({type(e).__name__}); "
                                "the skill may still be running there: check the robot on site", zone_ids=list(body.get("zone_ids", [])),
                                duration_s=time.monotonic() - t0)
        if r.status_code != 200:
            try:
                msg = r.json().get("error") or r.json().get("detail")
            except Exception:  # noqa: BLE001
                msg = r.text[:300]
            status = "rejected" if r.status_code == 422 else "aborted"
            log.warning("skill server refused %s: HTTP %s %s", name, r.status_code, msg)
            return SkillOutcome(name, status, f"{refusal_reason(r.status_code, str(msg), name)} (HTTP {r.status_code})",
                                zone_ids=list(body.get("zone_ids", [])), duration_s=time.monotonic() - t0,
                                events=[{"kind": "skill_server_refusal", "http_status": r.status_code, "detail": str(msg)[:500]}])
        d = r.json()
        out = SkillOutcome(d["skill"], d["status"], d.get("reason", ""), list(d.get("zone_ids", [])), d.get("force", {}),
                           d.get("events", []), float(d.get("duration_s", 0.0)), d.get("trace", {}) or {})
        if out.status == "fault":
            self.fault = out.reason
        return out

    def survey(self) -> SkillOutcome:
        return self._skill({"skill": "survey"})

    def closeup(self, zone_id: str) -> SkillOutcome:
        return self._skill({"skill": "inspect_closeup", "zone_ids": [zone_id]})

    def wipe(self, zone_ids: Sequence[str], force_n: float, pattern: str, passes: int = 1,
             bbox_norm: Optional[list[float]] = None) -> SkillOutcome:
        body: dict[str, Any] = {"zone_ids": list(zone_ids), "force_n": float(force_n), "passes": int(passes)}
        if bbox_norm is not None:
            body.update(skill="spot_scrub", bbox_norm=[float(v) for v in bbox_norm])
        else:
            body.update(skill="wipe_zone", pattern=pattern)
        return self._skill(body)

    def close(self) -> None:
        self.http.close()
