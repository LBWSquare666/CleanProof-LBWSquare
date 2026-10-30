"""Console sessions: replay a run directory; or run a job live and accept operator approve / pause / stop / e-stop:
  - LiveSession: the robot computer is the built-in local MuJoCo simulation (simulated deployment when the desktop edition is not paired);
  - RemoteSession: connects to the paired robot computer (skill server + camera service, over an encrypted tunnel); skills execute on the robot side.

The console process never creates an RTDE interface to a real robot: the real robot only runs whitelisted skills through the skill server, and the e-stop button calls the skill server /stop;
the physical e-stop on the teach pendant and the e-stop button remain authoritative (project convention: real-robot red line).
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from ..agent.loop import TaskRunner
from ..common.config import runs_root
from ..common.schemas import Decision
from ..report.rundata import RunData, load_run
from ..sim.run import POLICIES, VERIFIERS, run_sim_scenario


log = logging.getLogger("cleanproof.session")


def write_error_summary(runner: Any, reason: Optional[str]) -> None:
    """On an error midway (log write failed, disk full) try to write summary.json, so the job shows as error in Reports instead of vanishing; if that is impossible, leave it."""
    if runner is None:
        return
    try:
        st = runner.state
        runner.log.write_json("summary.json", {"run_id": runner.log.run_id, "status": "error", "reason": reason or "", "policy": runner.policy.name,
                                               "verifier": runner.ver.name, "task": runner.task.model_dump(), "targets": st.targets, "steps": st.step,
                                               "initial_status": st.initial_status, "passes": st.passes, "human_requests": st.human_requests})
    except Exception:  # noqa: BLE001
        log.warning("could not write the error summary")


def operator_error(e: BaseException) -> str:
    """Exception in a thread → one sentence for the operator. The raw exception class name, full paths and traceback go only to the log (log.exception), not to the UI."""
    import errno
    msg = str(e)
    if isinstance(e, PermissionError):
        return "CleanProof cannot write the job record because of folder permissions. Check that the runs folder is writable, then start the job again."
    if isinstance(e, OSError) and getattr(e, "errno", None) in (errno.ENOSPC, errno.EFBIG, errno.EDQUOT):
        return "The disk is full or a size limit was reached while saving the job record. Free some space, then start the job again."
    if isinstance(e, FileNotFoundError) and ("Menagerie" in msg or "menagerie" in msg):
        return "The robot model files were not found. Set CLEANPROOF_MENAGERIE to the MuJoCo Menagerie folder, then start the job again."
    if isinstance(e, FileNotFoundError):
        return "A file the job needs is missing. See the CleanProof log for the file name."
    if isinstance(e, (ConnectionError, TimeoutError)):
        return "The connection was lost during the job. Check the network and the robot computer, then start the job again."
    return f"The job stopped because of a software error ({type(e).__name__}). The details are in the CleanProof log."


class ReplaySession:
    """Offline replay: read-only access to an existing run directory."""

    mode = "replay"

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        if not (self.run_dir / "events.jsonl").is_file():
            raise FileNotFoundError(f"not a run dir (no events.jsonl): {self.run_dir}")

    def load(self) -> RunData:
        return load_run(self.run_dir)

    def status(self) -> dict[str, Any]:
        return {"mode": self.mode, "run_dir": str(self.run_dir), "live": None}

    def frame(self) -> Optional[bytes]:
        return None


class LiveSession:
    """Live mode: a background thread runs run_sim_scenario, and the console approves / rejects / e-stops through this object.

    - With require_approval=True every decision that moves the robot waits for the operator to click Approve/Reject (the approve hook blocks the simulation thread);
    - estop() sets the TaskRunner stop flag and calls MujocoRobot.emergency_stop() at the same time; the running skill exits with
      fatal on the next tick, and the loop finishes with status=estopped and writes summary.json;
    - With pace>0 simulation time is throttled to about pace times real time (pace=1 is real time), convenient for demos; 0 means run as fast as possible.
    """

    mode = "live"

    def __init__(self, scenario: str, policy: str = "rules", verifier: str = "cv", source: str = "topdown",
                 run_root: Optional[Path] = None, require_approval: bool = True, pace: float = 0.0,
                 client: Any = None, audit: bool = False, spec: Optional[dict[str, Any]] = None, recheck_all: bool = True,
                 site: Optional[str] = None):
        if policy not in POLICIES or verifier not in VERIFIERS:
            raise ValueError(f"policy ∈ {POLICIES}, verifier ∈ {VERIFIERS}")
        self.scenario, self.policy, self.verifier, self.source = scenario, policy, verifier, source
        self.run_root = Path(run_root) if run_root else runs_root()
        self.require_approval, self.pace, self.client, self.audit = require_approval, float(pace), client, audit
        self.spec = spec                                      # scenario spec override (jobs are created in the software from the instruction the operator typed)
        self.recheck_all = recheck_all                        # Jobs on the settings page: "re-check all zones after every wipe"
        # the site this job belongs to (several sites can share one simulation scene): written to site in reference_meta.json, which reports and Today use for display
        self.site = site or (spec or {}).get("operator_site") or None
        self.runner: Optional[TaskRunner] = None
        self.robot: Any = None
        self.src: Any = None
        self.run_dir: Optional[Path] = None
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[str] = None
        self.estopped = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pending: Optional[dict[str, Any]] = None       # {"step", "decision"}
        self._verdict: Optional[bool] = None
        self._wake = threading.Event()
        self.started_at: Optional[float] = None
        self._view: Any = None                                # camera view (same viewpoint as the software segment in the video, see _attach_view); False = this platform has none
        self._view_jpg: Optional[bytes] = None
        self._view_t = 0.0
        self._view_key: Any = None
        self._view_tick: Any = None

    # ---------- lifecycle ----------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def finished(self) -> bool:
        return self._thread is not None and not self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("session already started")
        self.started_at = time.time()
        self._thread = threading.Thread(target=self._target, name="cleanproof-live", daemon=True)
        self._thread.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def _target(self) -> None:
        try:
            self.result = run_sim_scenario(self.scenario, self.policy, self.verifier, self.source, self.run_root,
                                           client=self.client, audit=self.audit,
                                           approve=self._approve if self.require_approval else None,
                                           on_runner=self._on_runner, spec=self.spec, recheck_all=self.recheck_all,
                                           force_site=self.site)          # the force table is read once for this job site (not the scenario template site)
        except Exception as e:  # an exception in a thread must not be lost: details go to the log, the UI gets one readable reason; try to write the summary (if the disk is full or permissions fail, give up)
            log.exception("job thread failed")
            self.error = operator_error(e)
            self._write_error_summary()
        finally:
            self._close_view()                                # the renderer is created in the simulation thread and closed here too (the GL context is bound to the thread)
            self._wake.set()

    def _write_error_summary(self) -> None:
        write_error_summary(self.runner, self.error)

    def _close_view(self) -> None:
        if self.robot is not None and self._view_tick is not None and self._view_tick in self.robot.tick_hooks:
            self.robot.tick_hooks.remove(self._view_tick)
        if self._view not in (None, False):
            try:
                self._view.close()
            except Exception:  # noqa: BLE001
                pass
            self._view = False                                # the last frame (_view_jpg) is kept for the page after the job ends

    def _on_runner(self, runner: TaskRunner, sc: Any) -> None:
        self.runner, self.robot, self.src = runner, sc.robot, runner.src
        self.run_dir = runner.log.dir
        if self.site:                                         # run_sim_scenario has already written reference_meta.json: add the site
            import json
            p = runner.log.dir / "reference_meta.json"
            try:
                meta = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
            except (OSError, ValueError):
                meta = {}
            runner.log.write_json("reference_meta.json", {**meta, "site": self.site})
        if self.pace > 0:
            self._throttle(sc.robot)
        self._attach_view(sc)

    VIEW_EVERY_S = 0.2                                        # camera view refresh interval (wall-clock seconds)

    def _attach_view(self, sc: Any) -> None:
        """Camera view of the console: the display viewpoint of the platform scene (cleanproof/robot/platform_render.LiveView, same rendering as the software segment in the video),
        rendered in the simulation thread every VIEW_EVERY_S from the current joint angles and stain map, encoded as a cached JPEG that frame() returns directly.
        Read-only on state, does not change the simulation; the top-view camera used for perception is unaffected. UR3e (built-in platform) has no such viewpoint and falls back to the top view."""
        platform = (getattr(sc, "meta", None) or {}).get("platform")
        try:
            from ..robot.platforms import get_platform
            if not platform or get_platform(platform).builtin:
                self._view = False
                return
        except Exception:  # noqa: BLE001
            self._view = False
            return
        robot = sc.robot

        def tick() -> None:
            now = time.time()
            if self._view is False or now - self._view_t < self.VIEW_EVERY_S:
                return
            self._view_t = now
            try:
                if self._view is None:
                    from ..robot.platform_render import LiveView
                    self._view = LiveView(platform, (int(robot.dirt.nx), int(robot.dirt.ny)), size=(1280, 720))
                lv = robot.dirt.level
                key = (float(lv.sum()), float(lv.max()))
                rgb = self._view.dirt_rgb(robot.dirt) if key != self._view_key else None
                self._view_key = key
                from ..perception.sources import encode_jpeg
                self._view_jpg = encode_jpeg(self._view.render(robot.get_q(), rgb), 85)
            except Exception:  # noqa: BLE001  failing to produce the view does not affect the task: fall back to the top view
                self._view, self._view_jpg = False, None

        self._view_tick = tick
        robot.tick_hooks.append(tick)
        tick()

    def _throttle(self, robot: Any) -> None:
        """Keep simulation time from running faster than pace times wall clock: after wait_tick, sleep a bit if ahead."""
        orig = robot.wait_tick
        base = [time.time(), robot.now()]                     # alignment point of wall clock and simulation time

        def wait_tick() -> None:
            orig()
            ahead = (robot.now() - base[1]) / self.pace - (time.time() - base[0])
            if ahead > 0.002:
                time.sleep(min(ahead, 0.05))
            elif ahead < -0.1:                                # fell behind after waiting for approval / pause: realign from now instead of catching up all at once
                base[0], base[1] = time.time(), robot.now()

        robot.wait_tick = wait_tick

    # ---------- approval ----------
    def _approve(self, decision: Decision, state: Any) -> bool:
        with self._lock:
            self._pending = {"step": state.step, "decision": decision.model_dump(mode="json", exclude_none=True)}
            self._verdict = None
            self._wake.clear()
        while True:
            self._wake.wait(0.2)
            with self._lock:
                if self.runner is not None and self.runner.stop_reason is not None:
                    self._pending = None
                    return False
                if self._verdict is not None:
                    ok, self._pending, step = self._verdict, None, state.step
                    break
        if ok and self.runner is not None:
            self.runner.log.event("operator_approved", step=step)
        return ok

    def pending(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return dict(self._pending) if self._pending else None

    def _decide(self, ok: bool) -> bool:
        with self._lock:
            if self._pending is None:
                return False
            self._verdict = ok
            self._wake.set()
            return True

    def approve(self) -> bool:
        return self._decide(True)

    def reject(self) -> bool:
        return self._decide(False)

    # ---------- e-stop ----------
    def estop(self, reason: str = "operator E-STOP (console)") -> bool:
        """Stop the loop and put the simulated robot into the e-stop state. Only effective for the mujoco backend; there is no real-robot path.
        If this job has already ended: the simulation is closed, there is nothing to stop, and the finished run record is not changed."""
        if self.runner is None or self.finished:
            return False
        self.estopped = True
        self.runner.request_stop(reason)
        self.runner.stop_kind = "estop"                       # Stop clicked first, then e-stop: the outcome is recorded as e-stop
        robot = self.robot
        if robot is not None and getattr(robot, "backend_name", "") == "mujoco" and hasattr(robot, "emergency_stop"):
            robot.emergency_stop(reason)
        self._wake.set()
        return True

    # ---------- pause / resume / normal stop ----------
    def pause(self) -> bool:
        """Hold after the current skill finishes and before the next decision (the robot stays at the survey pose)."""
        if self.runner is None or not self.running:
            return False
        self.runner.request_pause("operator pause (software)")
        return True

    def resume(self) -> bool:
        if self.runner is None or not self.running:
            return False
        self.runner.resume()
        return True

    def stop(self) -> bool:
        """Normal stop: does not trigger a robot e-stop; wraps up after the current skill ends and writes the report (outcome stopped)."""
        if self.runner is None or not self.running:
            return False
        self.runner.request_stop("operator stop (software)", kind="stop")
        self._wake.set()
        return True

    # ---------- queries ----------
    def load(self) -> Optional[RunData]:
        if self.run_dir is None or not (self.run_dir / "events.jsonl").is_file():
            return None
        return load_run(self.run_dir)

    def status(self) -> dict[str, Any]:
        summary = self.result["summary"] if self.result else None
        return {
            "mode": self.mode, "run_dir": str(self.run_dir) if self.run_dir else None,
            "live": {
                "scenario": self.scenario, "policy": self.policy, "verifier": self.verifier, "source": self.source,
                "require_approval": self.require_approval, "pace": self.pace,
                "started": self._thread is not None, "running": self.running, "finished": self.finished,
                "pending": self.pending(), "estopped": self.estopped, "error": self.error,
                "paused": bool(self.runner is not None and self.runner.paused),
                "final_status": summary.get("status") if summary else None,
                "judge": self.result.get("judge") if self.result else None,
                "robot_safety_mode": (int(self.robot.get_safety_mode()) if self.robot is not None else None),
                "elapsed_s": round(time.time() - self.started_at, 1) if self.started_at else None,
                "started_at": self.started_at,                # the UI uses this to tell a "new job" from the one that just ended (late state of the old job is dropped on a job change)
            },
        }

    def frame(self) -> Optional[bytes]:
        """Current camera frame (JPEG): use the display viewpoint if there is one (rendered in the simulation thread), otherwise the top view.
        The top view is taken only from the topdown source: the GL context of the render source is bound to the simulation thread and cannot be used across threads."""
        if self._view_jpg is not None:
            return self._view_jpg
        if self._view is not False:                           # the display viewpoint is still being set up (about 0.5 s after job creation): return no image yet, the page keeps the previous frame and does not flash the top view
            return None
        if self.src is None or getattr(self.src, "mode", "") != "topdown":
            return None
        try:
            from ..perception.sources import encode_jpeg

            return encode_jpeg(self.src.rectified(), 85)
        except Exception:
            return None


class RemoteSession(LiveSession):
    """Live mode (paired robot computer): a background thread connects to the skill server and camera service to run a job (cleanproof/agent/remote_job.py),
    using the same TaskRunner and the same approve / pause / stop as the simulated deployment, with identical event and report file formats (the UI shows them the same way).
    E-stop = stop the loop + ask the skill server for a software stop (POST /stop, on a separate connection, delivered immediately even while a skill is executing); a normal stop does not send /stop.
    When the robot computer runs a simulation backend (deployment self-test), the reference image, camera and display viewpoint all come from it (/sim/reference, /frame/fixed, /frame/view).

    If before job creation the robot computer is already e-stopped / faulted / busy / not armed / unreachable / token rejected: no job is created, and the first line of error is an English explanation for the operator."""

    mode = "remote"
    FRAME_EVERY_S = 0.2                                       # camera frame cache (wall-clock seconds): page polling does not pile requests onto the robot computer
    FRAME_RETRY_S = 1.0                                       # how long to wait before retrying when no frame could be fetched

    def __init__(self, skill_url: str, camera_url: str, site: str, task: dict[str, Any], policy: str = "rules", verifier: str = "cv",
                 run_root: Optional[Path] = None, require_approval: bool = True, client: Any = None, audit: bool = False,
                 recheck_all: bool = True, robot: str = ""):
        if verifier not in ("cv", "vlm"):
            raise ValueError("verifier ∈ ('cv', 'vlm')")
        super().__init__(f"site:{site}", policy, verifier, "camera", run_root, require_approval=require_approval, pace=0.0,
                         client=client, audit=audit, recheck_all=recheck_all)
        self.skill_url, self.camera_url = skill_url.rstrip("/"), camera_url.rstrip("/")
        self.site, self.task = site, dict(task)
        self.robot_key = str(robot or "")                     # robot platform registered for the site (a key in configs/platforms.yaml), written to reference_meta
        self.link: Any = None
        self._view = False
        self._ctl = threading.Lock()                          # publishing of runner / link and stop requests (across threads)
        self._stop_req: Optional[tuple[str, str]] = None      # stop received before the runner was built: (kind, reason)
        self.estop_delivered: Optional[bool] = None           # whether the e-stop reached the skill server (if not: only the physical e-stop on site can help)
        self._frame_lock = threading.Lock()
        self._frame_try = 0.0
        self._frame_failed = False
        self._has_view: Optional[bool] = None                 # whether the robot computer has a display viewpoint (/frame/view; None = not known yet)

    # ---------- one job ----------
    def _target(self) -> None:
        from datetime import datetime

        from ..agent.force_table import load_force_table
        from ..agent.loop import LoopOptions
        from ..agent.remote_job import RemoteSetupError, connect_robot
        from ..agent.runlog import RunLogger
        from ..agent.state import TaskState
        from ..common.schemas import TaskSpec
        from ..perception.residue_cv import ResidueDetector
        from ..perception.verifier import CVVerifier, VLMVerifier
        from ..sim.run import make_policy
        link = None
        try:
            if self.verifier == "vlm" and self.client is None:
                raise RemoteSetupError("The Cosmos 3 zone check needs the model client. Check the model settings, then start the job.")
            named = [str(z) for z in (self.task.get("zones") or [])]
            link = connect_robot(self.skill_url, self.camera_url, self.site, self.task.get("forbidden_zones") or [], require_ready=True,
                                 expect_platform=self.robot_key, zones=named)
            with self._ctl:
                self.link = link
                pre = self._stop_req
            if pre is not None:                               # stop / e-stop was pressed during connection: no job, no run directory
                self._cancelled(pre)
                return
            det = ResidueDetector(link.grid, reference_rgb=link.reference)
            ver = VLMVerifier(link.grid, det, self.client) if self.verifier == "vlm" else CVVerifier(link.grid, det)
            table = load_force_table(self.site)               # force table of this job site: read once at job creation, used by guardrails and planning for the whole job
            pol = make_policy(self.policy, self.client, table=table)
            t = self.task                                     # named does not remove forbidden zones here: TaskState removes them (all removed != all zones)
            task = TaskSpec(instruction=str(t.get("instruction") or ""), surface=str(t.get("surface") or "sealed_wood"),
                            zones=named, forbidden_zones=list(link.forbidden),
                            max_passes_per_zone=int(t.get("max_passes_per_zone", 3)), max_steps=int(t.get("max_steps", 30)))
            if named and not TaskState(task, link.grid).targets:
                raise RemoteSetupError("None of the zones named in the instruction can be cleaned here "
                                       "(no-wipe zones or not on this surface).")
            h = link.health
            run_id = f"live-{self.site}-{h['backend']}-{h['mode']}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
            logger = RunLogger(self.run_root, run_id=run_id)
            logger.image("reference", link.reference)
            logger.write_json("reference_meta.json", self._reference_meta(link))
            truth = None
            if link.sim:                                       # simulation backend: inspection events carry ground truth (same as the built-in simulation, for evaluation only, not used in decisions)
                def truth() -> dict[str, bool]:
                    try:
                        return {k: bool(v) for k, v in link.ex.http.get("/sim/truth", timeout=5.0).json().items()}
                    except Exception:  # noqa: BLE001
                        return {}
            runner = TaskRunner(task, link.ex, link.src, ver, pol, logger, LoopOptions(audit=self.audit, recheck_all=self.recheck_all),
                                approve=self._approve if self.require_approval else None,
                                audit_client=self.client if self.audit else None, truth=truth, force_table=table)
            with self._ctl:                                   # publishing the runner and checking "stop pressed before job creation" happen under one lock: no stop can slip in between
                pre = self._stop_req
                if pre is None:
                    self.runner, self.src, self.run_dir = runner, link.src, logger.dir
            if pre is not None:                               # stop / e-stop pressed during preparation: do not start any motion
                self._cancelled(pre)
                return
            # when runner.run() raises, no skill is running on the robot side (skill requests are synchronous and the client does not raise), so there is no need to send /stop
            summary = runner.run()
            summary["robot_mode"], summary["robot_label"] = h["mode"], h["label"]
            logger.write_json("summary.json", summary)        # add the grading mode and labels of the robot computer (same as the built-in simulation)
            judge = None
            if link.sim:                                       # the robot computer is a simulation: judge by its ground truth (same as scripts/run_real.py --sim)
                tr = truth() if truth else {}
                judge = {"targets_truly_clean": all(tr.get(z, False) for z in summary["targets"])} if tr else None
            self.result = {"summary": summary, "judge": judge, "run_dir": str(logger.dir)}
        except RemoteSetupError as e:                          # unreachable / token rejected / already e-stopped / cannot create a job: one sentence for the operator
            with self._ctl:
                pre = self._stop_req
            if pre is not None and self.runner is None:       # e-stop was clicked during connection (/stop was delivered directly, so the connection was refused): the outcome is recorded as stop
                self._cancelled(pre)
            else:
                self.error = str(e)
        except Exception as e:  # noqa: BLE001  other exceptions in a thread: details go to the log, the UI gets one readable reason
            log.exception("remote job thread failed")
            self.error = operator_error(e)
            self._write_error_summary()
        finally:
            if link is not None:
                try:
                    link.ex.close()
                except Exception:  # noqa: BLE001
                    pass
            self._wake.set()

    def _cancelled(self, pre: tuple[str, str]) -> None:
        kind, reason = pre
        self.result = {"summary": {"status": "estopped" if kind == "estop" else "stopped", "reason": reason},
                       "judge": None, "run_dir": None}

    def _reference_meta(self, link: Any) -> dict[str, Any]:
        """reference_meta.json: the same keys as the built-in simulation (source / scenario / robot), plus the state of the robot computer;
        the report shows the site and robot from site and robot.platform."""
        from ..robot.platforms import load_platforms
        h = link.health
        platform = str(h.get("platform") or "") or self.robot_key       # trust what the robot computer reports (verified against the site before job creation)
        spec = load_platforms().get(platform) if platform else None
        robot = {"platform": platform, "platform_label": spec.label if spec else platform, "robot_backend": h.get("backend"),
                 "mode": h.get("mode"), "label": h.get("label"), "simulated": bool(link.sim),
                 "robot_computer": {"skill_url": self.skill_url, "camera_url": self.camera_url, "host": h.get("host"),
                                    "platform": h.get("platform")}}
        return {"source": link.src.name, **link.meta, "site": self.site, "robot": robot,
                "scenario": {"name": f"site:{self.site}", "site": self.site}}

    # ---------- e-stop / stop ----------
    def _post_stop(self, reason: str) -> bool:
        """When not yet connected (or already disconnected), send POST /stop directly to the address."""
        import httpx

        from ..robot.server import auth_headers
        try:
            return httpx.post(f"{self.skill_url}/stop", json={"reason": reason}, headers=auth_headers(), timeout=3.0).status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def estop(self, reason: str = "operator E-STOP (console)") -> bool:
        """Stop the loop and ask the skill server for a software stop (it halts the robot and rejects later skills; the robot can move again only after an on-site reset).
        Delivered immediately even while a skill is executing (separate connection; the skill server /stop does not wait for the skill lock); the loop stop flag is set first, so a skill exiting with fault is recorded as an e-stop."""
        if self._thread is None:
            return False
        if self.finished:                                     # this job has ended: only deliver the stop to the robot computer, do not change the finished run record
            self.estop_delivered = self._post_stop(reason)
            return True
        with self._ctl:
            self.estopped = True
            runner = self.runner
            if runner is None:
                self._stop_req = ("estop", reason)
        if runner is not None:
            runner.request_stop(reason)
            runner.stop_kind = "estop"                        # Stop clicked first, then e-stop: the outcome is recorded as e-stop
        link = self.link
        ok = link.ex.stop(reason) if link is not None else self._post_stop(reason)
        self.estop_delivered = ok
        if not ok and runner is not None:
            runner.log.event("estop_not_delivered", step=runner.state.step, skill_url=self.skill_url,
                             reason="robot computer not reachable: use the physical E-stop on site")
        self._wake.set()
        return True

    def stop(self, reason: str = "operator stop (software)") -> bool:
        """Normal stop: does not send /stop; wraps up and writes the report after the current skill finishes (outcome stopped). If clicked during connection: no job is created."""
        if not self.running:
            return False
        with self._ctl:
            runner = self.runner
            if runner is None:
                if self._stop_req is None:
                    self._stop_req = ("stop", reason)
                return True
        runner.request_stop(reason, kind="stop")
        self._wake.set()
        return True

    # ---------- queries ----------
    def status(self) -> dict[str, Any]:
        out = super().status()
        live = out["live"]
        h = self.link.health if self.link is not None else {}
        live["robot_computer"] = {"skill_url": self.skill_url, "camera_url": self.camera_url,
                                  "backend": h.get("backend"), "mode": h.get("mode"), "label": h.get("label"),
                                  "simulated": (bool(self.link.sim) if self.link is not None else None)}
        live["site"], live["robot"] = self.site, h.get("platform") or self.robot_key
        live["estop_delivered"] = self.estop_delivered
        return out

    def frame(self) -> Optional[bytes]:
        """Camera frame (JPEG): the camera service of the robot computer; when it runs a simulation backend, take the display viewpoint (/frame/view, same viewpoint as the built-in simulation),
        and take the fixed camera only if this platform has no display viewpoint (404). Cached per FRAME_EVERY_S; if a fetch fails, keep the previous frame and retry after FRAME_RETRY_S."""
        link = self.link
        if link is None:
            return self._view_jpg
        with self._frame_lock:
            now = time.time()
            if self._view_jpg is not None and now - self._view_t < self.FRAME_EVERY_S:
                return self._view_jpg
            if self._frame_failed and now - self._frame_try < self.FRAME_RETRY_S:
                return self._view_jpg                         # the last fetch failed: do not hit it again yet
            self._frame_try, self._frame_failed = now, True
            import httpx

            from ..robot.server import auth_headers
            base = link.camera_url.rstrip("/")
            cams = ("view", "fixed") if link.sim and self._has_view is not False else ("fixed",)
            for cam in cams:
                try:
                    r = httpx.get(f"{base}/frame/{cam}", timeout=(8.0 if cam == "view" else 3.0), headers=auth_headers())
                except Exception:  # noqa: BLE001  timeout / disconnect: keep the previous frame (so the fixed camera does not flash while the first display-viewpoint frame is slow)
                    break
                if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                    self._has_view = True if cam == "view" else self._has_view
                    self._view_jpg, self._view_t, self._frame_failed = r.content, time.time(), False
                    break
                if cam == "view" and r.status_code == 404:    # this platform has no display viewpoint: fetch only the fixed camera from now on
                    self._has_view = False
                    continue
                break
            return self._view_jpg
