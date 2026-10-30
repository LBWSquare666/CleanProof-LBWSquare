#!/usr/bin/env python3
"""Desktop edition screenshots from a real run:
open the desktop page (?app=1, window 1568x718) in headless Chrome and click like an operator: first-run wizard in four steps → enter a job, Start →
wait for the zone scan, force database screening, activity log, handoff notification → open Staff check → settings (+1, switch group, save) → logs (filter by model) → device page / site / report,
one screenshot per step, for page-by-page comparison with the software segment of the video (runs/v14/software_e.mp4). The service must already be up at --url (live mode, cleanproof.ui.desktop --no-window).
Wizard Finish and settings Save really write configs/site/ (the service and this script share one repo): save a copy of those files before running and write them back as they were afterwards, so the repo is left unchanged.

  python scripts/desktop_parity_shots.py --url http://127.0.0.1:8097 --out runs/v14/desktop_shots

--paired: the service is connected to a paired robot computer (over an SSH tunnel). Same click flow, wizard text checked against the paired deployment; also tests Pause → screenshot → Resume, one job ended with Stop,
one job ended with E-stop (E-stop goes last: afterwards the robot-side skill server is stopped), and after E-stop checks that /health at --skill-url reports stopped. Results go to paired_results.json.
New controls missing from the UI (settings page groups, new site / calibration) are only logged as [miss] and never crash the script.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


SITE_CFG = Path(__file__).resolve().parents[1] / "configs" / "site"


class Snap(dict):
    """File snapshot (path → content), also recording which directories already existed."""
    dirs: set[Path]


def snapshot() -> Snap:
    snap = Snap({f: f.read_bytes() for f in SITE_CFG.rglob("*") if f.is_file()})
    snap.dirs = {d for d in SITE_CFG.rglob("*") if d.is_dir()}
    return snap


def restore(snap: Snap) -> None:
    for f in SITE_CFG.rglob("*"):
        if f.is_file() and f not in snap:
            f.unlink()                                        # file created during this run (e.g. operator.yaml written by the first save on the settings page)
    for f, b in snap.items():
        if not f.is_file() or f.read_bytes() != b:
            f.write_bytes(b)
    for d in sorted((d for d in SITE_CFG.rglob("*") if d.is_dir() and d not in snap.dirs), key=lambda x: len(x.parts), reverse=True):
        try:
            d.rmdir()                                         # site dir created during this run; delete only when empty, after the files are restored
        except OSError:
            pass


def state(url: str) -> dict:
    with urllib.request.urlopen(url + "/api/jobs/state", timeout=5) as r:
        return json.loads(r.read())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def skill_health(skill_url: str) -> dict | None:
    """Read /health of the robot computer skill server (token read from the config dir, not printed)."""
    cfg = Path(os.environ.get("CLEANPROOF_CONFIG_DIR") or "~/.config/cleanproof").expanduser()
    req = urllib.request.Request(skill_url.rstrip("/") + "/health")
    try:
        req.add_header("X-CleanProof-Token", (cfg / "robot_token").read_text().strip())
    except OSError:
        pass
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read()) or {}


def assert_simulated_robot(url: str) -> bool:
    """The script clicks Start / E-stop: in both modes it may only run against a simulated target (robot_mode == sim in /api/meta, or link.simulated is true in /api/robot/link)."""
    try:
        sim = _get(url + "/api/meta").get("robot_mode") == "sim"
        if not sim:
            sim = (_get(url + "/api/robot/link").get("link") or {}).get("simulated") is True
    except Exception as e:  # noqa: BLE001
        print(f"refused: cannot read robot_mode / robot link at {url} ({type(e).__name__}), so cannot confirm the target is simulated", file=sys.stderr)
        return False
    if not sim:
        print("refused: the script clicks Start / E-stop and may only run against a simulated target (robot_mode is not sim and link.simulated is not true)", file=sys.stderr)
        return False
    return True


async def run(args) -> int:
    import websockets
    if not assert_simulated_robot(args.url):
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    port = args.devtools_port or free_port()
    R: dict = {}                                              # results of --paired, written to paired_results.json
    prof = tempfile.mkdtemp(prefix="cp_parity_")
    chrome = subprocess.Popen([args.chrome, "--headless=new", f"--remote-debugging-port={port}", "--remote-debugging-address=127.0.0.1",
                               f"--user-data-dir={prof}", "--window-size=1568,718", "--hide-scrollbars", "--force-device-scale-factor=1",
                               "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shots = []
    snap = snapshot()
    try:
        for _ in range(100):
            try:
                pages = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read())
                page = next(p for p in pages if p.get("type") == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.1)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=64 * 1024 * 1024) as ws:
            seq = 0

            async def call(method: str, **params):
                nonlocal seq
                seq += 1
                me = seq
                await ws.send(json.dumps({"id": me, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == me:
                        return msg.get("result", {})

            async def js(expr: str):
                r = await call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
                return (r.get("result") or {}).get("value")

            async def ready():
                for _ in range(200):
                    if await js("window.cpReady === true"):
                        return
                    await asyncio.sleep(0.1)

            async def shot(name: str, note: str = ""):
                await asyncio.sleep(0.15)
                r = await call("Page.captureScreenshot", format="png")
                (out / f"{name}.png").write_bytes(base64.b64decode(r["data"]))
                shots.append({"name": name, "note": note})
                print(f"[shot] {name}  {note}", flush=True)

            async def click(sel: str, wait: float = 0.4):
                ok = await js(f"(() => {{ const e = document.querySelector({json.dumps(sel)}); if (!e) return false; e.click(); return true; }})()")
                if not ok:
                    print(f"[miss] {sel}", flush=True)
                await asyncio.sleep(wait)
                return ok

            async def click_text(text: str, wait: float = 0.5):
                """Click a button / link by visible text (used while a new control has no id yet); log [miss] if not found."""
                ok = await js("(() => { const t = %s.toLowerCase(); const c = [...document.querySelectorAll('button, a, [role=button], .nv, .wcard, .tab')]"
                              ".filter(e => e.offsetParent !== null && (e.innerText || '').toLowerCase().includes(t)); if (!c.length) return false;"
                              " c[0].click(); return true; })()" % json.dumps(text))
                if not ok:
                    print(f"[miss] text:{text}", flush=True)
                await asyncio.sleep(wait)
                return ok

            async def page_text() -> str:
                return await js("document.body.innerText") or ""

            await call("Page.enable")
            await call("Emulation.setDeviceMetricsOverride", width=1568, height=718, deviceScaleFactor=1, mobile=False)
            await call("Page.navigate", url=args.url + "/?app=1")
            await asyncio.sleep(1.0)
            await ready()
            await asyncio.sleep(0.8)
            # ---- first-run wizard ----
            await shot("01_wizard_robot", "wizard step 1: choose the robot (none selected)")
            await click("#wz-r-spot_arm")
            await shot("02_wizard_robot_selected", "Spot + Arm selected")
            await click("#wz-next")
            await asyncio.sleep(0.5)
            await shot("03a_wizard_connect_idle", "Connect: before clicking Connect")
            if args.paired:
                it = await page_text()
                for w in ("robot-pc", "SSH tunnel · ports 8100 and 8101"):
                    hit = w in it
                    R.setdefault("connect_idle_texts", {})[w] = hit
                    print(f"[{'ok' if hit else 'warn'}] connect idle text: {w!r}", flush=True)
            await click("#wz-connect", 0.3)
            await shot("03b_wizard_connect_checking", "Connect: checks in progress (about 0.3 s)")
            await asyncio.sleep(2.3)
            await shot("03_wizard_connect", "Connect: three checks (%s)" % ("paired deployment" if args.paired else "simulated deployment"))
            txt = await page_text()
            want = ("connected · token OK", "1 camera stream live", "ready · software adapter", "robot-pc", "SSH tunnel") if args.paired else ()
            for w in want:
                hit = w in txt
                R.setdefault("wizard_texts", {})[w] = hit
                print(f"[{'ok' if hit else 'warn'}] wizard connect text: {w!r}", flush=True)
            await click("#wz-next")
            await asyncio.sleep(0.5)
            await shot("04a_wizard_services_idle", "Services: before clicking Test")
            await click("#wz-test", 2.6)
            await shot("04_wizard_services", "Services: Token Factory / Cosmos 3 / orchestrator (fallback item in amber)")
            await click("#wz-next")
            await click("#wz-t-Home", 1.2)
            await shot("05_wizard_site", "Site: read back the site calibration (zone grid)")
            if await js("!!document.querySelector('#wz-sitesel option[value=__new__]')"):      # new site → one-click calibration (simulated); the new site files are cleaned up by restore
                await js("(() => { const e = document.querySelector('#wz-sitesel'); e.value = '__new__'; e.dispatchEvent(new Event('change', {bubbles: true})); })()")
                await asyncio.sleep(0.8)
                await shot("05a_wizard_new_site", "Site: New site...")
                await js("(() => { const i = document.querySelector('#wz-sname'); i.value = 'Parity test site'; i.dispatchEvent(new Event('input', {bubbles: true})); })()")
                await click("#wz-t-Home", 0.4)
                if await click("#wz-cal", 7.0):
                    await shot("05b_wizard_calibrated", "Site: after Calibrate")
                await js("(() => { const e = document.querySelector('#wz-sitesel'); e.value = 'home_dining'; e.dispatchEvent(new Event('change', {bubbles: true})); })()")
                await asyncio.sleep(1.0)                              # go back to the existing site and continue the wizard
            else:
                print("[miss] #wz-sitesel option __new__ (not live mode?)", flush=True)
            if await js("!!document.querySelector('#wz-gcal')?.innerText"):
                await shot("05c_wizard_guided_panel", "Site: guided calibration panel")
            await click("#wz-next", 2.5)
            await ready()
            await asyncio.sleep(1.0)
            await shot("06_jobs_ready", "main UI Jobs: Ready")
            if args.paired:
                await asyncio.sleep(3.5)                           # the title bar polls the robot connection every 3 s
                R["robot_connected_chip"] = "Robot connected" in await page_text()
                print(f"[{'ok' if R['robot_connected_chip'] else 'warn'}] title bar: Robot connected", flush=True)
            # ---- run one job (local simulation; the default site job = the one in the video: a mark that will not come off → handoff to a human) ----
            await js("(() => { const t = document.querySelector('#jtask'); t.value = 'Clean the dining table'; return true; })()")
            await click("#jstart", 0.2)
            t0 = time.time()
            got = set()
            while time.time() - t0 < args.job_timeout:
                await asyncio.sleep(0.25)
                st = state(args.url)
                zs = st.get("zones") or {}
                log = st.get("log") or []
                if "started" not in got and st.get("status") == "running":
                    got.add("started")
                    R["job_started"] = True
                    await shot("07_job_started", "after Start")
                if "scan" not in got and any(z.get("status") for z in zs.values()):
                    got.add("scan")
                    await asyncio.sleep(0.5)
                    await shot("08_scan", "first check: zone scan animation in progress")
                if "plan" not in got and any(e.get("kind") == "plan" for e in log):
                    got.add("plan")
                    await asyncio.sleep(1.0)
                    await shot("09_force_db", "planning: force database screening")
                if "mid" not in got and len(log) >= 8:
                    got.add("mid")
                    await shot("10_running_log", "running: activity log")
                    if args.paired:                                # Pause → screenshot → Resume
                        await click("#jpause", 1.2)
                        R["paused"] = bool((state(args.url).get("live") or {}).get("paused"))
                        await shot("10a_paused", "Pause: paused")
                        await click("#jpause", 1.2)
                        R["resumed"] = not (state(args.url).get("live") or {}).get("paused")
                        await shot("10b_resumed", "Resume: continued")
                if st.get("status") not in ("running", "idle") and "end" not in got:
                    got.add("end")
                    await asyncio.sleep(1.2)
                    await shot("11_end_toast", f"end: {st.get('status')} (handoff notification / Today row)")
                    R["toast_seen"] = bool(await js("(document.querySelector('#jtoast')?.innerText || '').trim().length > 0"))
                    break
            if (state(args.url).get("requests") or []):
                await click("#jl-current", 0.9)
                await shot("12_staff_card", "click this job in Today: Staff check card expanded")
                if await click("#jresolve", 1.0):                  # leave the handling note empty and click Resolve directly
                    await shot("12a_resolved_empty_note", "Resolve (note left empty)")
                    R["resolved_empty_note"] = "Resolved" in await page_text()
                    print(f"[{'ok' if R['resolved_empty_note'] else 'warn'}] Resolved shown after empty-note resolve", flush=True)
            # ---- settings ----
            await click('.nv[data-page="settings"]', 0.8)
            await shot("13_settings_jobs", "Settings: Jobs group")
            await click("#sj-passes-inc", 0.5)
            await shot("14_settings_inc", "Wipes per zone +1")
            await click("#sc-safety", 0.6)
            await shot("15_settings_safety", "Safety & force group")
            await click("#sc-vision", 0.6)
            await shot("15a_settings_vision", "Vision: Zone check")
            ver0 = await js("document.querySelector('[id^=\"sch-ver-\"].sel')?.id")      # note the originally selected one first, then click back to it later
            await click("#sch-ver-cv", 0.5)
            await shot("15b_settings_vision_cv", "Zone check: Classical vision only (not saved)")
            if ver0:
                await click("#" + ver0, 0.4)
            await click("#sc-cloud", 0.6)
            await shot("15c_settings_cloud", "Cloud models: API key / Planner / Spending")
            pol0 = await js("document.querySelector('[id^=\"sch-pol-\"].sel')?.id")
            await click("#sch-pol-rules", 0.5)
            await shot("15d_settings_planner_rules", "Planner: Rules (not saved)")
            if pol0:
                await click("#" + pol0, 0.4)
            await click("#sc-robot", 0.6)
            await shot("15e_settings_robot", "Robot: address and connection")
            await click("#sruncheck", 4.0)
            await shot("15f_settings_run_checks", "Robot: Run checks")
            await click("#sc-about", 0.6)
            await shot("15g_settings_about", "About")
            await click("#sc-safety", 0.5)
            rt0 = _get(args.url + "/api/settings").get("runtime")
            await click("#ssave", 0.7)
            await shot("16_settings_saved", "save notice")
            tst = await js("(() => { const t = document.querySelector('#toast'); return t ? {bad: t.classList.contains('bad'), text: t.innerText, op: getComputedStyle(t).opacity} : null; })()") or {}
            R["save_toast_ok"] = bool(tst) and not tst.get("bad") and "Saved · applies from the next job" in (tst.get("text") or "")
            R["save_runtime_unchanged"] = _get(args.url + "/api/settings").get("runtime") == rt0
            print(f"[{'ok' if R['save_toast_ok'] else 'warn'}] Save toast: {tst}", flush=True)
            print(f"[{'ok' if R['save_runtime_unchanged'] else 'warn'}] runtime unchanged after Save", flush=True)
            await asyncio.sleep(3.0)
            await shot("17_settings_toast_gone", "notice fades out by itself")
            await click("#sc-jobs", 0.4)
            await click("#sj-passes-dec", 0.3)
            await click("#ssave", 0.5)                             # change it back (the run leaves no changes behind)
            # ---- logs ----
            await click('.nv[data-page="logs"]', 0.8)
            await shot("18_logs", "Logs: all model calls")
            await click("#lf-super", 0.7)
            await shot("19_logs_super", "filter by model")
            # ---- device page / site / report ----
            await click('.nv[data-page="robots"]', 0.9)
            await shot("20_robots", "Robots device page")
            if await click(".rc", 1.2):
                await shot("20a_robot_zoom", "click a device card: full-screen view")
                for typ in ("keyDown", "keyUp"):
                    await call("Input.dispatchKeyEvent", type=typ, key="Escape", code="Escape", windowsVirtualKeyCode=27)
                await asyncio.sleep(1.0)
                await shot("20b_robot_zoom_closed", "Esc closes the full-screen view")
            await click('.nv[data-page="sites"]', 0.8)
            await shot("21_sites", "Sites")
            await click('.nv[data-page="reports"]', 0.8)
            await shot("22_reports", "Reports")
            await click('[id^="rr-"]', 1.5)
            await shot("23_report_open", "open a report")
            # ---- after one job ends, start another (can run back to back in one launch; the new job starts from the scan, no state carried over from the last one) ----
            await click('#drclose', 0.3)
            await click('.nv[data-page="jobs"]', 0.8)
            await shot("24_jobs_after_first", "back to Jobs: previous job ended, Start is clickable")
            old = (state(args.url).get("live") or {}).get("started_at")
            await click("#jstart", 0.6)
            await shot("25_second_started", "second job just started: Running, stats reset to zero, scanning")
            t0 = time.time()
            while time.time() - t0 < 60:
                await asyncio.sleep(0.25)
                stt = state(args.url)
                if (stt.get("live") or {}).get("started_at") not in (None, old) and stt.get("zones"):
                    break
            await asyncio.sleep(2.5)
            await shot("26_second_running", "second job: check results and force library screening")
            await click("#jstop", 0.6)
            await shot("27_second_stopping", "Stop: wraps up after the current step is done")
            t0 = time.time()
            while time.time() - t0 < 60 and state(args.url).get("status") == "running":
                await asyncio.sleep(0.25)
            await asyncio.sleep(1.2)
            await shot("28_second_stopped", "Stopped by operator")
            R["stopped"] = state(args.url).get("status") not in ("running",)
            R["stopped_detail"] = f"status={state(args.url).get('status')}"
            if args.paired:
                # ---- third job ends with E-stop (goes last: afterwards the robot-side skill server is stopped and has to be reset on site) ----
                await click("#jstart", 0.6)
                t0 = time.time()
                while time.time() - t0 < 60 and state(args.url).get("status") != "running":
                    await asyncio.sleep(0.25)
                R["job_started"] = R.get("job_started", False) and state(args.url).get("status") == "running"
                await asyncio.sleep(4.0)
                await shot("29_third_running", "third job running, getting ready for E-stop")
                R["estop_clicked"] = bool(await click("#jestop", 1.0))
                await shot("30_estop", "E-stop: robot side stops immediately")
                t0 = time.time()
                while time.time() - t0 < 60 and state(args.url).get("status") == "running":
                    await asyncio.sleep(0.25)
                await asyncio.sleep(1.0)
                await shot("31_estop_ended", "after E-stop")
                if args.skill_url:
                    h = skill_health(args.skill_url) or {}
                    R["estop_health_stopped"] = bool(h.get("stopped"))
                    R["estop_health_stopped_detail"] = f"/health stopped={h.get('stopped')!r}"
                    print(f"[estop] skill server /health stopped={h.get('stopped')!r}", flush=True)
    finally:
        chrome.terminate()
        shutil.rmtree(prof, ignore_errors=True)
        restore(snap)
    (out / "results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")      # written in both modes
    if args.paired:
        (out / "paired_results.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "shots.json").write_text(json.dumps(shots, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8097")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chrome", default="google-chrome")
    ap.add_argument("--devtools-port", type=int, default=0, help="Chrome DevTools port (default: pick a free port; 9333 is often taken by other sessions)")
    ap.add_argument("--paired", action="store_true", help="service is connected to a paired robot computer: also tests Pause/Resume, Stop, E-stop")
    ap.add_argument("--skill-url", default=None, help="--paired: skill server of the robot computer (check that /health reports stopped after E-stop)")
    ap.add_argument("--job-timeout", type=float, default=600.0)
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
