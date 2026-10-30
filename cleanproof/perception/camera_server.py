"""Camera service on the robot host (robot-pc): RealSense streaming → HTTP (PLAN §4.1). Separate process from the skill server. Lesson from our earlier UR3e force-control work:
writing camera frames to disk once stalled the control loop for seconds, so the camera process never touches RTDE and the skill server process never touches the camera.

    GET  /health                 latest frame time, measured frame rate, exposure / white-balance settings and recording state of each camera
    GET  /frame/<camera>         latest JPEG frame (?fmt=png for lossless, used in calibration); returns 503 if there is no new frame for over 1 s
    GET  /intrinsics/<camera>    RealSense intrinsics
    POST /record/start {"name", "camera"} / POST /record/stop     record mp4 (whole-trial top-down view, video material and evidence)

Cameras (configs/cameras.yaml, can be overridden by configs/site/<site>/cameras.yaml):
    fixed  D455 (global top-down, 1280×720)  must be on a root USB port of the mainboard, not sharing a hub with the D405 (our earlier UR3e force-control work measured: sharing drops frames and freezes)
    wrist  D405 (wrist close-up, 640×480)   remove it from the config if not installed; the agent-side close-up skill then honestly records "close-up image unavailable"
Exposure and white balance: residue detection is a color difference against a clean reference image, and auto exposure would drift the whole image, so by default they are locked after warm-up to the auto values at that moment
(fixed values are used when the site config sets them). The locked values are written to /health and recorded in the site config at calibration, so every later boot is the same.

  python -m cleanproof.perception.camera_server                       # on robot-pc, default 127.0.0.1:8101
  python -m cleanproof.perception.camera_server --replay DIR          # no-camera self-test: serve the images in the directory in a loop
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from ..common.config import load_yaml

log = logging.getLogger("cleanproof.camera")


class _Camera:
    name: str

    def latest(self) -> tuple[Optional[np.ndarray], float, int]: ...
    def intrinsics(self) -> dict[str, Any]: ...
    def info(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class RealSenseCamera(_Camera):
    def __init__(self, name: str, serial: str, width: int, height: int, fps: int, lock_exposure: bool = True,
                 exposure: Optional[float] = None, gain: Optional[float] = None, white_balance: Optional[float] = None,
                 warmup_s: float = 2.0):
        import pyrealsense2 as rs

        self.rs, self.name, self.serial = rs, name, str(serial)
        self.size, self.fps = (int(width), int(height)), int(fps)
        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.size[0], self.size[1], rs.format.rgb8, self.fps)
        self.pipe = rs.pipeline()
        self.profile = self.pipe.start(cfg)
        self.sensor = self.profile.get_device().first_color_sensor()
        self._img: Optional[np.ndarray] = None
        self._t, self._n = 0.0, 0
        self._times: list[float] = []
        self._lock = threading.Lock()
        self._stop = False
        self.settings: dict[str, Any] = {"locked": False}
        self._th = threading.Thread(target=self._loop, daemon=True, name=f"cam-{name}")
        self._th.start()
        time.sleep(warmup_s)                       # let auto exposure converge first
        if lock_exposure:
            self._lock_exposure(exposure, gain, white_balance)

    def _opt(self, o: Any) -> Optional[float]:
        try:
            return float(self.sensor.get_option(o)) if self.sensor.supports(o) else None
        except Exception:  # noqa: BLE001
            return None

    def _lock_exposure(self, exposure: Optional[float], gain: Optional[float], wb: Optional[float]) -> None:
        o = self.rs.option
        cur = {"exposure": self._opt(o.exposure), "gain": self._opt(o.gain), "white_balance": self._opt(o.white_balance)}
        want = {"exposure": exposure if exposure is not None else cur["exposure"],
                "gain": gain if gain is not None else cur["gain"],
                "white_balance": wb if wb is not None else cur["white_balance"]}
        try:
            self.sensor.set_option(o.enable_auto_exposure, 0)
            self.sensor.set_option(o.enable_auto_white_balance, 0)
            for k, opt in (("exposure", o.exposure), ("gain", o.gain), ("white_balance", o.white_balance)):
                if want[k] is not None and self.sensor.supports(opt):
                    self.sensor.set_option(opt, float(want[k]))
            self.settings = {"locked": True, **want, "auto_values_at_lock": cur}
        except Exception as e:  # noqa: BLE001
            self.settings = {"locked": False, "error": f"{type(e).__name__}: {e}", "auto_values": cur}
        log.info("%s exposure settings: %s", self.name, self.settings)

    def _loop(self) -> None:
        while not self._stop:
            try:
                fs = self.pipe.wait_for_frames(2000)
                c = fs.get_color_frame()
                if not c:
                    continue
                img = np.asanyarray(c.get_data()).copy()
                now = time.time()
                with self._lock:
                    self._img, self._t, self._n = img, now, self._n + 1
                    self._times = (self._times + [now])[-60:]
            except Exception as e:  # noqa: BLE001
                log.warning("%s stream error: %s", self.name, e)
                time.sleep(0.2)

    def latest(self) -> tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            return self._img, self._t, self._n

    def intrinsics(self) -> dict[str, Any]:
        i = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile().get_intrinsics()
        return {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "model": str(i.model), "coeffs": list(i.coeffs), "serial": self.serial}

    def info(self) -> dict[str, Any]:
        with self._lock:
            ts = list(self._times)
        fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else None
        return {"serial": self.serial, "size": self.size, "fps_nominal": self.fps, "fps_measured": fps, "exposure": self.settings}

    def close(self) -> None:
        self._stop = True
        try:
            self.pipe.stop()
        except Exception:  # noqa: BLE001
            pass


class ReplayCamera(_Camera):
    """No-camera self-test: serve the images in the directory in a loop at fps (RGB)."""

    def __init__(self, name: str, directory: str | Path, fps: float = 5.0):
        self.name = name
        self.files = sorted(Path(directory).glob("*.jpg")) + sorted(Path(directory).glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"no images in {directory}")
        self.fps, self.t0 = float(fps), time.time()
        self._cache: dict[int, np.ndarray] = {}

    def latest(self) -> tuple[Optional[np.ndarray], float, int]:
        n = int((time.time() - self.t0) * self.fps)
        k = n % len(self.files)
        if k not in self._cache:
            self._cache[k] = cv2.cvtColor(cv2.imread(str(self.files[k])), cv2.COLOR_BGR2RGB)
        return self._cache[k], time.time(), n

    def intrinsics(self) -> dict[str, Any]:
        img = self.latest()[0]
        return {"width": int(img.shape[1]), "height": int(img.shape[0]), "replay": True}

    def info(self) -> dict[str, Any]:
        return {"replay": str(self.files[0].parent), "frames": len(self.files)}

    def close(self) -> None:
        pass


class Recorder:
    def __init__(self, cam: _Camera, path: Path, fps: float):
        self.cam, self.path, self.fps = cam, path, float(fps)
        self.n, self._stop, self.t0 = 0, False, time.time()
        self._th = threading.Thread(target=self._loop, daemon=True, name=f"rec-{cam.name}")
        self._th.start()

    def _loop(self) -> None:
        w = None
        last_n = -1
        stamps = open(self.path.with_suffix(".times.csv"), "w", encoding="utf-8")
        stamps.write("frame,unix_time\n")
        try:
            while not self._stop:
                img, t, n = self.cam.latest()
                if img is None or n == last_n:
                    time.sleep(0.2 / self.fps)
                    continue
                last_n = n
                if w is None:
                    w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (img.shape[1], img.shape[0]))
                w.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                stamps.write(f"{self.n},{t:.4f}\n")
                self.n += 1
        finally:
            if w is not None:
                w.release()
            stamps.close()

    def stop(self) -> dict[str, Any]:
        self._stop = True
        self._th.join(timeout=5.0)
        return {"path": str(self.path), "frames": self.n, "duration_s": round(time.time() - self.t0, 2)}


class RecReq(BaseModel):
    name: str = "trial"
    camera: str = "fixed"


class CameraHub:
    def __init__(self, cams: dict[str, _Camera], record_dir: Path):
        self.cams, self.record_dir = cams, Path(record_dir)
        self.rec: Optional[Recorder] = None
        self.rec_name = ""

    def frame(self, name: str, max_age_s: float = 1.0) -> np.ndarray:
        if name not in self.cams:
            raise KeyError(f"no camera {name!r} (available: {sorted(self.cams)})")
        img, t, _ = self.cams[name].latest()
        if img is None or time.time() - t > max_age_s:
            raise TimeoutError(f"camera {name} has had no new frame for over {max_age_s} s")
        return img

    def record_start(self, name: str, camera: str = "fixed") -> dict[str, Any]:
        if self.rec is not None:
            raise RuntimeError(f"already recording: {self.rec_name}")
        self.record_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch for ch in name if ch.isalnum() or ch in "-_") or "trial"
        path = self.record_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}_{camera}.mp4"
        cam = self.cams[camera]
        fps = float(getattr(cam, "fps", 15) or 15)
        self.rec, self.rec_name = Recorder(cam, path, fps), str(path)
        return {"recording": str(path)}

    def record_stop(self) -> dict[str, Any]:
        if self.rec is None:
            return {"recording": None}
        out = self.rec.stop()
        self.rec, self.rec_name = None, ""
        return out

    def health(self) -> dict[str, Any]:
        out = {}
        for k, c in self.cams.items():
            _, t, n = c.latest()
            out[k] = {"frames": n, "age_s": round(time.time() - t, 3) if t else None, **c.info()}
        return {"cameras": out, "recording": self.rec_name or None}

    def close(self) -> None:
        self.record_stop()
        for c in self.cams.values():
            c.close()


def create_app(hub: CameraHub):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response

    app = FastAPI(title="CleanProof camera service")

    from ..robot.server import TOKEN_HEADER, read_token

    token = read_token()
    if token:                      # shares the token with the skill server: frames and recording control are not open to other accounts on the shared machine
        import hmac

        from fastapi.responses import JSONResponse

        @app.middleware("http")
        async def require_token(request, call_next):
            if not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), token):
                return JSONResponse({"error": "missing or wrong token"}, status_code=401)
            return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return hub.health()

    @app.get("/frame/{camera}")
    def frame(camera: str, fmt: str = "jpg", q: int = 95):
        try:
            img = hub.frame(camera)
        except KeyError as e:
            raise HTTPException(404, str(e)) from e
        except TimeoutError as e:
            raise HTTPException(503, str(e)) from e
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if fmt == "png":
            ok, buf = cv2.imencode(".png", bgr)
            return Response(buf.tobytes(), media_type="image/png")
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(q, 50, 100))])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.get("/intrinsics/{camera}")
    def intrinsics(camera: str) -> dict[str, Any]:
        if camera not in hub.cams:
            raise HTTPException(404, camera)
        return hub.cams[camera].intrinsics()

    @app.post("/record/start")
    def record_start(req: RecReq) -> dict[str, Any]:
        try:
            return hub.record_start(req.name, req.camera)
        except (RuntimeError, KeyError) as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/record/stop")
    def record_stop() -> dict[str, Any]:
        return hub.record_stop()

    return app


def build_hub(replay: Optional[str], record_dir: Path, only: Optional[list[str]] = None) -> CameraHub:
    if replay:
        return CameraHub({"fixed": ReplayCamera("fixed", replay)}, record_dir)
    cfg = load_yaml("cameras.yaml")
    cams: dict[str, _Camera] = {}
    for name, c in (cfg.get("cameras") or {}).items():
        if only and name not in only:
            continue
        if not c or not c.get("enabled", True):
            continue
        try:
            if str(c.get("serial", "")).startswith("<"):
                raise RuntimeError(f"camera {name}: set the real serial number in configs/cameras.yaml (it is still a placeholder)")
            cams[name] = RealSenseCamera(name, c["serial"], c["width"], c["height"], c["fps"],
                                         lock_exposure=bool(c.get("lock_exposure", True)), exposure=c.get("exposure"),
                                         gain=c.get("gain"), white_balance=c.get("white_balance"))
        except Exception as e:  # noqa: BLE001
            log.error("camera %s (%s) failed to start: %s", name, c.get("serial"), e)
    if "fixed" not in cams:
        raise RuntimeError("fixed camera (D455) failed to start; check the USB connection (root USB port) and the serial number in configs/cameras.yaml")
    return CameraHub(cams, record_dir)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--replay", default=None, help="no-camera self-test: image directory")
    ap.add_argument("--only", nargs="*", default=None, help="start only these cameras (e.g. fixed)")
    ap.add_argument("--record-dir", default=str(Path("~/cleanproof_data/recordings").expanduser()))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    hub = build_hub(args.replay, Path(args.record_dir), args.only)
    print(json.dumps(hub.health(), ensure_ascii=False, default=str))
    import uvicorn

    try:
        uvicorn.run(create_app(hub), host=args.bind, port=args.port, log_level="info", workers=1)
    finally:
        hub.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
