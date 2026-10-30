"""FastAPI console: static page + JSON API. The page polls /api/status and /api/run (the event stream is events.jsonl).

Endpoints:
  GET  /                 console page (static/index.html, in English)
  GET  /api/status       session state (mode, whether running, decision awaiting approval, e-stop flag)
  GET  /api/run          structured data of the whole run (steps, snapshots, model calls)
  GET  /api/image/{rel}  image inside the run directory
  GET  /api/trace        ?path=traces/xxx.npz → downsampled force curve
  GET  /api/frame        current camera image in live mode (404 if none)
  GET  /api/report       HTML cleaning report of the current run
  POST /api/start | /api/approve | /api/reject | /api/estop   live-mode control
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from ..report.html import render_report
from ..report.rundata import load_trace, run_to_dict
from .session import LiveSession, ReplaySession

STATIC_DIR = Path(__file__).resolve().parent / "static"
Session = Union[ReplaySession, LiveSession]


def create_app(session: Session) -> FastAPI:
    app = FastAPI(title="CleanProof Console", docs_url=None, redoc_url=None)
    app.state.session = session

    def run_dir() -> Path:
        d = session.run_dir
        if d is None:
            raise HTTPException(404, "run has not started yet")
        return Path(d)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return session.status()

    @app.get("/api/run")
    def run() -> JSONResponse:
        data = session.load()
        if data is None:
            return JSONResponse({"status": "starting", "steps": [], "snapshots": [], "calls": []})
        return JSONResponse(run_to_dict(data))

    @app.get("/api/image/{rel:path}")
    def image(rel: str) -> FileResponse:
        base = run_dir().resolve()
        path = (base / rel).resolve()
        if base not in path.parents or not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise HTTPException(404, "image not found")
        return FileResponse(path)

    @app.get("/api/trace")
    def trace(path: str = Query(...), max_points: int = Query(600, ge=50, le=5000)) -> dict[str, Any]:
        base = run_dir().resolve()
        full = (base / path).resolve()
        if base not in full.parents or full.suffix != ".npz":
            raise HTTPException(404, "trace not found")
        tr = load_trace(base, path, max_points)
        if tr is None:
            raise HTTPException(404, "trace not found")
        return tr

    @app.get("/api/frame")
    def frame() -> Response:
        jpg = session.frame()
        if jpg is None:
            raise HTTPException(404, "no live frame")
        return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/report", response_class=HTMLResponse)
    def report() -> str:
        data = session.load()
        if data is None:
            raise HTTPException(404, "run has not started yet")
        return render_report(data, embed_images=True)

    # ---- live control (returns 409 in replay mode) ----
    def live() -> LiveSession:
        if not isinstance(session, LiveSession):
            raise HTTPException(409, "replay mode: no live controls")
        return session

    @app.post("/api/start")
    def start() -> dict[str, Any]:
        s = live()
        if s._thread is None:
            s.start()
        return s.status()

    @app.post("/api/approve")
    def approve() -> dict[str, Any]:
        s = live()
        return {"ok": s.approve(), **s.status()}

    @app.post("/api/reject")
    def reject() -> dict[str, Any]:
        s = live()
        return {"ok": s.reject(), **s.status()}

    @app.post("/api/estop")
    def estop() -> dict[str, Any]:
        s = live()
        return {"ok": s.estop(), **s.status()}

    return app
