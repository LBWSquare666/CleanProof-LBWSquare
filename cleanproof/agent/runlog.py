"""Run record: event stream, images, skill traces and summary under runs/<run_id>/ (console, report and evaluation all read from here)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..common.config import runs_root
from ..common.jsonl import append_jsonl


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


class RunLogger:
    def __init__(self, root: Optional[Path] = None, run_id: Optional[str] = None):
        self.run_id = run_id or new_run_id()
        self.dir = (root or runs_root()) / self.run_id
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        (self.dir / "traces").mkdir(exist_ok=True)
        self.events_path = self.dir / "events.jsonl"
        self._t0 = time.time()
        self.listeners: list[Any] = []

    def event(self, kind: str, **data: Any) -> dict[str, Any]:
        rec = {"kind": kind, "wall_s": round(time.time() - self._t0, 3),
               "ts": datetime.now(timezone.utc).isoformat(), **data}
        append_jsonl(self.events_path, rec)
        for fn in self.listeners:
            fn(rec)
        return rec

    def image(self, name: str, rgb: np.ndarray, quality: int = 90) -> str:
        from ..perception.sources import encode_jpeg

        rel = f"images/{name}.jpg"
        (self.dir / rel).write_bytes(encode_jpeg(rgb, quality))
        return rel

    def trace(self, name: str, trace: dict[str, list[float]]) -> Optional[str]:
        if not trace or not trace.get("t"):
            return None
        rel = f"traces/{name}.npz"
        np.savez_compressed(self.dir / rel, **{k: np.asarray(v, dtype=np.float32) for k, v in trace.items()})
        return rel

    def write_json(self, name: str, data: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self.dir / name
        path.write_text(text, encoding="utf-8")
        return path
