"""Local Cosmos smoke test (PLAN section 12 step 4).

Through the verifier role of LLMClient (configs/models.yaml → local_vllm), with the model name taken from whatever vLLM is actually serving
so Edge / Nano can be compared without changing the config, describe the cookbook example image and one simulated render,
recording latency and tokens; the first-token latency and GPU memory are read from vLLM /metrics and nvidia-smi. The summary is written to runs/w0/cosmos_smoke.json.

  .venv/bin/python scripts/server/cosmos_smoke.py [--role verifier] [--images data/cosmos_smoke/*]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cleanproof.common.config import REPO_ROOT  # noqa: E402
from cleanproof.common.llm_client import LLMClient, image_part  # noqa: E402

PROMPT = "Caption the image in detail."


def ttft_stats(base: str) -> tuple[float, int]:
    """(sum, count) of the first-token latency histogram in vLLM /metrics."""
    text = httpx.get(base.replace("/v1/", "/metrics").rstrip("/"), timeout=10).text
    s = re.search(r"^vllm:time_to_first_token_seconds_sum\{[^}]*\} ([0-9.e+-]+)", text, re.M)
    c = re.search(r"^vllm:time_to_first_token_seconds_count\{[^}]*\} ([0-9.e+-]+)", text, re.M)
    return (float(s.group(1)) if s else 0.0), (int(float(c.group(1))) if c else 0)


def gpu_mem_mib() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory,process_name",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if "VLLM" in line or "vllm" in line]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="verifier")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    client = LLMClient()
    rc = client.role_cfg(args.role)
    base = client.cfg["backends"][rc["backend"]]["base_url"]
    served = httpx.get(base + "models", timeout=10).json()["data"][0]["id"]
    images = args.images or sorted(glob.glob(str(REPO_ROOT / "data/cosmos_smoke/*")))
    renders = sorted(glob.glob(str(REPO_ROOT / "runs/sim-S2_dried-rules-cv-*/images/step00_survey.jpg")))
    images += renders[-1:]

    s0, c0 = ttft_stats(base)
    rows = []
    for path in images:
        p = Path(path)
        mime = "image/png" if p.suffix == ".png" else "image/jpeg"
        messages = [{"role": "user", "content": [image_part(p.read_bytes(), mime), {"type": "text", "text": PROMPT}]}]
        r = client.chat(args.role, messages, tag=f"cosmos_smoke/{p.name}", max_tokens=512, model=served)
        rows.append({"image": str(p.relative_to(REPO_ROOT)), "latency_s": round(r.latency_s, 2),
                     "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                     "finish_reason": r.raw.get("finish_reason"), "text": r.text[:600]})
        print(f"{p.name:24s} {r.latency_s:6.2f}s tok={r.prompt_tokens}/{r.completion_tokens}  {r.text[:100]!r}")
    s1, c1 = ttft_stats(base)
    n = c1 - c0
    summary = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "role": args.role, "backend": rc["backend"], "served_model": served,
               "mean_ttft_s": round((s1 - s0) / n, 3) if n else None, "gpu": gpu_mem_mib(), "calls": rows}
    out = REPO_ROOT / "runs" / "w0" / "cosmos_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"mean first token {summary['mean_ttft_s']}s; GPU memory {summary['gpu']} → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
