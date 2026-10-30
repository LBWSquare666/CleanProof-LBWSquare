"""Command line:
  python -m cleanproof.ui --replay runs/<run_id>
  python -m cleanproof.ui --live S2_dried --policy rules --verifier cv [--source topdown] [--no-approval] [--pace 1]

Binds to 127.0.0.1 only by default. --policy nemotron / --verifier vlm / --audit create an LLMClient; Token Factory incurs charges,
local Cosmos is free. There is no real-robot backend: see the TODO(skill server) in session.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.config import runs_root
from ..sim.run import POLICIES, VERIFIERS


def main() -> int:
    ap = argparse.ArgumentParser(description="CleanProof web console (replay / live simulation)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--replay", metavar="RUN_DIR", help="replay a run directory")
    mode.add_argument("--live", metavar="SCENARIO", help="run a simulation scenario in a background thread (e.g. S2_dried)")
    ap.add_argument("--policy", default="rules", choices=POLICIES)
    ap.add_argument("--verifier", default="cv", choices=VERIFIERS)
    ap.add_argument("--source", default="topdown", choices=("topdown", "render"))
    ap.add_argument("--no-approval", action="store_true", help="live mode does not wait for operator approval")
    ap.add_argument("--autostart", action="store_true", help="start right after the server is up, without waiting for Start on the page")
    ap.add_argument("--pace", type=float, default=1.0, help="throttle the simulation to about N times real time (0 = run as fast as possible)")
    ap.add_argument("--audit", action="store_true", help="call the audit role after the run (Token Factory)")
    ap.add_argument("--runs-dir", default=str(runs_root()))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import uvicorn

    from .app import create_app
    from .session import LiveSession, ReplaySession

    if args.replay:
        session = ReplaySession(args.replay)
    else:
        client = None
        if args.policy == "nemotron" or args.verifier == "vlm" or args.audit:
            from ..common.llm_client import LLMClient

            client = LLMClient()
        session = LiveSession(args.live, args.policy, args.verifier, args.source, Path(args.runs_dir),
                              require_approval=not args.no_approval, pace=args.pace, client=client, audit=args.audit)
        if args.autostart:
            session.start()
    app = create_app(session)
    print(f"CleanProof Console ({session.mode}) → http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
