#!/usr/bin/env python3
"""Manual control of the skill server (with token): check status, software stop, on-site reset.

  python scripts/robot_ctl.py status
  python scripts/robot_ctl.py camera                # camera service status (frame rate, whether exposure is locked, recording)
  python scripts/robot_ctl.py stop --reason "person entered the work area"
  python scripts/robot_ctl.py reset --by <name>        # after on-site staff confirm the robot state (a protective stop must first be cleared on the teach pendant)
A software stop does not replace the physical E-stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cleanproof.robot.server import auth_headers  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-url", default="http://127.0.0.1:8100")
    ap.add_argument("--camera-url", default="http://127.0.0.1:8101")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("camera")
    s = sub.add_parser("stop")
    s.add_argument("--reason", default="operator stop (robot_ctl)")
    r = sub.add_parser("reset")
    r.add_argument("--by", required=True)
    args = ap.parse_args()
    c = httpx.Client(base_url=args.skill_url, headers=auth_headers(), timeout=5.0)
    if args.cmd == "status":
        resp = c.get("/health")
    elif args.cmd == "camera":
        resp = httpx.get(f"{args.camera_url.rstrip('/')}/health", headers=auth_headers(), timeout=5.0)
    elif args.cmd == "stop":
        resp = c.post("/stop", json={"reason": args.reason})
    else:
        resp = c.post("/reset", json={"by": args.by})
    print(resp.status_code, json.dumps(resp.json(), ensure_ascii=False, indent=1))
    return 0 if resp.status_code == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
