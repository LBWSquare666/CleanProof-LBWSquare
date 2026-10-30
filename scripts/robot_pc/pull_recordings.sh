#!/bin/bash
# robot-pc → GPU workstation: fetch the camera service recordings and skill server logs (stored in store, not in git).
set -euo pipefail
. "$(dirname "$0")/common.sh"
DEST=${DEST:-./store/real}
mkdir -p "$DEST/recordings" "$DEST/skill_server"
rsync -az "$ROBOT_PC_HOST:cleanproof_data/recordings/" "$DEST/recordings/"
rsync -az "$ROBOT_PC_HOST:$ROBOT_PC_DIR/runs/skill_server/" "$DEST/skill_server/"
du -sh "$DEST"/*
