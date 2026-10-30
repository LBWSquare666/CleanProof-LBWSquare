#!/bin/bash
# GPU workstation → robot-pc: copy summary.json / events.jsonl of a real-robot run to robot-pc, as gate evidence for the commissioning record.
# Usage: scripts/robot_pc/push_evidence.sh runs/real/<run_id>
set -euo pipefail
. "$(dirname "$0")/common.sh"
run=${1:?run directory}
id=$(basename "$run")
ssh "$ROBOT_PC_HOST" "mkdir -p $ROBOT_PC_DIR/runs/evidence/$id"
rsync -az "$run/summary.json" "$run/events.jsonl" "$ROBOT_PC_HOST:$ROBOT_PC_DIR/runs/evidence/$id/"
echo "register on robot-pc (run by on-site staff): python -m cleanproof.robot.commissioning record --backend ur3e --stage <shadow|dry_run|full> --evidence runs/evidence/$id/summary.json --by <name>"
