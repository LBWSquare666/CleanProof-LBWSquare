#!/bin/bash
# Run on robot-pc: start the camera service and skill server in the tmux session cleanproof (both listen on 127.0.0.1 only).
# Usage: scripts/robot_pc/start_servers.sh shadow|dry_run|full
#   dry_run / full: on-site staff first run `python -m cleanproof.robot.commissioning arm --by <name>` (valid for 300 s), then start;
#   the gate requires the previous stage to be registered as passed (see commissioning status).
set -euo pipefail
MODE=${1:?shadow|dry_run|full}
cd "$(dirname "$0")/../.."
SITE=${CLEANPROOF_SITE:-ur3e_site}
mkdir -p runs ~/cleanproof_data/recordings
PY=$PWD/.venv/bin/python
tmux has-session -t cleanproof 2>/dev/null || tmux new-session -d -s cleanproof -n shell
if ! $PY scripts/robot_ctl.py camera >/dev/null 2>&1; then
  tmux new-window -t cleanproof -n camera "cd $PWD && CLEANPROOF_SITE=$SITE $PY -m cleanproof.perception.camera_server 2>&1 | tee -a runs/camera_server.log"
fi
tmux kill-window -t cleanproof:skill 2>/dev/null || true
tmux new-window -t cleanproof -n skill "cd $PWD && CLEANPROOF_SITE=$SITE $PY -m cleanproof.robot.server --backend ur3e --mode $MODE 2>&1 | tee -a runs/skill_server.log; echo 'skill server exited, press Enter to close'; read"
sleep 4
$PY scripts/robot_ctl.py status || echo "skill server did not start: tmux attach -t cleanproof and look at the skill window"
