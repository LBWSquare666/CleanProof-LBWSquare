#!/bin/bash
# GPU workstation: map the skill server and camera service of the robot computer to 127.0.0.1 on this machine (both services listen only on localhost on the robot computer,
# so other machines on the campus network cannot reach them directly). Reconnects automatically after a drop. Best kept running in tmux or under systemd-run --user.
#
# Environment variables (all have defaults):
#   ROBOT_PC_HOST           SSH host name of the robot computer (common.sh, default robot-pc)
#   SKILL_PORT          local port: skill server (default 8100)
#   CAMERA_PORT         local port: camera service (default 8101)
#   REMOTE_SKILL_PORT   skill server port on the robot computer (default = SKILL_PORT)
#   REMOTE_CAMERA_PORT  camera service port on the robot computer (default = CAMERA_PORT)
#   CP_SSH_OPTS         extra ssh options (default read from <config dir>/ssh_opts), e.g. "-F /path/ssh_config -i key" (split on spaces)
set -uo pipefail
. "$(dirname "$0")/common.sh"
SKILL_PORT=${SKILL_PORT:-8100}
CAMERA_PORT=${CAMERA_PORT:-8101}
REMOTE_SKILL_PORT=${REMOTE_SKILL_PORT:-$SKILL_PORT}
REMOTE_CAMERA_PORT=${REMOTE_CAMERA_PORT:-$CAMERA_PORT}
# shellcheck disable=SC2206
SSH_OPTS=(${CP_SSH_OPTS:-})
while true; do
  ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=5 -o ServerAliveCountMax=3 "${SSH_OPTS[@]}" \
      -L "127.0.0.1:$SKILL_PORT:127.0.0.1:$REMOTE_SKILL_PORT" -L "127.0.0.1:$CAMERA_PORT:127.0.0.1:$REMOTE_CAMERA_PORT" "$ROBOT_PC_HOST"
  echo "$(date +%T) tunnel dropped, reconnecting in 2 s" >&2
  sleep 2
done
