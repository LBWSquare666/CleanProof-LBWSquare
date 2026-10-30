#!/bin/bash
# Robot computer: install CleanProof (the "install script" in the second half of dp2 in the video's "hardware deployment" section). The code is synced from the GPU workstation (scripts/robot_pc/sync_to_robot_pc.sh) or cloned with git.
#
#   scripts/install/install_robot.sh [--backend ur3e|mujoco] [--site ur3e_site] [--scenario S1_light] [--no-start] [--dry-run]
#
# Steps: environment check → Python environment (ur3e: reuses scripts/robot_pc/setup_env.sh, installs ur_rtde and pyrealsense2 and self-tests them) → shared token check
#       → two systemd user services, camera service and skill server (both listen on 127.0.0.1 only; the GPU workstation reaches them through the SSH tunnel) → start and check status.
# Safety: the skill server starts in shadow mode (read-only telemetry, the robot is not allowed to move). dry_run / full modes need on-site staff to run
#       python -m cleanproof.robot.commissioning arm --by <name> to arm it, and then switch manually; this script does not do that step.
#       mujoco backend = simulated self-test (camera frames come from the skill server itself, pace --pace 3 same as the desktop app); no motion authorization issue.
# --dry-run only prints the commands to run and the files to write; nothing is changed.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

BACKEND=ur3e
SITE=${CLEANPROOF_SITE:-ur3e_site}
NO_START=0
DRY=0
SCENARIO=S1_light
need() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }   # need <option> $#
while [ $# -gt 0 ]; do
  case "$1" in
    --backend) need "${1}" $#; BACKEND=$2; shift 2 ;;
    --site) need "${1}" $#; SITE=$2; shift 2 ;;
    --scenario) need "${1}" $#; SCENARIO=$2; shift 2 ;;
    --no-start) NO_START=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$BACKEND" in ur3e|mujoco) ;; *) echo "--backend only supports ur3e (real robot, RTDE) or mujoco (simulated self-test)" >&2; exit 2 ;; esac

CFG=${CLEANPROOF_CONFIG_DIR:-$HOME/.config/cleanproof}
UNITS=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
TOKEN=$CFG/robot_token
CFG_ENV=""
[ -z "${CLEANPROOF_CONFIG_DIR:-}" ] || CFG_ENV="Environment=\"CLEANPROOF_CONFIG_DIR=$CFG\""

step() { echo; echo "[$1] $2"; }
run() {
  echo "+ $*"
  [ "$DRY" = 1 ] || "$@"
}
write_file() {
  local path=$1 mode=$2
  if [ "$DRY" = 1 ]; then
    echo "+ writing $path (mode $mode):"; sed 's/^/    /'
  else
    mkdir -p "$(dirname "$path")"
    cat > "$path"
    chmod "$mode" "$path"
    echo "+ wrote $path"
  fi
}

step 1/5 "environment check"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' && echo "Python ≥ 3.10: OK" || { echo "Python ≥ 3.10: MISSING" >&2; exit 1; }
command -v tmux >/dev/null 2>&1 && echo "tmux: OK" || echo "tmux: MISSING (only affects scripts/robot_pc/start_servers.sh, used for manual troubleshooting)"

step 2/5 "CleanProof Python environment (backend $BACKEND)"
if [ "$BACKEND" = ur3e ]; then
  run scripts/robot_pc/setup_env.sh              # venv + ur_rtde + pyrealsense2 + method and import self-test (no robot connection, no camera)
else
  [ -d .venv ] || run python3 -m venv .venv
  run .venv/bin/pip install -q -e '.[ui,sim]'
fi

step 3/5 "shared token ($TOKEN)"
if [ -f "$TOKEN" ] && [ "$(stat -c %a "$TOKEN")" = 600 ]; then
  echo "robot_token: OK"
else
  echo "robot_token: MISSING (run scripts/install/install_edge.sh --pair $(hostname) on the GPU workstation)"
fi

step 4/5 "services: cleanproof-skill (8100, shadow mode)$([ "$BACKEND" = ur3e ] && echo ", cleanproof-camera (8101)")"
SERVICES="cleanproof-skill.service"
if [ "$BACKEND" = ur3e ]; then                 # the simulated backend's skill server provides its own camera frames (--source topdown), no camera service needed
  SERVICES="cleanproof-camera.service $SERVICES"
  write_file "$UNITS/cleanproof-camera.service" 644 <<EOF
[Unit]
Description=CleanProof camera service

[Service]
WorkingDirectory=$REPO
Environment=CLEANPROOF_SITE=$SITE
$CFG_ENV
ExecStart=$REPO/.venv/bin/python -m cleanproof.perception.camera_server --bind 127.0.0.1 --port 8101
Restart=on-failure

[Install]
WantedBy=default.target
EOF
fi
SIM_ARGS=""
[ "$BACKEND" = mujoco ] && SIM_ARGS=" --scenario $SCENARIO --pace 3"      # scenario and pace for the simulated backend; not passed for ur3e
write_file "$UNITS/cleanproof-skill.service" 644 <<EOF
[Unit]
Description=CleanProof skill server (the only process that moves the robot)

[Service]
WorkingDirectory=$REPO
Environment=CLEANPROOF_SITE=$SITE
$CFG_ENV
ExecStart=$REPO/.venv/bin/python -m cleanproof.robot.server --backend $BACKEND --mode shadow --bind 127.0.0.1 --port 8100$SIM_ARGS
Restart=on-failure

[Install]
WantedBy=default.target
EOF
run systemctl --user daemon-reload

step 5/5 "start and check status"
if [ "$NO_START" = 0 ]; then
  # shellcheck disable=SC2086
  run systemctl --user enable --now $SERVICES
  run .venv/bin/python scripts/robot_ctl.py status
else
  echo "not started (--no-start)"
fi
echo
echo "Robot computer install complete. Go back to the GPU workstation and run scripts/install/install_edge.sh --pair $(hostname)."
