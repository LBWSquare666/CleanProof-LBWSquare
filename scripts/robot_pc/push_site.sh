#!/bin/bash
# GPU workstation → robot computer: sync the site config configs/site/<site>/ (zones, calibration, reference image) over, optionally making the robot computer services use that site.
#
#   scripts/robot_pc/push_site.sh <site> [--restart] [--dry-run]
#
# Environment variables: ROBOT_PC_HOST (SSH host name of the robot computer), ROBOT_PC_DIR (deployment directory, common.sh), CP_SSH_OPTS (extra ssh options such as "-F cfg"; if unset, read <config dir>/ssh_opts saved at pairing time).
# No silent overwrite: before syncing, the existing site directory on the robot computer is backed up to <ROBOT_PC_DIR>/.site_backups/<site>-<timestamp>/ (the first sync backs up an empty directory).
# --restart: write a systemd user drop-in (cleanproof-skill.service.d/site.conf, and likewise cleanproof-camera when a camera service exists),
#            with content Environment=CLEANPROOF_SITE=<site>, then restart these services after daemon-reload.
# Does not move the robot and does not arm it: restarting the skill server only returns to its startup mode (real robot shadow).
# --dry-run only prints the commands that would run.
set -euo pipefail
. "$(dirname "$0")/common.sh"

DST=""
RESTART=0
DRY=0
need() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }   # need <option> $#
while [ $# -gt 0 ]; do
  case "$1" in
    --restart) RESTART=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    -*) echo "unknown argument $1" >&2; exit 2 ;;
    *) [ -z "$DST" ] || { echo "only one site allowed" >&2; exit 2; }; DST=$1; shift ;;
  esac
done
[ -n "$DST" ] || { echo "usage: $0 <site> [--restart] [--dry-run]" >&2; exit 2; }
case "$DST" in *[!A-Za-z0-9_-]*) echo "site name may contain only letters, digits, _ and -" >&2; exit 2 ;; esac
[ -d "$REPO/configs/site/$DST" ] || { echo "$REPO/configs/site/$DST does not exist" >&2; exit 1; }

SSH_OPTS=${CP_SSH_OPTS:-}
run() {
  echo "+ $*"
  [ "$DRY" = 1 ] || "$@"
}

# shellcheck disable=SC2086
TS=$(date +%Y%m%d-%H%M%S)
run ssh $SSH_OPTS "$ROBOT_PC_HOST" "mkdir -p '$ROBOT_PC_DIR/configs/site/$DST' '$ROBOT_PC_DIR/.site_backups' && cp -a '$ROBOT_PC_DIR/configs/site/$DST' '$ROBOT_PC_DIR/.site_backups/$DST-$TS'"
run rsync -az -e "ssh $SSH_OPTS" "$REPO/configs/site/$DST/" "$ROBOT_PC_HOST:$ROBOT_PC_DIR/configs/site/$DST/"

if [ "$RESTART" = 1 ]; then
  # one remote execution: write the drop-in for services that exist, reload and restart (skip if the camera service does not exist)
  REMOTE='set -e
D=$HOME/.config/systemd/user
UNITS=""
for s in cleanproof-skill cleanproof-camera; do
  if [ -f "$D/$s.service" ] || systemctl --user cat "$s.service" >/dev/null 2>&1; then
    mkdir -p "$D/$s.service.d"
    printf "[Service]\nEnvironment=CLEANPROOF_SITE=%s\n" "'"$DST"'" > "$D/$s.service.d/site.conf"
    UNITS="$UNITS $s.service"
  fi
done
[ -n "$UNITS" ] || { echo "no cleanproof-skill service on the robot computer (run install_robot.sh first)" >&2; exit 1; }
systemctl --user daemon-reload
systemctl --user restart $UNITS
echo "restarted: $UNITS (site '"$DST"')"'
  # shellcheck disable=SC2086
  run ssh $SSH_OPTS "$ROBOT_PC_HOST" "$REMOTE"
fi
echo "site $DST synced to $ROBOT_PC_HOST:$ROBOT_PC_DIR/configs/site/$DST/"
