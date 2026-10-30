#!/bin/bash
# robot-pc → GPU workstation: fetch the site calibration results (configs/site/<site>/); the agent side needs the same zones, homography and reference image.
set -euo pipefail
. "$(dirname "$0")/common.sh"
mkdir -p "$REPO/configs/site/$SITE"
rsync -az "$ROBOT_PC_HOST:$ROBOT_PC_DIR/configs/site/$SITE/" "$REPO/configs/site/$SITE/"
ls -l "$REPO/configs/site/$SITE/"
