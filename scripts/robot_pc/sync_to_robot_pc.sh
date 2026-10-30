#!/bin/bash
# GPU workstation → robot-pc: sync only the code and config needed to run (no videos, data, weights or keys, and do not touch the calibrated configs/site/<site>/ on robot-pc).
# Usage: scripts/robot_pc/sync_to_robot_pc.sh            (ROBOT_PC_HOST defaults to robot-pc)
set -euo pipefail
. "$(dirname "$0")/common.sh"
cd "$REPO"
rev=$(git rev-parse --short HEAD)
dirty=$(git status --porcelain -- cleanproof configs scripts pyproject.toml | grep -q . && echo dirty || echo clean)
echo "$rev $dirty $(date -u +%FT%TZ)" > DEPLOYED_FROM
ssh "$ROBOT_PC_HOST" "mkdir -p $ROBOT_PC_DIR"
rsync -az --delete --exclude '__pycache__' --exclude '*.pyc' --exclude 'configs/site/*/' \
  cleanproof configs scripts tests pyproject.toml LICENSE DEPLOYED_FROM "$ROBOT_PC_HOST:$ROBOT_PC_DIR/"
[ -d wheelhouse ] && rsync -az wheelhouse "$ROBOT_PC_HOST:$ROBOT_PC_DIR/"
# Shared token (auth for skill server / camera service): generated on the GPU workstation if missing, mode 0600, synced to the same path on robot-pc. Contents are not printed.
TOK=~/.config/cleanproof/robot_token
if [ ! -f "$TOK" ]; then
  mkdir -p ~/.config/cleanproof
  (umask 077; python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOK")
  echo "generated shared token $TOK"
fi
ssh "$ROBOT_PC_HOST" "mkdir -p ~/.config/cleanproof && chmod 700 ~/.config/cleanproof"
rsync -a --chmod=F600 "$TOK" "$ROBOT_PC_HOST:.config/cleanproof/robot_token"
echo "synced $rev ($dirty) to $ROBOT_PC_HOST:~/$ROBOT_PC_DIR"
