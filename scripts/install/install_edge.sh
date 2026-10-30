#!/bin/bash
# GPU workstation (the host machine with an NVIDIA GPU): install CleanProof. The video's "hardware deployment" section (dp2 / dp3) shows exactly these two steps.
#
#   scripts/install/install_edge.sh [--site home_dining] [--skip-model] [--no-start] [--dry-run]
#   scripts/install/install_edge.sh --pair <robot computer SSH host name> [--no-start] [--dry-run]
#
# Install: Python environment → Cosmos 3 runtime (vLLM, local GPU) → secrets file template (NEBIUS_API_KEY; only reports OK / MISSING, never prints the value)
#       → shared token (auth for the skill server / camera service, 0600) → console systemd user service → start Cosmos 3.
# Pair: copy the shared token to the robot computer (0600) → create the SSH tunnel service (the two services on the robot computer listen on localhost only and are mapped through the tunnel to local 8100 / 8101)
#       → run the deployment check (cleanproof/ui/deploy_checks.py).
#       --pair with --no-start: only write the tunnel service file, do not start it (prints the start command).
# Environment variables (--pair): CP_SSH_OPTS (extra ssh options, e.g. "-F cfg"; used by both ssh and rsync), CP_REMOTE_CONFIG_DIR (config directory on the robot computer,
#       default .config/cleanproof, relative to its home directory), SKILL_PORT / CAMERA_PORT (local tunnel ports, default 8100 / 8101),
#       REMOTE_SKILL_PORT / REMOTE_CAMERA_PORT (ports on the robot computer, default same as local).
# --dry-run only prints the commands to run and the files to write; nothing is changed.
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD

SITE=
SKIP_MODEL=0
NO_START=0
DRY=0
PAIR=""
need() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 2; }; }   # need <option> $#
while [ $# -gt 0 ]; do
  case "$1" in
    --site) need "${1}" $#; SITE=$2; shift 2 ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    --no-start) NO_START=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --pair) need "${1}" $#; PAIR=$2; shift 2 ;;
    -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

CFG=${CLEANPROOF_CONFIG_DIR:-$HOME/.config/cleanproof}
UNITS=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
SECRETS=$CFG/secrets.env
TOKEN=$CFG/robot_token
# With a custom config directory, write it into the systemd services (services do not see a login shell's environment variables)
SITE_ENV=""       # pin the site only when --site is given explicitly; otherwise the site chosen in the wizard (setup.json) stays in effect after a restart
[ -z "$SITE" ] || SITE_ENV="Environment=CLEANPROOF_SITE=$SITE"
CFG_ENV=""
[ -z "${CLEANPROOF_CONFIG_DIR:-}" ] || CFG_ENV="Environment=\"CLEANPROOF_CONFIG_DIR=$CFG\""
COSMOS_VENV=${COSMOS_VENV:-.venv-vllm30}
COSMOS_MODEL=${COSMOS_MODEL:-nvidia/Cosmos3-Nano}

step() { echo; echo "[$1] $2"; }
run() {   # execute, or just print with --dry-run
  echo "+ $*"
  [ "$DRY" = 1 ] || "$@"
}
write_file() {   # write_file <path> <mode>, content is read from stdin
  local path=$1 mode=$2
  if [ "$DRY" = 1 ]; then
    echo "+ writing $path (mode $mode):"; sed 's/^/    /'
  else
    mkdir -p "$(dirname "$path")"
    (umask 077; cat > "$path")
    chmod "$mode" "$path"
    echo "+ wrote $path"
  fi
}

if [ -n "$PAIR" ]; then
  # ------------------------------------------------------------------ pairing
  SKILL_PORT=${SKILL_PORT:-8100}
  CAMERA_PORT=${CAMERA_PORT:-8101}
  REMOTE_SKILL_PORT=${REMOTE_SKILL_PORT:-$SKILL_PORT}
  REMOTE_CAMERA_PORT=${REMOTE_CAMERA_PORT:-$CAMERA_PORT}
  RCFG=${CP_REMOTE_CONFIG_DIR:-.config/cleanproof}
  SSH_OPTS=${CP_SSH_OPTS:-}
  step 1/3 "shared token → $PAIR (0600, contents not printed)"
  [ -f "$TOKEN" ] || [ "$DRY" = 1 ] || { echo "$TOKEN is missing: run the install once without --pair first" >&2; exit 1; }
  # shellcheck disable=SC2086
  run ssh $SSH_OPTS "$PAIR" "mkdir -p '$RCFG' && chmod 700 '$RCFG'"
  run rsync -a --chmod=F600 -e "ssh $SSH_OPTS" "$TOKEN" "$PAIR:$RCFG/robot_token"
  echo "$PAIR" | write_file "$CFG/robot_host" 644          # the software wizard and settings page show this robot computer
  # The ssh options used for pairing are stored in the config directory: tunnel.sh / push_site.sh (including the in-app calibration sync) read them when CP_SSH_OPTS is not set explicitly
  [ -z "$SSH_OPTS" ] || echo "$SSH_OPTS" | write_file "$CFG/ssh_opts" 644
  step 2/3 "encrypted link: SSH tunnel service cleanproof-tunnel (local $SKILL_PORT skill server, $CAMERA_PORT camera service)"
  write_file "$UNITS/cleanproof-tunnel.service" 644 <<EOF
[Unit]
Description=CleanProof encrypted link to the robot computer ($PAIR)
After=network-online.target

[Service]
Environment=ROBOT_PC_HOST=$PAIR
$CFG_ENV
Environment=SKILL_PORT=$SKILL_PORT
Environment=CAMERA_PORT=$CAMERA_PORT
Environment=REMOTE_SKILL_PORT=$REMOTE_SKILL_PORT
Environment=REMOTE_CAMERA_PORT=$REMOTE_CAMERA_PORT
Environment="CP_SSH_OPTS=$SSH_OPTS"
ExecStart=$REPO/scripts/robot_pc/tunnel.sh
Restart=always
RestartSec=2

[Install]
WantedBy=default.target
EOF
  if [ "$NO_START" = 0 ]; then
    run systemctl --user daemon-reload
    run systemctl --user enable --now cleanproof-tunnel.service
  else
    echo "tunnel not started (--no-start). To start it: systemctl --user daemon-reload && systemctl --user enable --now cleanproof-tunnel.service"
  fi
  step 3/3 "deployment check (same as the check in the software's first-run wizard)"
  CHECK=("$REPO/.venv/bin/python" -m cleanproof.ui.deploy_checks --site "${SITE:-home_dining}"
         --skill-url "http://127.0.0.1:$SKILL_PORT" --camera-url "http://127.0.0.1:$CAMERA_PORT")
  if [ "$NO_START" = 0 ]; then
    run "${CHECK[@]}"
  else
    echo "skipped (tunnel is not started yet). Run after starting the tunnel: ${CHECK[*]}"
  fi
  exit 0
fi

# -------------------------------------------------------------------- install
step 1/6 "environment check"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' && echo "Python ≥ 3.10: OK" || { echo "Python ≥ 3.10: MISSING" >&2; exit 1; }
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA GPU: OK ($(nvidia-smi -L | wc -l) found)"
else
  echo "NVIDIA GPU: MISSING (Cosmos 3 needs a local NVIDIA GPU; without a GPU the verifier falls back to classical vision)"
fi

step 2/6 "CleanProof Python environment (.venv)"
[ -d .venv ] || run python3 -m venv .venv
run .venv/bin/pip install -q -U pip wheel
run .venv/bin/pip install -q -e '.[ui]'

step 3/6 "Cosmos 3 runtime (vLLM 0.30.0, $COSMOS_VENV)"
[ -x "$COSMOS_VENV/bin/vllm" ] || { run python3 -m venv "$COSMOS_VENV"; run "$COSMOS_VENV/bin/pip" install -q 'vllm==0.30.0'; }
if [ "$SKIP_MODEL" = 0 ]; then
  run "$COSMOS_VENV/bin/huggingface-cli" download "$COSMOS_MODEL"
else
  echo "skipping model weight download (--skip-model)"
fi

step 4/6 "Nebius Token Factory key ($SECRETS, 0600)"
if [ -f "$SECRETS" ] && grep -q '^NEBIUS_API_KEY=.\+' "$SECRETS"; then
  echo "NEBIUS_API_KEY: OK"
else
  [ -f "$SECRETS" ] || write_file "$SECRETS" 600 <<'EOF'
# CleanProof secrets (local machine only; never in git or logs). Fill it in on the Models step of the first-run wizard, or edit this file directly.
NEBIUS_API_KEY=
EOF
  echo "NEBIUS_API_KEY: MISSING (fill it in the first-run wizard, or edit $SECRETS)"
fi

step 5/6 "shared token ($TOKEN, 0600; copied to the robot computer when pairing)"
if [ -f "$TOKEN" ]; then
  echo "robot_token: OK"
else
  if [ "$DRY" = 1 ]; then
    echo "+ generate 32-byte random token, write to $TOKEN (mode 600)"
  else
    python3 -c 'import secrets; print(secrets.token_urlsafe(32))' | write_file "$TOKEN" 600
  fi
fi

step 6/6 "services: console cleanproof-console (http://127.0.0.1:8092), Cosmos 3 (local GPU, port 8001)"
write_file "$UNITS/cleanproof-console.service" 644 <<EOF
[Unit]
Description=CleanProof software (operator console)
After=network-online.target

[Service]
WorkingDirectory=$REPO
$SITE_ENV
$CFG_ENV
ExecStart=$REPO/.venv/bin/python -m cleanproof.ui.operator${SITE:+ --site $SITE} --policy nemotron --verifier vlm
Restart=on-failure

[Install]
WantedBy=default.target
EOF
run systemctl --user daemon-reload
if [ "$NO_START" = 0 ]; then
  run systemctl --user enable --now cleanproof-console.service
  run env MODEL="$COSMOS_MODEL" VENV="$COSMOS_VENV" "$REPO/scripts/server/serve_cosmos.sh"
fi
echo
echo "Install complete. Next: run scripts/install/install_robot.sh on the robot computer, then run $0 --pair <robot computer> here."
