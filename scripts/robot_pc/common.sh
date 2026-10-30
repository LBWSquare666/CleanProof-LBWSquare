# Variables shared by both machines (source on the GPU workstation). The robot-pc SSH alias is configured in ~/.ssh/config (Host robot-pc, HostName is robot-pc current wireless address or Tailscale name).
ROBOT_PC_HOST=${ROBOT_PC_HOST:-robot-pc}
ROBOT_PC_DIR=${ROBOT_PC_DIR:-cleanproof}              # deployment directory on robot-pc (relative to the robot-pc home directory)
SITE=${CLEANPROOF_SITE:-ur3e_site}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
# extra ssh options: an explicit CP_SSH_OPTS takes priority, otherwise read <config dir>/ssh_opts saved at pairing time (written by install_edge.sh --pair; used by in-software calibration sync)
if [ -z "${CP_SSH_OPTS:-}" ]; then
  _cp_cfg=${CLEANPROOF_CONFIG_DIR:-$HOME/.config/cleanproof}
  [ -f "$_cp_cfg/ssh_opts" ] && CP_SSH_OPTS=$(cat "$_cp_cfg/ssh_opts")
fi
CP_SSH_OPTS=${CP_SSH_OPTS:-}
