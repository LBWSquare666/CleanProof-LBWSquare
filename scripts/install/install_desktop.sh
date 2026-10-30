#!/bin/bash
# CleanProof desktop app (Linux): create .venv-desktop (system Python + pywebview, so it can see the system GTK WebKit2) and write the application menu launcher.
#
#   scripts/install/install_desktop.sh [--dry-run]
#
# Prerequisite: scripts/install/install_edge.sh has been run (project .venv, where the service itself runs). The system needs python3-gi and gir1.2-webkit2-4.1 (or 4.0):
#   sudo apt install python3-gi gir1.2-webkit2-4.1        # this script never calls sudo; it only prints a hint if they are missing
# After installing: the "CleanProof" entry in the application menu, or on the command line: .venv-desktop/bin/python -m cleanproof.ui.desktop
# --dry-run only prints the commands to run and the files to write; nothing is changed.
# The launcher does not pin a site: the site is chosen in the first-run wizard and then taken from ~/.config/cleanproof/setup.json (a site changed in the wizard still applies after a restart).
# To pick a site temporarily: CLEANPROOF_SITE=<site> .venv-desktop/bin/python -m cleanproof.ui.desktop (or --site).
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD
DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --site) echo "Note: the launcher no longer pins a site (--site $2 is ignored); choose the site in the first-run wizard, after which setup.json is used." >&2; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
APPS=${XDG_DATA_HOME:-$HOME/.local/share}/applications
SYS_PY=${SYS_PY:-/usr/bin/python3}

run() { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
write_file() {   # write_file <path>, content is read from stdin
  if [ "$DRY" = 1 ]; then echo "+ writing $1:"; sed 's/^/    /'; else mkdir -p "$(dirname "$1")"; cat > "$1"; echo "+ wrote $1"; fi
}

echo "[1/3] system GTK WebKit2"
if "$SYS_PY" -c 'import gi
try:
    gi.require_version("WebKit2", "4.1")
except ValueError:
    gi.require_version("WebKit2", "4.0")
from gi.repository import WebKit2' 2>/dev/null; then
  echo "GTK WebKit2: OK"
else
  echo "GTK WebKit2: MISSING (sudo apt install python3-gi gir1.2-webkit2-4.1)"; [ "$DRY" = 1 ] || exit 1
fi
[ -x .venv/bin/python ] || { echo "project .venv is missing: run scripts/install/install_edge.sh first" >&2; [ "$DRY" = 1 ] || exit 1; }

echo "[2/3] .venv-desktop (pywebview)"
[ -d .venv-desktop ] || run "$SYS_PY" -m venv --system-site-packages .venv-desktop
run .venv-desktop/bin/pip install -q pywebview

echo "[3/3] application menu launcher"
write_file "$APPS/cleanproof.desktop" <<EOF2
[Desktop Entry]
Type=Application
Name=CleanProof
Comment=Physical AI cleaning software · appropriate force, varied surfaces, proven clean
Exec=$REPO/.venv-desktop/bin/python -m cleanproof.ui.desktop
Path=$REPO
Icon=$REPO/assets/brand_icons/cleanproof_app.png
Terminal=false
Categories=Utility;Development;
StartupWMClass=CleanProof
EOF2
echo
echo "Install complete: CleanProof is in the application menu (or run .venv-desktop/bin/python -m cleanproof.ui.desktop)."
