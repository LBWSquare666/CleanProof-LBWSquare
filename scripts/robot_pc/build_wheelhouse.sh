#!/bin/bash
# Run on the GPU workstation (optional): pre-download all dependency wheels for robot-pc (x86_64, Python 3.10) so it can be installed offline when the site network is poor.
# Must run pip download with the 3.10 interpreter: environment markers in the dependencies (e.g. python_version < "3.11") are evaluated by the interpreter running pip.
set -euo pipefail
. "$(dirname "$0")/common.sh"
cd "$REPO"
mkdir -p wheelhouse
uvx --python 3.10 pip download --dest wheelhouse --only-binary=:all: --python-version 3.10 \
    --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
    pip wheel setuptools 'ur_rtde>=1.5.9,<1.7' pyrealsense2 pytest fastapi uvicorn \
    'numpy>=1.26' 'pydantic>=2.6' 'pyyaml>=6' 'python-dotenv>=1.0' 'openai>=1.40' 'httpx>=0.27' 'pillow>=10' 'opencv-python-headless>=4.9'
du -sh wheelhouse
