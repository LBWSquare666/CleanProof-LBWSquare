"""Build hook: the wheel carries the read-only resources (configs/, assets/, scripts/) inside the package as cleanproof/_bundle,
which cleanproof.common.config uses as the resource root when the code does not run from a source checkout."""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
ASSET_DIRS = ("brand_icons", "fonts", "robots", "robots_photos", "surfaces")
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "raw", ".DS_Store")


class BuildPy(build_py):
    def run(self):
        super().run()
        dest = Path(self.build_lib) / "cleanproof" / "_bundle"
        shutil.rmtree(dest, ignore_errors=True)
        for name in ("configs", "scripts"):
            shutil.copytree(ROOT / name, dest / name, ignore=IGNORE)
        for name in ASSET_DIRS:
            shutil.copytree(ROOT / "assets" / name, dest / "assets" / name, ignore=IGNORE)


setup(cmdclass={"build_py": BuildPy})
