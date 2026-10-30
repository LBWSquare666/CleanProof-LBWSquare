"""Config loading: locate the repo root, read YAML, load the secrets environment."""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ENV_HOME = "CLEANPROOF_HOME"               # read-only resource root (configs/, assets/, scripts/); when set it takes precedence over the detection below
ENV_DATA_DIR = "CLEANPROOF_DATA_DIR"       # writable user data root (site configs, run records); when unset the old behavior is kept and data is written under REPO_ROOT
ENV_CONFIG_DIR = "CLEANPROOF_CONFIG_DIR"   # user config dir (secrets.env, robot_token, robot_host, setup.json, app.json)


def _resolve_root() -> Path:
    """Resource root: $CLEANPROOF_HOME; the repository root in a source checkout; the package's _bundle after a pip / deb install (the wheel build puts configs/, assets/ and scripts/ there)."""
    if os.environ.get(ENV_HOME):
        return Path(os.environ[ENV_HOME]).expanduser().resolve()
    here = Path(__file__).resolve()
    checkout = here.parents[2]
    return checkout if (checkout / "pyproject.toml").is_file() and (checkout / "configs").is_dir() else here.parents[1] / "_bundle"


REPO_ROOT = _resolve_root()
CONFIG_DIR = REPO_ROOT / "configs"


def data_dir() -> Path | None:
    """User data root: $CLEANPROOF_DATA_DIR (the launcher of an installed copy sets ~/.local/share/cleanproof); None when unset, and data stays under REPO_ROOT."""
    v = os.environ.get(ENV_DATA_DIR)
    return Path(v).expanduser() if v else None


def runs_root() -> Path:
    """Root directory of run records (runs/)."""
    d = data_dir()
    return (d / "runs") if d else REPO_ROOT / "runs"


@lru_cache(maxsize=8)
def _seed_sites(user_root: Path) -> None:
    user_root.mkdir(parents=True, exist_ok=True)
    src = CONFIG_DIR / "site"
    for d in sorted(src.iterdir()) if src.is_dir() else []:
        if d.is_dir() and not (user_root / d.name).exists():
            shutil.copytree(d, user_root / d.name)


def site_root() -> Path:
    """Root of the site directories: configs/site by default; <data root>/site when a user data root is set, with the bundled sites copied there on first use (existing ones are never overwritten),
    and the settings page and the wizard write only there, so the install directory stays read-only."""
    d = data_dir()
    if d is None:
        return CONFIG_DIR / "site"
    _seed_sites(d / "site")
    return d / "site"


def user_config_dir() -> Path:
    """User config dir: $CLEANPROOF_CONFIG_DIR, otherwise ~/.config/cleanproof. The env var is read on every call (tests and self-checks use temp dirs)."""
    return Path(os.environ.get(ENV_CONFIG_DIR) or "~/.config/cleanproof").expanduser()


def secrets_files() -> tuple[Path, ...]:
    """Secrets files, in load order: secrets.env in the user config dir, then .env in the repo root (gitignored)."""
    return (user_config_dir() / "secrets.env", REPO_ROOT / ".env")


ENV_SITE = "CLEANPROOF_SITE"     # real-robot site: a same-named YAML under configs/site/<site>/ overrides the defaults (calibration results, measured geometry)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def site_dir(site: str | None = None) -> Path | None:
    site = site if site is not None else os.environ.get(ENV_SITE, "")
    return (site_root() / site) if site else None


_SITE_SET_BY_SIM: str | None = None


def use_site(site: str | None) -> None:
    """Site registered by a simulated scene / recording (configs/site/<site>/, e.g. a larger dining-table workspace): set it as this process's CLEANPROOF_SITE.

    With an empty site, only clear a value previously set by this function (a CLEANPROOF_SITE set by hand for a real-robot site is left alone). Afterwards load_yaml / ZoneGrid() read for this site.
    """
    global _SITE_SET_BY_SIM
    if site:
        if not (site_root() / site).is_dir():
            raise FileNotFoundError(f"site config does not exist: configs/site/{site}")
        os.environ[ENV_SITE] = site
        _SITE_SET_BY_SIM = site
    elif _SITE_SET_BY_SIM is not None and os.environ.get(ENV_SITE) == _SITE_SET_BY_SIM:
        os.environ.pop(ENV_SITE, None)
        _SITE_SET_BY_SIM = None


def current_site() -> str:
    return os.environ.get(ENV_SITE, "")


def load_yaml(name_or_path: str | Path, site: str | None = None) -> dict[str, Any]:
    """Read YAML under configs/ (a file name or an absolute path can be given).

    If env var CLEANPROOF_SITE is set and configs/site/<site>/<name> exists, deep-merge it onto the default config
    (only for configs read by file name under configs/; simulation and tests do not set this variable, so behavior is unchanged).
    site: explicit site (ignores the env var; "" = no site overlay), used when the software lists several sites at once.
    """
    path = Path(name_or_path)
    by_name = not path.is_absolute() and not path.exists()
    if by_name:
        path = CONFIG_DIR / path
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    sd = site_dir(site)
    if by_name and sd is not None:
        over = sd / Path(name_or_path)
        if over.is_file():
            with open(over, encoding="utf-8") as f:
                data = _deep_merge(data, yaml.safe_load(f) or {})
    return data


@lru_cache(maxsize=8)
def _load_secrets(files: tuple[Path, ...]) -> tuple[str, ...]:
    loaded = []
    for path in files:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(str(path))
    return tuple(loaded)


def load_secrets() -> tuple[str, ...]:
    """Load the secrets files in order (without overriding existing environment variables) and return the files actually loaded. A given set of files is loaded only once;
    the config dir ($CLEANPROOF_CONFIG_DIR) is determined at call time, so changing it loads the files in the new dir."""
    return _load_secrets(secrets_files())


load_secrets.cache_clear = _load_secrets.cache_clear  # type: ignore[attr-defined]  compatible with the former lru_cache interface


def env_status(names: list[str]) -> dict[str, str]:
    """Only report whether the secret exists, never return the value."""
    load_secrets()
    return {name: ("OK" if os.environ.get(name) else "MISSING") for name in names}


def expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()
