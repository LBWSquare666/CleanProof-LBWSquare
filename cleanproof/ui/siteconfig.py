"""Site config self-check: when a YAML under configs/site/<site>/ is wrong, say which file, which line and what is wrong, instead of 500 / a blank page.

It is used at UI startup (config_errors of /api/meta), in the settings page and before a job starts; before a job it blocks errors such as rows: 0, a job_scenario that does not exist, and a negative number in the force table.
Reads the config only and changes no file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from ..common.config import CONFIG_DIR, REPO_ROOT

# files in the site directory that are read (a missing one is not an error)
SITE_FILES = ("site.yaml", "zones.yaml", "force_table.yaml", "models.yaml", "operator.yaml", "sim.yaml", "robot.yaml", "platforms.yaml")


class ConfigError(Exception):
    """The site config has problems; problems is [{file, line, message}]."""

    def __init__(self, problems: list[dict[str, Any]]):
        self.problems = problems
        super().__init__(describe(problems))


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def describe(problems: list[dict[str, Any]]) -> str:
    return "; ".join(f"{p['file']}{', line ' + str(p['line']) if p.get('line') else ''}: {p['message']}" for p in problems[:3])


def _line(text: str, path: tuple[str, ...]) -> Optional[int]:
    """Key path → line number (if not found, the line number of the nearest level)."""
    try:
        node = yaml.compose(text)
    except yaml.YAMLError:
        return None
    line = node.start_mark.line + 1 if node is not None else None
    for k in path:
        if not isinstance(node, yaml.MappingNode):
            break
        nxt = next(((kn, vn) for kn, vn in node.value if getattr(kn, "value", None) == k), None)
        if nxt is None:
            break
        line, node = nxt[0].start_mark.line + 1, nxt[1]
    return line


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_site(site: str, site_dir: Path) -> list[dict[str, Any]]:
    """List of problems in the site directory (empty = none)."""
    out: list[dict[str, Any]] = []
    data: dict[str, Any] = {}
    texts: dict[str, str] = {}

    def add(name: str, path: tuple[str, ...], msg: str) -> None:
        out.append({"file": _rel(site_dir / name), "line": _line(texts.get(name, ""), path), "message": msg})

    for name in SITE_FILES:
        f = site_dir / name
        if not f.is_file():
            continue
        try:
            texts[name] = f.read_text(encoding="utf-8")
            val = yaml.safe_load(texts[name])
        except yaml.YAMLError as e:
            mk = getattr(e, "problem_mark", None)
            out.append({"file": _rel(f), "line": mk.line + 1 if mk else None,
                        "message": "YAML syntax error: " + str(getattr(e, "problem", None) or "cannot be read").strip()})
            continue
        except (OSError, UnicodeDecodeError) as e:
            out.append({"file": _rel(f), "line": None, "message": f"The file cannot be read ({type(e).__name__})."})
            continue
        if val is not None and not isinstance(val, dict):
            out.append({"file": _rel(f), "line": 1, "message": "The file must be a list of 'key: value' entries."})
            continue
        data[name] = val or {}
    zones = data.get("zones.yaml", {}).get("work_area")
    if zones is not None:
        if not isinstance(zones, dict):
            add("zones.yaml", ("work_area",), "work_area must be a block with rows, cols and size_xy.")
        else:
            for k, hi in (("rows", 26), ("cols", 99)):
                v = zones.get(k)
                if v is not None and (not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= hi):
                    add("zones.yaml", ("work_area", k), f"work_area.{k} must be a whole number from 1 to {hi} (found {v!r}).")
            sz = zones.get("size_xy")
            if sz is not None and not (isinstance(sz, list) and len(sz) == 2 and all(_num(x) and x > 0 for x in sz)):
                add("zones.yaml", ("work_area", "size_xy"), "work_area.size_xy must be two positive numbers (metres).")
    elif "zones.yaml" in data and (site_dir / "site.yaml").is_file() and "work_area" in data["zones.yaml"]:
        add("zones.yaml", ("work_area",), "work_area is empty.")
    js = data.get("site.yaml", {}).get("job_scenario")
    if js is not None and not (isinstance(js, str) and (CONFIG_DIR / "scenarios" / f"{js}.yaml").is_file()):
        add("site.yaml", ("job_scenario",), f"job_scenario {js!r} is not a known job template (see configs/scenarios/).")
    ft = data.get("force_table.yaml", {})
    for k in ("min_n", "hard_max_n", "escalate_step_n"):
        if k in ft and not (_num(ft[k]) and ft[k] > 0):
            add("force_table.yaml", (k,), f"{k} must be a positive number of newtons (found {ft[k]!r}).")
    for sk, sv in (ft.get("surfaces") or {}).items():
        if isinstance(sv, dict) and "max_n" in sv and not (_num(sv["max_n"]) and sv["max_n"] > 0):
            add("force_table.yaml", ("surfaces", str(sk), "max_n"), f"surfaces.{sk}.max_n must be a positive number of newtons (found {sv['max_n']!r}).")
    for stain, row in (ft.get("forces") or {}).items():
        for surf, v in (row or {}).items() if isinstance(row, dict) else []:
            if not (_num(v) and v > 0):
                add("force_table.yaml", ("forces", str(stain), str(surf)), f"forces.{stain}.{surf} must be a positive number of newtons (found {v!r}).")
    return out


def require_valid(site: str, site_dir: Path) -> None:
    probs = check_site(site, site_dir)
    if probs:
        raise ConfigError(probs)
