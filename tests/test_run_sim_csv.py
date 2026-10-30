"""Result table of scripts/run_sim.py: the new column platform is in FIELDS, and an old header gets the column added by migration."""
import csv
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("run_sim", Path(__file__).resolve().parents[1] / "scripts" / "run_sim.py")
run_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_sim)


def _row(**kw):
    r = {k: "" for k in run_sim.FIELDS}
    r.update(kw)
    return r


def test_fields_have_platform_and_row_is_written(tmp_path):
    out = tmp_path / "r.csv"
    run_sim.append_rows(out, [_row(scenario="S2_dried", platform="ur3e")])
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert rows[0]["platform"] == "ur3e" and "platform" in run_sim.FIELDS


def test_old_header_is_migrated_with_defaults(tmp_path):
    out = tmp_path / "r.csv"
    old = [f for f in run_sim.FIELDS if f not in ("robot", "mode", "platform")]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=old)
        w.writeheader()
        w.writerow({k: "x" for k in old})
    run_sim.append_rows(out, [_row(scenario="new", platform="panda")])
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    assert [r["platform"] for r in rows] == ["ur3e", "panda"] and rows[0]["robot"] == "mujoco" and rows[0]["mode"] == "full"
