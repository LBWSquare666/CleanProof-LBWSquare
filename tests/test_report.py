"""Cleaning report: generate a single-file HTML from a real run directory, all numbers come from the run files."""

import json
import re
from pathlib import Path

import pytest

from cleanproof.common.config import REPO_ROOT
from cleanproof.report import build_report, load_run, load_trace, run_to_dict
from cleanproof.report.html import markdown_to_html
from cleanproof.report.plots import force_curve_svg, zone_map_svg


def test_load_run_structure(quick_run_dir):
    r = load_run(quick_run_dir, llm_log=None)
    summary = json.loads((quick_run_dir / "summary.json").read_text())
    assert r.status == summary["status"] == "done" and r.is_sim
    assert r.targets == summary["targets"]
    assert r.initial_snapshot.step == 0 and r.initial_snapshot.label == "survey"
    assert r.final_snapshot.label == "final"
    # the zone statuses of the final snapshot match summary.final_status
    assert {z: v.status for z, v in r.final_snapshot.zones.items() if z in r.targets} == summary["final_status"]
    # every ok wipe has a trace, and the trace is readable
    wipes = [sk for s in r.steps for sk in s.wipes if sk["status"] == "ok"]
    assert wipes and all(sk["trace"] for sk in wipes)
    tr = load_trace(quick_run_dir, wipes[0]["trace"], max_points=100)
    assert set(tr) >= {"t", "f_n", "f_ref", "state"} and len(tr["t"]) <= 100 and tr["n_samples"] >= len(tr["t"])
    d = run_to_dict(r)
    json.dumps(d)                                   # serializable for the console API
    assert d["grid"]["rows"] * d["grid"]["cols"] == len(sum(d["grid"]["ids"], []))


def test_report_renders_self_contained(quick_run_dir, tmp_path):
    out = build_report(quick_run_dir, tmp_path / "report.html")
    html = out.read_text(encoding="utf-8")
    summary = json.loads((quick_run_dir / "summary.json").read_text())
    assert "Simulation (MuJoCo)" in html and quick_run_dir.name in html
    assert "<script" not in html and "http://" not in html and "https://" not in html   # offline, no external links
    assert html.count("data:image/jpeg;base64,") >= 3        # survey, final and reference images, at least three embedded
    n_traces = sum(1 for s in load_run(quick_run_dir, llm_log=None).steps for sk in s.wipes if sk.get("trace"))
    assert html.count('<svg class="force"') == n_traces
    assert html.count('<svg class="zonemap"') == 2
    assert f"{summary['robot_time_s']:.1f} s" in html         # numbers come from summary.json
    for z in summary["targets"]:
        assert z in html
    # variant with images by relative path
    out2 = build_report(quick_run_dir, tmp_path / "rel.html", embed_images=False)
    assert 'src="images/step00_survey.jpg"' in out2.read_text(encoding="utf-8")


def test_report_cli(quick_run_dir, tmp_path):
    import subprocess
    import sys

    out = tmp_path / "cli.html"
    subprocess.run([sys.executable, "-m", "cleanproof.report", str(quick_run_dir), "-o", str(out)], check=True,
                   cwd=REPO_ROOT, capture_output=True)
    assert out.stat().st_size > 10_000


def test_markdown_subset():
    html = markdown_to_html("## Outcome\nAll **clean**.\n\n| Zone | Final |\n|---|---|\n| B2 | clean |\n\n- one\n- two\n")
    assert "<h2>Outcome</h2>" in html and "<strong>clean</strong>" in html
    assert "<th>Zone</th>" in html and "<td>B2</td>" in html and "<li>two</li>" in html


def test_svg_helpers_degrade():
    assert "no trace samples" in force_curve_svg({"t": []}, "x")
    grid = {"rows": 2, "cols": 2, "ids": [["A1", "A2"], ["B1", "B2"]]}
    svg = zone_map_svg(grid, {"A1": {"status": "residue", "confidence": 0.9, "source": "cv"}}, ["A1", "A2"], ["B1"], "t")
    assert "forbidden" in svg and "not in task" in svg and "residue" in svg and "not inspected" in svg


_NEMOTRON = sorted((REPO_ROOT / "runs").glob("sim-S2_dried-nemotron-vlm-*")) if (REPO_ROOT / "runs").is_dir() else []


@pytest.mark.skipif(not _NEMOTRON, reason="needs a local Nemotron + Cosmos run directory")
def test_report_on_real_nemotron_run(tmp_path):
    run_dir = _NEMOTRON[-1]
    r = load_run(run_dir)
    assert any(c["role"] == "decision" and c["backend"] == "tokenfactory" for c in r.calls)
    html = build_report(run_dir, tmp_path / "nemotron.html").read_text(encoding="utf-8")
    assert "nemotron" in html.lower() and "tokenfactory" in html
    if r.audit_text:
        assert "Audit summary" in html
    # the decision spend in the report matches summary.json
    assert f"${r.summary['decision_cost_usd']:.4f}" in html
    assert not re.search(r"NEBIUS_API_KEY|sk-[A-Za-z0-9]{10,}", html)


def test_operator_reason_is_plain_language():
    """The reason for handing over to a person is written in the operator's words in the product report: the rule planner's original wording (with sealed_wood), merged cells, and internal field names in the model are all rewritten."""
    from cleanproof.report.html import operator_reason as f
    a = f("B3 residue survived a 15 N wipe, the limit for sealed_wood; likely permanent; C2 still not clean after 3 passes")
    assert "sealed_wood" not in a and "B3" in a and "15 N" in a and "C2" in a and "3 wipes" in a
    assert "surface_max_n" not in f("C3 still dirty at surface_max_n=15N")
    assert f("something else") == "something else"


# the report and the activity log name the models by their actual source
def _mini_run(tmp_path, policy, verifier, prov, final=True, calls=0, status="needs_human"):
    import json as _j
    rd = tmp_path / f"run-{policy}-{verifier}"
    (rd / "images").mkdir(parents=True)
    zr = {"status": "residue", "confidence": 0.9, "likely_cause": "stuck", "evidence": "x"}
    ev = [{"kind": "task_start", "ts": "2026-10-30T01:00:00+00:00", "task": {"instruction": "t", "zones": ["B3"]}, "targets": ["B3"], "policy": policy, "verifier": verifier},
          {"kind": "inspection", "step": 0, "image": "images/step00_survey.png", "zones": ["B3"], "results": {"B3": zr}, "provenance": {"B3": prov},
           "verifier": verifier, "llm": {"calls": [{"model": "m"}] * calls}},
          {"kind": "decision", "step": 1, "policy": policy, "decision": {"action": "wipe_zone", "zone_ids": ["B3"], "force_n": 5, "reason": "r"}, "guard": "ok"}]
    if final:
        ev.append({"kind": "inspection", "step": 2, "image": "images/step02_final.png", "zones": ["B3"], "results": {"B3": zr}, "provenance": {"B3": prov},
                   "verifier": verifier, "llm": {"calls": []}})
    ev.append({"kind": "task_end", "ts": "2026-10-30T01:05:00+00:00", "status": status, "reason": "", "steps": 2, "final_status": {"B3": "residue"}})
    (rd / "events.jsonl").write_text("".join(_j.dumps(e) + "\n" for e in ev))
    (rd / "summary.json").write_text(_j.dumps({"status": status, "policy": policy, "verifier": verifier, "targets": ["B3"], "task": {"instruction": "t", "zones": ["B3"]},
                                               "human_requests": [{"step": 3, "zones": ["B3"], "reason": "B3 still not clean after 3 passes"}] if status == "needs_human" else []}))
    return rd


def test_report_names_the_models_that_actually_ran(tmp_path):
    from cleanproof.report.html import render_product_report
    from cleanproof.report.rundata import load_run
    rd = _mini_run(tmp_path, "rules", "cv", "cv")
    h = render_product_report(load_run(rd))
    assert "Rule planner" in h and "Classical vision" in h
    assert "Nemotron" not in h and "Cosmos 3" not in h
    assert "vision check" in h
    rd = _mini_run(tmp_path, "rules", "vlm", "cv_fallback")
    h = render_product_report(load_run(rd))
    assert "Cosmos 3 not reachable, classical vision used" in h and "Cosmos 3 check" not in h
    rd = _mini_run(tmp_path, "nemotron", "vlm", "vlm", calls=1)
    h = render_product_report(load_run(rd))
    assert "Nemotron 3 Super" in h and "Cosmos 3" in h and "Clean at the first check and at the final check." in h


def test_report_final_check_only_when_there_was_one(tmp_path):
    from cleanproof.report.html import render_product_report
    from cleanproof.report.rundata import load_run
    h = render_product_report(load_run(_mini_run(tmp_path, "rules", "cv", "cv", final=False)))
    assert "final check" not in h.lower() or "No final check was made" in h
    assert "Clean at the first check and at the final check." not in h
