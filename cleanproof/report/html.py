"""Render RunData as a single-file HTML cleaning report (images embedded as base64, SVG inline, opens offline).

Visible page text is in English (the report appears in the demo video). Every number comes directly from files in the run directory;
missing fields show as "n/a", nothing is estimated.
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Optional

from ..agent.force_table import force_label
from .provenance import FALLBACK_NOTE, check_kind, check_kinds, check_name, planner_name
from .plots import STATUS_COLORS, force_curve_svg, zone_map_svg
from .rundata import RunData, Snapshot, load_run, load_trace

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e0;--card:#ffffff;--accent:#2a78d6;
--clean:#0ca30c;--residue:#d03b3b;--uncertain:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:32px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 8px}
.muted{color:var(--muted)}.small{font-size:12px}
.badge{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;border:1px solid var(--line);background:#fff;margin-right:6px}
.badge.sim{background:#eef4fc;border-color:#c6d9f3;color:#1c4f8a}
.badge.ok{background:#e9f7e9;border-color:#b9e3b9;color:#0a5a0a}.badge.bad{background:#fbeaea;border-color:#f0bcbc;color:#8a1f1f}
.badge.warn{background:#fff6dd;border-color:#f3dc92;color:#7a5600}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:10px 0}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:760px){.grid2{grid-template-columns:1fr}}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:12px}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px}.kv dt{color:var(--muted)}.kv dd{margin:0}
img.shot{width:100%;border:1px solid var(--line);border-radius:6px;background:#eee}
.dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px;vertical-align:-1px}
.step{border-left:3px solid var(--accent);padding-left:14px;margin:18px 0}
.step .hdr{font-weight:600}.reject{border-left-color:var(--residue)}
pre{background:#f6f6f3;border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;font-size:12px}
.audit table{margin:8px 0}.audit h2{font-size:15px;border:none;margin:14px 0 6px}
code{font-size:12px;background:#f2f2ef;padding:1px 4px;border-radius:3px}
svg.force,svg.zonemap{max-width:100%;height:auto}
footer{margin-top:40px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:10px}
"""


def _fmt(v: Any, nd: int = 3, suffix: str = "") -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{escape(str(v))}{suffix}"


def _usd(v: Any) -> str:
    return "n/a" if v is None else f"${float(v):.4f}"


def _img_b64(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _img_tag(r: RunData, rel: Optional[str], alt: str, embed: bool) -> str:
    p = r.image_path(rel)
    if p is None:
        return f'<p class="muted small">{escape(alt)}: image missing</p>'
    src = _img_b64(p) if embed else escape(rel or "")
    return f'<img class="shot" src="{src}" alt="{escape(alt)}" loading="lazy"/>'


def _status_badge(status: str) -> str:
    cls = {"done": "ok", "needs_human": "warn", "model_unavailable": "warn", "fault": "bad", "estopped": "bad", "rejected": "bad", "running": "warn"}.get(status, "")
    return f'<span class="badge {cls}">{escape(status or "unknown")}</span>'


def _zone_chip(status: Optional[str]) -> str:
    color = STATUS_COLORS.get(status, STATUS_COLORS[None])
    return f'<span class="dot" style="background:{color}"></span>{escape(status or "not inspected")}'


def _md_inline(s: str) -> str:
    s = escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def markdown_to_html(text: str) -> str:
    """Minimal Markdown (headings, tables, lists, paragraphs) → HTML for the audit text, without adding dependencies."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if line.startswith("#"):
            level = min(len(line) - len(line.lstrip("#")), 4)
            out.append(f"<h{level}>{_md_inline(line.lstrip('#').strip())}</h{level}>")
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_md_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
        elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
            items = []
            while i < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]))
                i += 1
            out.append("<ul>" + "".join(f"<li>{_md_inline(x)}</li>" for x in items) + "</ul>")
        else:
            para = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "-", "*")):
                para.append(lines[i].rstrip())
                i += 1
            out.append(f"<p>{_md_inline(' '.join(para))}</p>")
    return "\n".join(out)


def _decision_html(d: Optional[dict[str, Any]], guard: str) -> str:
    if not d:
        return '<span class="muted">no decision recorded</span>'
    bits = [f"<strong>{escape(str(d.get('action', '')))}</strong>"]
    if d.get("zone_ids"):
        bits.append("zones " + ", ".join(escape(z) for z in d["zone_ids"]))
    if force_label(d):
        bits.append(f"force {escape(force_label(d))}")
    for k in ("pattern", "passes"):
        if d.get(k) is not None:
            bits.append(f"{k.replace('_', ' ')} {escape(str(d[k]))}")
    if d.get("bbox_norm"):
        bits.append("bbox " + ", ".join(f"{float(v):.2f}" for v in d["bbox_norm"]))
    head = " · ".join(bits)
    guard_cls = "ok" if guard.startswith("ok") and ":" not in guard else ("warn" if guard.startswith("ok_after") else "bad" if guard.startswith("fallback") else "")
    guard_html = f' <span class="badge {guard_cls}">guardrail: {escape(guard)}</span>' if guard and guard != "not_applied" else ""
    reason = f'<div><span class="muted">Reason:</span> {escape(str(d.get("reason", "")))}</div>' if d.get("reason") else ""
    exp = f'<div><span class="muted">Expected:</span> {escape(str(d.get("expected_outcome", "")))}</div>' if d.get("expected_outcome") else ""
    return f"<div>{head}{guard_html}</div>{reason}{exp}"


def _calls_rows(calls: list[dict[str, Any]]) -> str:
    rows = []
    for c in calls:
        rows.append("<tr>" + "".join([
            f"<td>{_fmt(c.get('step'))}</td>", f"<td>{escape(str(c.get('role', '')))}</td>",
            f"<td>{_fmt(c.get('backend'))}</td>", f"<td>{_fmt(c.get('model'))}</td>",
            f"<td><code>{escape(str(c.get('request_id') or 'n/a'))}</code></td>",
            f"<td class='num'>{_fmt(c.get('latency_s'), 3, ' s') if c.get('latency_s') is not None else 'n/a'}</td>",
            f"<td class='num'>{_fmt(c.get('prompt_tokens'))} / {_fmt(c.get('completion_tokens'))}</td>",
            f"<td class='num'>{_usd(c.get('cost_usd'))}</td>",
            f"<td>{_fmt(c.get('schema_ok'))}</td>"]) + "</tr>")
    return "".join(rows)


def _force_table(force: dict[str, dict[str, Any]]) -> str:
    if not force:
        return ""
    head = "<tr><th>Zone</th><th class='num'>Target (N)</th><th class='num'>Mean (N)</th><th class='num'>Peak (N)</th>" \
           "<th class='num'>Within ±1 N</th><th class='num'>Contact losses</th><th class='num'>Coverage</th><th class='num'>Contact time</th></tr>"
    rows = []
    for z, f in force.items():
        rows.append(f"<tr><td>{escape(z)}</td><td class='num'>{_fmt(f.get('target_n'), 1)}</td><td class='num'>{_fmt(f.get('mean_normal_n'), 2)}</td>"
                    f"<td class='num'>{_fmt(f.get('peak_normal_n'), 2)}</td><td class='num'>{_fmt(f.get('within_1n_ratio'), 2)}</td>"
                    f"<td class='num'>{_fmt(f.get('contact_loss_count'))}</td><td class='num'>{_fmt(f.get('coverage_ratio'), 2)}</td>"
                    f"<td class='num'>{_fmt(f.get('contact_duration_s'), 1, ' s')}</td></tr>")
    return f"<table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def _inspection_html(r: RunData, insp: dict[str, Any], embed: bool) -> str:
    results = insp.get("results") or {}
    prov = insp.get("provenance") or {}
    cvf = insp.get("cv_fraction") or {}
    truth = insp.get("truth_clean") or {}
    has_truth = bool(truth)
    head = "<tr><th>Zone</th><th>Verdict</th><th class='num'>Conf.</th><th>Source</th><th>Likely cause</th><th>Evidence</th><th class='num'>CV residue frac.</th>"
    head += "<th>Sim ground truth</th></tr>" if has_truth else "</tr>"
    rows = []
    for z in insp.get("zones", list(results)):
        res = results.get(z, {})
        row = (f"<tr><td>{escape(z)}</td><td>{_zone_chip(res.get('status'))}</td><td class='num'>{_fmt(res.get('confidence'), 2)}</td>"
               f"<td>{escape(str(prov.get(z, '')))}</td><td>{escape(str(res.get('likely_cause', '')))}</td>"
               f"<td class='small'>{escape(str(res.get('evidence', '')))}</td><td class='num'>{_fmt(cvf.get(z), 4)}</td>")
        if has_truth:
            tv = truth.get(z)
            row += f"<td>{'clean' if tv else 'residue' if tv is not None else 'n/a'}</td>"
        rows.append(row + "</tr>")
    spills = insp.get("new_spills") or []
    spill = f"<p><span class='badge warn'>new spill detected</span> {', '.join(escape(z) for z in spills)}</p>" if spills else ""
    calls = (insp.get("llm") or {}).get("calls") or []
    call_txt = ""
    if calls:
        c = calls[0]
        call_txt = (f"<p class='small muted'>Verifier call: {escape(str(c.get('model', '')))} on {escape(str(c.get('backend', '')))}, "
                    f"latency {_fmt(c.get('latency_s'), 2, ' s')}, request <code>{escape(str(c.get('request_id', '')))}</code></p>")
    img = _img_tag(r, insp.get("image"), f"inspection image step {insp.get('step')}", embed)
    return (f"<h3>Inspection ({escape(insp.get('label', ''))}) — verifier <code>{escape(str(insp.get('verifier', '')))}</code></h3>"
            f"<div class='grid2'><div>{img}</div><div><table><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>{spill}{call_txt}</div></div>")


def _step_html(r: RunData, s: Any, embed: bool) -> str:
    parts = []
    cls = "step reject" if any(w.get("status") in ("fault", "aborted", "no_contact", "rejected") for w in s.skills) else "step"
    parts.append(f"<div class='{cls}'><div class='hdr'>Step {s.index}</div>")
    for n in s.notes:
        parts.append(f"<p class='muted'>{escape(n)}</p>")
    for g in s.guard_rejects:
        d = g.get("decision") or {}
        parts.append(f"<div class='card' style='border-color:#f0bcbc'><span class='badge bad'>guardrail rejected</span> "
                     f"<strong>{escape(str(d.get('action', '')))}</strong> {escape(', '.join(d.get('zone_ids', [])))} "
                     f"{escape(force_label(d))} — {escape(str(g.get('reason', '')))}</div>")
    if s.decision is not None:
        parts.append(f"<div class='card'><div class='muted small'>Decision by <code>{escape(s.policy)}</code></div>{_decision_html(s.decision, s.guard)}")
        if s.calls:
            dc = [c for c in s.calls if c.get("role") == "decision"]
            if dc:
                parts.append("<div class='small muted'>" + "; ".join(
                    f"{escape(str(c.get('model', '')))} on {escape(str(c.get('backend', '')))}, {_fmt(c.get('latency_s'), 2, ' s')}, {_usd(c.get('cost_usd'))}"
                    for c in dc) + "</div>")
        parts.append("</div>")
    for op in s.operator:
        kind = op.get("kind", "")
        cls_b = "ok" if kind == "operator_approved" else "bad"
        label = {"operator_approved": "operator approved", "operator_rejected": "operator rejected",
                 "stop_requested": "E-STOP / stop requested", "estop": "E-STOP"}.get(kind, kind)
        extra = f" — {escape(str(op.get('reason', '')))}" if op.get("reason") else ""
        parts.append(f"<p><span class='badge {cls_b}'>{escape(label)}</span>{extra}</p>")
    for sk in s.skills:
        if sk.get("skill") == "survey":
            continue
        st = sk.get("status", "")
        cls_b = "ok" if st == "ok" else "bad"
        parts.append(f"<div class='card'><div><span class='badge {cls_b}'>skill {escape(str(sk.get('skill', '')))}: {escape(st)}</span>"
                     f"<span class='muted small'> zones {escape(', '.join(sk.get('zone_ids', [])))} · duration {_fmt(sk.get('duration_s'), 1, ' s')}</span></div>")
        if sk.get("reason"):
            parts.append(f"<p>{escape(str(sk['reason']))}</p>")
        if sk.get("force"):
            parts.append(_force_table(sk["force"]))
        if sk.get("trace"):
            tr = load_trace(r.run_dir, sk["trace"])
            if tr:
                parts.append(force_curve_svg(tr, f"Force curve — step {s.index} {sk.get('skill')} ({', '.join(sk.get('zone_ids', []))})", sk.get("force")))
                parts.append(f"<p class='small muted'>{tr['n_samples']} samples, {tr['duration_s']} s of robot time; shaded = force-controlled wiping phase.</p>")
        safety = [e for e in sk.get("events", []) if e.get("event") in ("safety", "contact_lost", "ramp_timeout")]
        if safety:
            parts.append("<ul class='small'>" + "".join(
                f"<li>{escape(str(e.get('event')))}: {escape(str(e.get('reason') or e.get('count') or ''))}</li>" for e in safety) + "</ul>")
        parts.append("</div>")
    for insp in s.inspections:
        parts.append(_inspection_html(r, insp, embed))
    parts.append("</div>")
    return "".join(parts)


def render_report(r: RunData, embed_images: bool = True) -> str:
    summ = r.summary or {}
    task = r.task
    sim_badge = '<span class="badge sim">Simulation (MuJoCo)</span>' if r.is_sim else f'<span class="badge">robot: {escape(r.robot or "unknown")}</span>'
    init_snap, final_snap = r.initial_snapshot, r.final_snapshot
    passes = summ.get("passes") or {}
    scenario = (r.reference_meta or {}).get("scenario") or {}

    kv = [
        ("Run ID", f"<code>{escape(r.run_id)}</code>"),
        ("Outcome", _status_badge(r.status) + (f" {escape(str(summ.get('reason')))}" if summ.get("reason") else "")),
        ("Instruction", escape(str(task.get("instruction", "")))),
        ("Target zones", ", ".join(escape(z) for z in r.targets) or "n/a"),
        ("Forbidden zones", ", ".join(escape(z) for z in r.forbidden) or "none"),
        ("Decision policy", escape(str(summ.get("policy") or (r.task_start or {}).get("policy", "n/a")))),
        ("Verifier", escape(str(summ.get("verifier") or (r.task_start or {}).get("verifier", "n/a")))),
        ("Camera source", escape(str((r.task_start or {}).get("source", "n/a")))),
        ("Steps", _fmt(summ.get("steps"))),
        ("Robot time", _fmt(summ.get("robot_time_s"), 1, " s")),
        ("Wall time", _fmt(summ.get("wall_time_s"), 1, " s")),
        ("Decision cost (Token Factory)", _usd(summ.get("decision_cost_usd")) if summ else "n/a"),
        ("Total model cost (all logged calls)", _usd(r.total_cost_usd())),
    ]
    if scenario:
        kv.insert(2, ("Scenario", f"<code>{escape(str(scenario.get('name', '')))}</code>"))
    if r.judge:
        checks = ", ".join(f"{k}: {'pass' if v else 'FAIL'}" for k, v in (r.judge.get("checks") or {}).items())
        kv.append(("Sim ground-truth check", f"{'<span class=\"badge ok\">passed</span>' if r.judge.get('passed') else '<span class=\"badge bad\">failed</span>'} {escape(checks)}"))
    kv_html = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in kv)

    zone_rows = []
    for z in r.targets:
        iv = init_snap.zones.get(z) if init_snap else None
        fv = final_snap.zones.get(z) if final_snap else None
        zone_rows.append(f"<tr><td>{escape(z)}</td><td>{_zone_chip(iv.status if iv else None)}</td><td>{_zone_chip(fv.status if fv else None)}</td>"
                         f"<td class='num'>{_fmt(fv.confidence if fv else None, 2)}</td><td>{escape(fv.source if fv else '')}</td>"
                         f"<td class='num'>{_fmt(passes.get(z))}</td><td>{escape(str((fv.last_preset if fv else None) or '—'))}</td>"
                         + (f"<td>{'clean' if fv.truth_clean else 'residue' if fv.truth_clean is not None else 'n/a'}</td>" if r.is_sim and fv else ("<td>n/a</td>" if r.is_sim else "")) + "</tr>")
    zone_head = "<tr><th>Zone</th><th>Initial</th><th>Final</th><th class='num'>Final conf.</th><th>Source</th><th class='num'>Passes</th><th>Last preset</th>" + ("<th>Sim ground truth</th>" if r.is_sim else "") + "</tr>"

    maps = ""
    if init_snap and final_snap:
        maps = ("<div class='grid2'>"
                + zone_map_svg(r.grid, init_snap.zones, r.targets, r.forbidden, f"Initial survey (step {init_snap.step})")
                + zone_map_svg(r.grid, final_snap.zones, r.targets, r.forbidden, f"Final inspection (step {final_snap.step})", passes)
                + "</div>")

    before_after = ""
    if init_snap and final_snap:
        before_after = ("<div class='grid2'><div><h3>Before (survey)</h3>" + _img_tag(r, init_snap.image, "before", embed_images)
                        + "</div><div><h3>After (final inspection)</h3>" + _img_tag(r, final_snap.image, "after", embed_images) + "</div></div>")
        if r.image_path("images/reference.jpg"):
            before_after += "<details><summary class='muted small'>Clean reference image (captured before the task)</summary>" \
                            + _img_tag(r, "images/reference.jpg", "reference", embed_images) + "</details>"

    steps_html = "".join(_step_html(r, s, embed_images) for s in r.steps)

    human = summ.get("human_requests") or []
    human_html = ""
    if human:
        human_html = "<h2>Hand-offs to a human</h2><ul>" + "".join(
            f"<li>Step {h.get('step')}: zones {escape(', '.join(h.get('zones', [])))} — {escape(str(h.get('reason', '')))}</li>" for h in human) + "</ul>"

    audit_html = ""
    if r.audit_text:
        am = summ.get("audit") or {}
        meta = f"<p class='small muted'>Generated by <code>{escape(str(am.get('model', 'n/a')))}</code>, request <code>{escape(str(am.get('request_id', 'n/a')))}</code>, cost {_usd(am.get('cost_usd'))}.</p>" if am else ""
        audit_html = f"<h2>Audit summary (Nemotron)</h2>{meta}<div class='card audit'>{markdown_to_html(r.audit_text)}</div>"

    calls_html = ("<h2>Model calls</h2><table><thead><tr><th>Step</th><th>Role</th><th>Backend</th><th>Model</th><th>Request ID</th>"
                  "<th class='num'>Latency</th><th class='num'>Tokens in / out</th><th class='num'>Cost</th><th>Schema OK</th></tr></thead>"
                  f"<tbody>{_calls_rows(r.calls)}</tbody></table>"
                  "<p class='small muted'>Latency and cost as logged by the LLM client at call time; local Cosmos calls are not billed. "
                  "Cells marked n/a were not recorded.</p>") if r.calls else ""

    legend = " ".join(f'<span class="small">{_zone_chip(s)}</span>' for s in ("clean", "residue", "uncertain", None))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    robot_label = _robot_label(r.run_dir)
    sim_note = (f"<p class='muted small'>This run was executed in the MuJoCo simulation (virtual {escape(robot_label)}, synthetic stains). "
                "Camera images are rendered or synthesized; ground-truth columns exist only because the simulator knows the true stain map.</p>" if r.is_sim else "")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CleanProof — Cleaning Report {escape(r.run_id)}</title><style>{CSS}</style></head>
<body><main>
<header><div>{sim_badge}{_status_badge(r.status)}</div><h1>CleanProof — Cleaning Report</h1>
<p class="muted">CleanProof on a {escape(robot_label)}: a cleaning agent that checks its own work. Every number below is read from the run directory <code>{escape(str(r.run_dir))}</code>.</p>{sim_note}</header>
<section><h2>Task and outcome</h2><div class="card"><dl class="kv">{kv_html}</dl></div></section>
<section><h2>Zone map</h2><p class="small">{legend}</p>{maps}
<table><thead>{zone_head}</thead><tbody>{''.join(zone_rows)}</tbody></table></section>
<section><h2>Before / after</h2>{before_after}</section>
<section><h2>Step-by-step timeline</h2>{steps_html}</section>
{human_html}
{audit_html}
{calls_html}
<footer>Report generated {now} by <code>cleanproof.report</code>. Source files: events.jsonl, summary.json, scenario.json, traces/*.npz, images/*.jpg{', audit.md' if r.audit_text else ''}.</footer>
</main></body></html>"""


def _robot_label(run_dir: Path) -> str:
    """Robot platform name of this job (robot.platform in reference_meta.json → label in configs/platforms.yaml; default UR3e)."""
    import json
    try:
        meta = json.loads((Path(run_dir) / "reference_meta.json").read_text(encoding="utf-8"))
        key = str((meta.get("robot") or {}).get("platform") or "ur3e")
    except Exception:
        key = "ur3e"
    if key == "ur3e":
        return "UR3e"
    try:
        from ..robot.platforms import load_platforms
        return str(load_platforms()[key].label)
    except Exception:
        return key


def build_report(run_dir: str | Path, out: Optional[str | Path] = None, embed_images: bool = True) -> Path:
    r = load_run(run_dir)
    html = render_report(r, embed_images=embed_images)
    out_path = Path(out) if out else r.run_dir / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------- Operator-facing product report (software Reports page)
def operator_reason(r: str) -> str:
    """Hand-off reason → words an operator understands (same rules as friendlyReason in the software's operator.html; the technical record part keeps the original text)."""
    import re
    parts = re.split(r";\s(?=[A-H]\d\b)", r or "")
    if len(parts) > 1:
        return " ".join(operator_reason(x) for x in parts)
    m = re.search(r"\b([A-H]\d)\b residue survived a (\d+(?:\.\d+)?) N wipe, the limit", r) or \
        re.search(r"\b([A-H]\d)\b.*?(?:limit|max(?:imum)? force)\D{0,3}(\d+(?:\.\d+)?) ?N", r)
    if m:
        return f"A mark in {m.group(1)} is still visible after wiping at this surface's maximum force ({m.group(2)} N)."
    m = re.search(r"\b([A-H]\d) still not clean after (\d+) passes", r)
    if m:
        return f"A mark in {m.group(1)} is still visible after {m.group(2)} wipes."
    m = re.search(r"\b([A-H]\d): robot rejected every planned wipe", r)
    if m:
        return f"The robot could not run a safe wipe in {m.group(1)}."
    if re.search(r"\b[a-z]+_[a-z_]+\s*=", r):                  # internal field names in the model's own words (surface_max_n=15N etc.)
        z = re.search(r"\b([A-H]\d)\b", r)
        n = re.search(r"(\d+(?:\.\d+)?) ?N\b", r)
        return (f"A mark{' in ' + z.group(1) if z else ''} is still visible after wiping"
                f"{f' at this surface' + chr(39) + f's maximum force ({n.group(1)} N)' if n else ''}.")
    return r


PRODUCT_CSS = """
@font-face{font-family:Manrope;font-weight:400;src:url(/fonts/manrope/Manrope-Regular.ttf)}
@font-face{font-family:Manrope;font-weight:600;src:url(/fonts/manrope/Manrope-SemiBold.ttf)}
:root{--ink:#1d2433;--muted:#6b7280;--line:#e4e1da;--sub:#f4f2ed;--clean:#0f766e;--clean-bg:#e3f2ef;--slate:#475569;--slate-bg:#eef0f3;
--cos:#1d4ed8;--nemo:#4f3fc8;--amber:#c26a12}
*{box-sizing:border-box}body{margin:0;background:#fff;color:var(--ink);font:14px/1.45 Manrope,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
main{max-width:1040px;margin:0 auto;padding:26px 26px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.muted{color:var(--muted)}.small{font-size:12px}
.badge{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;border:1px solid var(--line);color:var(--muted)}
.badge.ok{background:var(--clean-bg);border-color:#b7dcd6;color:var(--clean)}.badge.staff{background:var(--slate-bg);border-color:#d5dae1;color:var(--slate)}
.sum{display:grid;grid-template-columns:repeat(5,1fr);gap:13px;margin-top:18px}
.sum div{background:var(--sub);border-radius:10px;padding:10px 13px}.sum b{display:block;font-size:20px}.sum span{font-size:12px;color:var(--muted);font-weight:600}
.zones{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}
.zc{border:1px solid var(--line);border-radius:10px;padding:10px}
.zc .t{display:flex;align-items:center;gap:8px;font-weight:600;margin-bottom:8px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:6px}.pair img{width:100%;border-radius:6px;display:block}
.pair .l{font-size:11px;color:var(--muted);font-weight:600;margin-top:3px}
.chips{display:flex;flex-wrap:wrap;gap:6px}.chip{padding:2px 8px;border-radius:6px;background:var(--clean-bg);color:var(--clean);font-size:12px;font-weight:600}
.tl{border-left:2px solid var(--line);margin-left:6px;padding-left:16px}
.tl p{margin:6px 0}.tl b.c{color:var(--cos)}.tl b.n{color:var(--nemo)}.tl b.g{color:var(--amber)}.tl b.s{color:var(--slate)}
.req{border:1px solid #d5dae1;background:#fbfbfc;border-radius:10px;padding:12px 14px}
.audit{border:1px solid var(--line);border-radius:10px;padding:4px 16px;font-size:13px}.audit h2{font-size:14px;margin:12px 0 4px}
img.full{width:100%;border-radius:8px;border:1px solid var(--line)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:13px}
"""


def _zone_crop_uri(r: RunData, rel: Optional[str], zone: str) -> Optional[str]:
    """Crop of one zone from the inspection image (rectified top-down image, 0.5 mm / pixel), embedded as a JPEG data URI."""
    import io

    from PIL import Image
    p = r.image_path(rel)
    if p is None:
        return None
    g = r.grid
    try:
        im = Image.open(p).convert("RGB")
        ids = g["ids"]
        ri = next(i for i, row in enumerate(ids) if zone in row)
        ci = ids[ri].index(zone)
        ch, cw = im.height / int(g["rows"]), im.width / int(g["cols"])
        crop = im.crop((int(ci * cw), int(ri * ch), int((ci + 1) * cw), int((ri + 1) * ch)))
        crop.thumbnail((260, 260))
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def render_product_report(r: RunData, site_name: str = "", robot_label: Optional[str] = None,
                          resolutions: Optional[dict[str, Any]] = None, interrupted: bool = False) -> str:
    """Cleaning report opened in the software: result, before / after comparison per processed zone, hand-off requests, process, audit summary.
    All numbers come from the run directory; no run directory paths or developer fields appear."""
    summ = r.summary or {}
    robot_label = robot_label or _robot_label(r.run_dir)
    init_snap, final_snap = r.initial_snapshot, r.final_snapshot
    passes = summ.get("passes") or {}
    human = summ.get("human_requests") or []
    staff_z = [z for h in human for z in h.get("zones", [])]
    status = r.status
    badge = ('<span class="badge warn">Interrupted</span>' if interrupted and status == "running" else
             '<span class="badge ok">All zones clean</span>' if status == "done" else
             f'<span class="badge staff">Staff check{" · " + escape(", ".join(staff_z)) if staff_z else ""}</span>' if status == "needs_human" else
             '<span class="badge warn">Model unavailable</span>' if status == "model_unavailable" else
             f'<span class="badge">{escape(status)}</span>')
    end = next((e for e in reversed(r.events) if e.get("kind") == "task_end"), {})
    ts = str(end.get("ts") or "")[:16].replace("T", " ")
    worked = [z for z in r.targets if passes.get(z) or z in staff_z or
              (init_snap and init_snap.zones.get(z) and init_snap.zones[z].status not in (None, "clean"))]
    untouched = [z for z in r.targets if z not in worked]
    wipes = sum(1 for e in r.events if e.get("kind") == "skill" and e.get("skill") in ("wipe_zone", "spot_scrub") and e.get("status") == "ok")
    checks = sum(1 for e in r.events if e.get("kind") == "inspection")
    ver = str(summ.get("verifier") or (r.task_start or {}).get("verifier") or "")
    insps = [i for st_ in r.steps for i in st_.inspections]
    kinds = check_kinds(insps, ver) or {"classical" if ver.startswith("cv") else "cosmos"}
    # the labels of the top statistics follow the actual check source (Cosmos 3 is named only if it took part throughout; a fallback to classical vision is stated)
    base = ("Cosmos 3 check" if kinds == {"cosmos"} else "vision check" if kinds <= {"classical", "fallback"} else "check")
    check_label = base + ("" if checks == 1 else "s") + (" (Cosmos 3 unavailable)" if "fallback" in kinds and kinds <= {"classical", "fallback"} else "")
    has_final = any(i.get("label") == "final" for i in insps)             # write "final check" only if there really was one (the all-zone check after the planner said finish)
    model_err: dict[int, Any] = {}                                         # the step where the planning model was unavailable (decision event meta.policy[*].model_error)
    for e in r.events:
        if e.get("kind") == "decision":
            me = next((m.get("model_error") for m in ((e.get("meta") or {}).get("policy") or []) if m.get("model_error")), None)
            if me:
                model_err[int(e.get("step", 0) or 0)] = me
    cards = []
    for z in worked:
        b = _zone_crop_uri(r, init_snap.image if init_snap else None, z)
        a = _zone_crop_uri(r, final_snap.image if final_snap else None, z)
        fv = final_snap.zones.get(z) if final_snap else None
        st = ('<span class="badge staff">Staff check</span>' if z in staff_z else
              '<span class="badge ok">Clean</span>' if fv and fv.status == "clean" else f'<span class="badge">{escape(fv.status if fv else "n/a")}</span>')
        cards.append(f"<div class='zc'><div class='t'>Zone {escape(z)} {st}<span class='muted small' style='margin-left:auto'>{passes.get(z, 0)} wipe{"" if passes.get(z, 0) == 1 else "s"}</span></div>"
                     f"<div class='pair'><div>{f'<img src=\"{b}\">' if b else ''}<div class='l'>Before</div></div>"
                     f"<div>{f'<img src=\"{a}\">' if a else ''}<div class='l'>After</div></div></div></div>")
    # Process (brief, in time order): inspect → plan → guardrail → wipe
    tl = []
    for s in r.steps:
        d = s.decision
        if d is not None and d.get("action") in ("wipe_zone", "spot_scrub"):
            tl.append(f"<p><b class='n'>{planner_name(s.policy)}</b> planned {escape(', '.join(d.get('zone_ids', [])))} at {escape(force_label(d))}"
                      + (" — <b class='g'>guardrail</b> sent the first plan back" if s.guard_rejects else "") + ".</p>")
        for sk in s.skills:
            if sk.get("skill") in ("wipe_zone", "spot_scrub") and sk.get("status") == "ok":
                f = sk.get("force") or {}
                tl.append("<p><b class='s'>Robot</b> wiped " + ", ".join(f"{escape(z)} at {float(v.get('mean_normal_n', 0)):.1f} N" for z, v in f.items()) + ".</p>")
        for insp in s.inspections:
            res = insp.get("results") or {}
            dirty = [z for z, v in res.items() if v.get("status") != "clean"]
            kd = check_kind(insp.get("provenance"), insp.get("verifier") or ver, len((insp.get("llm") or {}).get("calls") or []))
            tl.append(f"<p><b class='c'>{check_name(kd)}</b> checked {len(res)} zones: " + (f"{escape(', '.join(dirty))} still need{'s' if len(dirty) == 1 else ''} work." if dirty else "all clean.")
                      + (f" <span class='muted'>({FALLBACK_NOTE}.)</span>" if kd in ("fallback", "mixed") and "cv_fallback" in (insp.get("provenance") or {}).values() else "") + "</p>")
        if d is not None and d.get("action") == "ask_human":
            if model_err.get(s.index):
                tl.append(f"<p><b class='n'>{planner_name(s.policy)}</b> was unavailable: {escape(str(model_err[s.index].get('message', 'model unavailable')))}</p>")
            else:
                tl.append(f"<p><b class='n'>{planner_name(s.policy)}</b> asked staff to check {escape(', '.join(d.get('zone_ids', [])) or 'the job')}.</p>")
    req_html = ""
    if human:
        rs = resolutions or {}
        items = []
        for i, h in enumerate(human):
            res = rs.get(str(i))
            items.append(f"<div class='req'><b>{('Zone ' + escape(', '.join(h.get('zones', [])))) if h.get('zones') else 'Job-level'}</b> · {escape(operator_reason(str(h.get('reason', ''))))}"
                         + (f"<div class='small' style='margin-top:4px;color:var(--clean)'>Resolved by {escape(res.get('by', 'operator'))}: {escape(res.get('note', ''))}</div>"
                            if res else "<div class='small muted' style='margin-top:4px'>Open</div>") + "</div>")
        req_html = "<h2>Staff checks</h2>" + "".join(items)
    audit_html = f"<h2>Audit (Nemotron 3 Ultra)</h2><div class='audit'>{markdown_to_html(r.audit_text)}</div>" if r.audit_text else ""
    ba = ""
    if init_snap and final_snap:
        ba = ("<h2>Whole surface · start and finish</h2><div class='grid2'><div>" + _img_tag(r, init_snap.image, "before", True).replace('class="shot"', 'class="full"')
              + "<div class='small muted'>First check</div></div><div>" + _img_tag(r, final_snap.image, "after", True).replace('class="shot"', 'class="full"')
              + f"<div class='small muted'>{'Final check' if has_final else 'Last check'}</div></div></div>")
    untouched_note = ("Clean at the first check and at the final check." if has_final else
                      "Clean at the first check. " + ("No final check was made: the job ended early." if status != "done" else "No final check was recorded."))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Cleaning report</title><style>{PRODUCT_CSS}</style></head>
<body><main>
<div>{badge}</div><h1>Cleaning report</h1>
<div class="muted">{escape(site_name or "Site")} · {escape(robot_label)} · {"stopped before it finished (CleanProof was closed)" if interrupted and status == "running" else "finished " + escape(ts) + " UTC"} · task: {escape(str(r.task.get("instruction", "")))}</div>
<div class="sum"><div><b>{len(r.targets)}</b><span>zones checked</span></div><div><b>{len(worked)}</b><span>zones cleaned or flagged</span></div>
<div><b>{wipes}</b><span>wipe{"" if wipes == 1 else "s"}</span></div><div><b>{checks}</b><span>{check_label}</span></div><div><b>{_usd(summ.get("decision_cost_usd"))}</b><span>model cost</span></div></div>
<h2>Image proof · by zone</h2><div class="zones">{''.join(cards) or '<p class="muted">No zone needed work.</p>'}</div>
<h2>Every other zone</h2><div class="chips">{''.join(f'<span class="chip">{escape(z)} clean</span>' for z in untouched)}</div>
<p class="small muted">{untouched_note}</p>
{req_html}
{ba}
<h2>What happened</h2><div class="tl">{''.join(tl)}</div>
{audit_html}
</main></body></html>"""
