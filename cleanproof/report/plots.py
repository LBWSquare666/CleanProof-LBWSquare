"""Inline SVG for the report: force curve, zone status map. No external libraries or CDN, opens offline.

Colors: status colors are fixed (clean green / residue red / uncertain yellow / not checked grey) and always come with a text label, never color alone.
"""

from __future__ import annotations

from html import escape
from typing import Any, Optional

STATUS_COLORS = {"clean": "#0ca30c", "residue": "#d03b3b", "uncertain": "#fab219", None: "#b5b4ae", "": "#b5b4ae"}
STATUS_MARK = {"clean": "OK", "residue": "!!", "uncertain": "?", None: "–", "": "–"}
SERIES_FORCE = "#2a78d6"
SERIES_REF = "#52514e"
CONTROL_BAND = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRIDLINE = "#e6e5e0"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    raw = span / max(1, n)
    mag = 10 ** int(f"{raw:e}".split("e")[1])
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        if span / step <= n:
            break
    start = (int(lo / step)) * step
    out = []
    v = start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            out.append(round(v, 6))
        v += step
    return out


def force_curve_svg(trace: dict[str, Any], title: str = "", force: Optional[dict[str, dict[str, Any]]] = None,
                    width: int = 760, height: int = 240) -> str:
    """Normal-force curve of one wipe: measured f_n (solid blue), reference f_ref (dashed grey), control-phase shading."""
    t = trace.get("t") or []
    if not t:
        return f'<p class="muted">{escape(title)}: no trace samples</p>'
    t0 = t[0]
    ts = [v - t0 for v in t]
    fn = trace.get("f_n") or [0.0] * len(ts)
    fr = trace.get("f_ref") or [0.0] * len(ts)
    st = trace.get("state") or [0] * len(ts)
    ml, mr, mt, mb = 48, 16, 28, 36
    pw, ph = width - ml - mr, height - mt - mb
    x_lo, x_hi = 0.0, max(ts[-1], 1e-3)
    y_lo = min(0.0, min(fn))
    y_hi = max(max(fn), max(fr), 1.0) * 1.1

    def sx(v: float) -> float:
        return ml + (v - x_lo) / (x_hi - x_lo) * pw

    def sy(v: float) -> float:
        return mt + ph - (v - y_lo) / (y_hi - y_lo) * ph

    parts = [f'<svg class="force" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" aria-label="{escape(title)} normal force over time">']
    if title:
        parts.append(f'<text x="{ml}" y="16" fill="{TEXT}" font-size="13" font-weight="600">{escape(title)}</text>')
    # Control-phase shading (state 5)
    i = 0
    n = len(ts)
    while i < n:
        if int(st[i]) == 5:
            j = i
            while j + 1 < n and int(st[j + 1]) == 5:
                j += 1
            parts.append(f'<rect x="{sx(ts[i]):.1f}" y="{mt}" width="{max(1.0, sx(ts[j]) - sx(ts[i])):.1f}" '
                         f'height="{ph}" fill="{CONTROL_BAND}" opacity="0.08"/>')
            i = j + 1
        else:
            i += 1
    # Grid and ticks
    for yv in _ticks(y_lo, y_hi, 5):
        parts.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="{GRIDLINE}" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 6}" y="{sy(yv) + 4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end">{yv:g}</text>')
    for xv in _ticks(x_lo, x_hi, 6):
        parts.append(f'<text x="{sx(xv):.1f}" y="{mt + ph + 16}" fill="{MUTED}" font-size="11" text-anchor="middle">{xv:g}</text>')
    parts.append(f'<text x="{ml + pw / 2:.1f}" y="{height - 4}" fill="{MUTED}" font-size="11" text-anchor="middle">time in wipe (s)</text>')
    parts.append(f'<text transform="translate(12,{mt + ph / 2:.1f}) rotate(-90)" fill="{MUTED}" font-size="11" text-anchor="middle">normal force (N)</text>')
    # Curve
    ref_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(ts, fr))
    fn_pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(ts, fn))
    parts.append(f'<polyline points="{ref_pts}" fill="none" stroke="{SERIES_REF}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    parts.append(f'<polyline points="{fn_pts}" fill="none" stroke="{SERIES_FORCE}" stroke-width="2" stroke-linejoin="round"/>')
    # legend
    lx = ml + pw - 250
    parts.append(f'<line x1="{lx}" x2="{lx + 18}" y1="12" y2="12" stroke="{SERIES_FORCE}" stroke-width="2"/>'
                 f'<text x="{lx + 24}" y="16" fill="{TEXT}" font-size="11">measured f_n</text>'
                 f'<line x1="{lx + 110}" x2="{lx + 128}" y1="12" y2="12" stroke="{SERIES_REF}" stroke-width="1.5" stroke-dasharray="5 4"/>'
                 f'<text x="{lx + 134}" y="16" fill="{TEXT}" font-size="11">reference f_ref</text>'
                 f'<rect x="{lx + 218}" y="6" width="12" height="12" fill="{CONTROL_BAND}" opacity="0.15"/>'
                 f'<text x="{lx + 234}" y="16" fill="{TEXT}" font-size="11">control</text>')
    parts.append("</svg>")
    return "".join(parts)


def zone_map_svg(grid: dict[str, Any], zones: dict[str, Any], targets: list[str], forbidden: list[str],
                 title: str = "", passes: Optional[dict[str, int]] = None, cell: int = 92) -> str:
    """Zone status map. Rows (A, B...) go down the page, columns (1, 2...) go right. Each cell: status color + text label + confidence + source."""
    rows, cols, ids = int(grid["rows"]), int(grid["cols"]), grid["ids"]
    ml, mt = 26, 26 + (18 if title else 0)
    w, h = ml + cols * cell + 8, mt + rows * cell + 8
    parts = [f'<svg class="zonemap" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{escape(title or "zone map")}">']
    if title:
        parts.append(f'<text x="{ml}" y="16" fill="{TEXT}" font-size="13" font-weight="600">{escape(title)}</text>')
    for c in range(cols):
        parts.append(f'<text x="{ml + c * cell + cell / 2:.0f}" y="{mt - 8}" fill="{MUTED}" font-size="11" text-anchor="middle">{c + 1}</text>')
    for r in range(rows):
        parts.append(f'<text x="{ml - 10}" y="{mt + r * cell + cell / 2 + 4:.0f}" fill="{MUTED}" font-size="11" text-anchor="middle">{chr(ord("A") + r)}</text>')
        for c in range(cols):
            z = ids[r][c]
            x, y = ml + c * cell, mt + r * cell
            v = zones.get(z)
            v = vars(v) if v is not None and not isinstance(v, dict) else (v or {})
            if z in forbidden:
                fill, label, sub = "#ffffff", "forbidden", "no-go zone"
                extra = f'<line x1="{x + 4}" y1="{y + 4}" x2="{x + cell - 6}" y2="{y + cell - 6}" stroke="#8a8a86" stroke-width="1.5"/>' \
                        f'<line x1="{x + cell - 6}" y1="{y + 4}" x2="{x + 4}" y2="{y + cell - 6}" stroke="#8a8a86" stroke-width="1.5"/>'
            elif z not in targets:
                fill, label, sub, extra = "#f1f0ec", "", "not in task", ""
            else:
                status = v.get("status")
                fill = STATUS_COLORS.get(status, STATUS_COLORS[None])
                label = f"{STATUS_MARK.get(status, '–')} {status}" if status else "not inspected"
                conf = v.get("confidence")
                src = v.get("source") or ""
                sub = (f"conf {conf:.2f}" if isinstance(conf, (int, float)) else "") + (f" · {src}" if src else "")
                extra = ""
            opacity = 0.85 if z in targets and z not in forbidden else 1.0
            parts.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" rx="4" fill="{fill}" '
                         f'opacity="{opacity}" stroke="#ffffff" stroke-width="2"/>{extra}')
            dark_text = z in targets and z not in forbidden and v.get("status") in ("clean", "residue")
            color = "#ffffff" if dark_text else TEXT
            parts.append(f'<text x="{x + 8}" y="{y + 18}" fill="{color}" font-size="12" font-weight="700">{z}</text>')
            if label:
                parts.append(f'<text x="{x + 8}" y="{y + 38}" fill="{color}" font-size="11">{escape(label)}</text>')
            if sub:
                parts.append(f'<text x="{x + 8}" y="{y + 54}" fill="{color}" font-size="10">{escape(sub)}</text>')
            if passes and z in passes and z in targets:
                parts.append(f'<text x="{x + 8}" y="{y + cell - 10}" fill="{color}" font-size="10">wipes: {passes[z]}</text>')
    parts.append("</svg>")
    return "".join(parts)
