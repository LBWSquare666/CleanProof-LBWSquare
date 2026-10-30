"""Original model / component emblems: procedurally drawn vector art (PIL + numpy); the same geometry outputs both a high-resolution PNG (gradient, glow) and an SVG.

Use: the model cards, step rings and architecture diagram in the demo video share one set of emblems (assets/brand_icons/), and the submission video can reuse them.
Constraint: there are no NVIDIA / Nebius logos, key visuals or commercial trade dress here,
the colors are this project's own palette (deep blue / teal, lavender, warm orange, gold, slate gray), and sponsors and model names appear only as text.

Emblems:
- cosmos            eye + scan grid ("sees & judges")
- nemotron_super    brain made of a node network ("reasons & decides")
- nemotron_ultra    same brain family + magnifying glass ("audits")
- token_factory     cloud + token stream (small badge for Nebius Token Factory, the name is still text)
- ur3e              robot arm silhouette (UR3e / skill server)
- guardrail         shield + check mark
- skill_server      gear + force arrow
- hawker_table      hawker center table + a bowl of noodles (scenario card of the Singapore page, fourteenth version)
- counter_desk      clinic / office counter + monitor
- handrail_gate     station stair handrail
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

EMBLEMS = ("cosmos", "nemotron_super", "nemotron_ultra", "token_factory", "ur3e", "guardrail", "skill_server",
           "hawker_table", "counter_desk", "handrail_gate")
TAGLINE = {"cosmos": "sees & judges", "nemotron_super": "reasons & decides", "nemotron_ultra": "audits", "token_factory": "Nebius Token Factory",
           "ur3e": "force control", "guardrail": "checks every plan", "skill_server": "force control",
           "hawker_table": "tables", "counter_desk": "counters", "handrail_gate": "handrails"}
# badge gradient (top → bottom): two colors per emblem; all from this project's palette
PALETTE = {
    "cosmos": ((0x1F, 0x4E, 0xB4), (0x3F, 0xC1, 0xE6)),
    "nemotron_super": ((0x4F, 0x3F, 0xC8), (0x9C, 0x6B, 0xFF)),
    "nemotron_ultra": ((0x33, 0x2A, 0x8A), (0x7D, 0x5C, 0xE0)),
    "token_factory": ((0x2E, 0x3F, 0x66), (0x56, 0x72, 0xA8)),
    "ur3e": ((0xD9, 0x6B, 0x2F), (0xF6, 0xB0, 0x7A)),
    "guardrail": ((0xC2, 0x8A, 0x12), (0xF2, 0xC9, 0x5C)),
    "skill_server": ((0x2F, 0x3A, 0x4C), (0x6B, 0x7A, 0x90)),
    "hawker_table": ((0xD9, 0x6B, 0x2F), (0xF6, 0xB0, 0x7A)),          # the three scenario cards of the Singapore page: warm orange / lavender / teal (this project's palette)
    "counter_desk": ((0x4F, 0x3F, 0xC8), (0x9C, 0x6B, 0xFF)),
    "handrail_gate": ((0x0F, 0x6E, 0x6A), (0x3F, 0xBF, 0xA8)),
}
WHITE = (255, 255, 255)


class _Canvas:
    """The same drawing calls land on both PIL (pixels) and SVG (text). Coordinates are in [0, 1] unit space."""

    def __init__(self, size: int, svg: bool = True):
        self.S = size
        self.img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.svg: list[str] = []
        self.want_svg = svg

    def _px(self, v: float) -> float:
        return v * self.S

    def _col(self, rgba: tuple) -> str:
        r, g, b = rgba[:3]
        a = rgba[3] / 255.0 if len(rgba) > 3 else 1.0
        return f'fill="rgb({r},{g},{b})" fill-opacity="{a:.3f}"'

    def _stroke(self, rgba: tuple, w: float, cap: str = "round") -> str:
        r, g, b = rgba[:3]
        a = rgba[3] / 255.0 if len(rgba) > 3 else 1.0
        return f'fill="none" stroke="rgb({r},{g},{b})" stroke-opacity="{a:.3f}" stroke-width="{w * 1000:.2f}" stroke-linecap="{cap}" stroke-linejoin="round"'

    def circle(self, cx: float, cy: float, r: float, fill: Optional[tuple] = None, outline: Optional[tuple] = None, width: float = 0.0) -> None:
        px = self._px
        self.d.ellipse([px(cx - r), px(cy - r), px(cx + r), px(cy + r)], fill=fill, outline=outline, width=int(round(px(width))) if outline else 0)
        if self.want_svg:
            if fill:
                self.svg.append(f'<circle cx="{cx * 1000:.2f}" cy="{cy * 1000:.2f}" r="{r * 1000:.2f}" {self._col(fill)}/>')
            if outline and width > 0:
                self.svg.append(f'<circle cx="{cx * 1000:.2f}" cy="{cy * 1000:.2f}" r="{r * 1000:.2f}" {self._stroke(outline, width)}/>')

    def polyline(self, pts: list[tuple[float, float]], color: tuple, width: float, close: bool = False) -> None:
        px = self._px
        p = [(px(x), px(y)) for x, y in pts] + ([(px(pts[0][0]), px(pts[0][1]))] if close else [])
        w = max(1, int(round(px(width))))
        self.d.line(p, fill=color, width=w, joint="curve")
        for x, y in p:                                   # round caps
            self.d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=color)
        if self.want_svg:
            s = " ".join(f"{x * 1000:.2f},{y * 1000:.2f}" for x, y in pts)
            tag = "polygon" if close else "polyline"
            self.svg.append(f'<{tag} points="{s}" {self._stroke(color, width)}/>')

    def polygon(self, pts: list[tuple[float, float]], fill: tuple) -> None:
        px = self._px
        self.d.polygon([(px(x), px(y)) for x, y in pts], fill=fill)
        if self.want_svg:
            s = " ".join(f"{x * 1000:.2f},{y * 1000:.2f}" for x, y in pts)
            self.svg.append(f'<polygon points="{s}" {self._col(fill)}/>')

    def rrect(self, x0: float, y0: float, x1: float, y1: float, r: float, fill: Optional[tuple] = None, outline: Optional[tuple] = None,
              width: float = 0.0) -> None:
        px = self._px
        self.d.rounded_rectangle([px(x0), px(y0), px(x1), px(y1)], radius=px(r), fill=fill, outline=outline, width=int(round(px(width))) if outline else 0)
        if self.want_svg:
            if fill:
                self.svg.append(f'<rect x="{x0 * 1000:.2f}" y="{y0 * 1000:.2f}" width="{(x1 - x0) * 1000:.2f}" height="{(y1 - y0) * 1000:.2f}" '
                                f'rx="{r * 1000:.2f}" {self._col(fill)}/>')
            if outline and width > 0:
                self.svg.append(f'<rect x="{x0 * 1000:.2f}" y="{y0 * 1000:.2f}" width="{(x1 - x0) * 1000:.2f}" height="{(y1 - y0) * 1000:.2f}" '
                                f'rx="{r * 1000:.2f}" {self._stroke(outline, width)}/>')

    def glow(self, radius: float, strength: float = 0.6, color: tuple = WHITE) -> None:
        """Apply a soft glow to the white shapes drawn so far (affects the PNG only; the SVG approximates it with feGaussianBlur)."""
        a = self.img.split()[3].filter(ImageFilter.GaussianBlur(self._px(radius)))
        g = Image.new("RGBA", self.img.size, color + (0,))
        g.putalpha(a.point(lambda v: int(v * strength)))
        base = Image.new("RGBA", self.img.size, (0, 0, 0, 0))
        base.alpha_composite(g)
        base.alpha_composite(self.img)
        self.img = base
        self.d = ImageDraw.Draw(self.img)


def _badge(size: int, c0: tuple, c1: tuple, radius: float = 0.22, inset: float = 0.04) -> Image.Image:
    """Rounded square badge: diagonal linear gradient + top highlight + outer soft glow."""
    S = size
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / max(S - 1, 1)
    t = np.clip(0.15 + 0.85 * (0.55 * yy + 0.45 * xx), 0, 1)[..., None]
    rgb = (np.array(c0, np.float32) * (1 - t) + np.array(c1, np.float32) * t)
    hl = np.clip(1.0 - (yy / 0.45), 0, 1)[..., None] ** 2 * 38           # top highlight
    rgb = np.clip(rgb + hl, 0, 255).astype(np.uint8)
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([S * inset, S * inset, S * (1 - inset), S * (1 - inset)], radius=S * radius, fill=255)
    img = Image.fromarray(np.dstack([rgb, np.asarray(mask)]), "RGBA")
    # outer soft glow (same color, slightly brighter)
    glow = Image.new("RGBA", (S, S), tuple(min(255, int(v * 0.55 + 90)) for v in c1) + (0,))
    glow.putalpha(mask.filter(ImageFilter.GaussianBlur(S * 0.03)).point(lambda v: int(v * 0.45)))
    out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    out.alpha_composite(glow)
    out.alpha_composite(img)
    # inner stroke
    ImageDraw.Draw(out).rounded_rectangle([S * inset, S * inset, S * (1 - inset), S * (1 - inset)], radius=S * radius, outline=(255, 255, 255, 70),
                                          width=max(1, int(S * 0.008)))
    return out


# ---------------------------------------------------------------- shapes of each emblem (unit coordinates, white shapes)
def _brain_nodes() -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Nodes and links on a two-lobe brain shape (hand-placed points, left-right symmetric)."""
    left = [(0.30, 0.36), (0.24, 0.48), (0.28, 0.61), (0.37, 0.69), (0.40, 0.46), (0.36, 0.56), (0.44, 0.33)]
    right = [(1 - x, y) for x, y in left]
    nodes = left + right
    e = [(0, 1), (1, 2), (2, 3), (0, 6), (6, 4), (4, 5), (5, 3), (1, 4), (2, 5), (4, 6)]
    n = len(left)
    edges = e + [(a + n, b + n) for a, b in e] + [(6, 6 + n), (3, 3 + n), (4, 4 + n), (5, 5 + n)]
    return nodes, edges


def draw_glyph(c: _Canvas, name: str) -> None:
    W = WHITE + (255,)
    W2 = WHITE + (150,)
    if name == "cosmos":
        # scan grid (lower right) + eye + scan line
        for k in range(5):
            u = 0.30 + 0.10 * k
            c.polyline([(u, 0.60), (u, 0.84)], WHITE + (70,), 0.008)
            c.polyline([(0.30, 0.60 + 0.06 * k), (0.70, 0.60 + 0.06 * k)], WHITE + (70,), 0.008)
        # almond shape: upper and lower sine arcs
        top = [(0.20 + 0.60 * k / 24, 0.47 - 0.19 * math.sin(math.pi * k / 24)) for k in range(25)]
        bot = [(0.80 - 0.60 * k / 24, 0.47 + 0.19 * math.sin(math.pi * k / 24)) for k in range(25)]
        c.polyline(top + bot, W, 0.04, close=True)
        c.circle(0.5, 0.47, 0.115, fill=W2)
        c.circle(0.5, 0.47, 0.07, fill=W)
        c.circle(0.53, 0.43, 0.022, fill=(0x1F, 0x4E, 0xB4, 255))
        c.polyline([(0.16, 0.47), (0.84, 0.47)], WHITE + (110,), 0.012)                  # scan line
    elif name in ("nemotron_super", "nemotron_ultra"):
        nodes, edges = _brain_nodes()
        if name == "nemotron_ultra":
            nodes = [(x * 0.86 + 0.02, y * 0.86 + 0.02) for x, y in nodes]
        for a, b in edges:
            c.polyline([nodes[a], nodes[b]], WHITE + (140,), 0.016)
        for k, (x, y) in enumerate(nodes):
            c.circle(x, y, 0.034 if k in (4, 11, 6, 13) else 0.026, fill=W)
        # small brainstem arc
        if name == "nemotron_super":
            c.polyline([(0.46, 0.70), (0.48, 0.80), (0.56, 0.84)], W2, 0.03)
        else:
            c.circle(0.69, 0.70, 0.12, fill=None, outline=W, width=0.034)
            c.polyline([(0.78, 0.79), (0.90, 0.91)], W, 0.05)
    elif name == "token_factory":
        # cloud
        for cx, cy, r in ((0.40, 0.56, 0.17), (0.55, 0.47, 0.20), (0.70, 0.58, 0.15)):
            c.circle(cx, cy, r, fill=W)
        c.rrect(0.23, 0.56, 0.85, 0.74, 0.08, fill=W)
        # token stream (enters from the left, fading out)
        for k, (x, a) in enumerate(((0.07, 240), (0.13, 170), (0.19, 100))):
            c.rrect(x, 0.33 + 0.01 * k, x + 0.05, 0.40 + 0.01 * k, 0.012, fill=WHITE + (a,))
            c.rrect(x + 0.01, 0.46 + 0.01 * k, x + 0.06, 0.53 + 0.01 * k, 0.012, fill=WHITE + (a,))
        # three tokens in the cloud
        for k in range(3):
            c.rrect(0.38 + 0.11 * k, 0.60, 0.46 + 0.11 * k, 0.68, 0.015, fill=(0x2E, 0x3F, 0x66, 255))
    elif name == "ur3e":
        c.rrect(0.16, 0.80, 0.56, 0.87, 0.02, fill=W)                                   # base
        j0, j1, j2 = (0.30, 0.80), (0.36, 0.44), (0.66, 0.30)
        c.polyline([j0, j1, j2], W, 0.085)
        c.polyline([j2, (0.76, 0.52)], W, 0.07)                                          # wrist → tool
        for j in (j0, j1, j2):
            c.circle(j[0], j[1], 0.075, fill=W)
            c.circle(j[0], j[1], 0.03, fill=(0xD9, 0x6B, 0x2F, 255))
        c.rrect(0.66, 0.54, 0.87, 0.62, 0.02, fill=W)                                   # sponge disc
    elif name == "guardrail":
        pts = [(0.5, 0.12), (0.80, 0.24), (0.78, 0.52), (0.62, 0.76), (0.5, 0.86), (0.38, 0.76), (0.22, 0.52), (0.20, 0.24)]
        c.polygon(pts, W)
        inner = [(0.5 + (x - 0.5) * 0.78, 0.49 + (y - 0.49) * 0.78) for x, y in pts]
        c.polygon(inner, (0xC2, 0x8A, 0x12, 255))
        c.polyline([(0.38, 0.50), (0.46, 0.60), (0.63, 0.40)], W, 0.06)
    elif name == "skill_server":
        gx, gy = 0.44, 0.50
        teeth = []
        for k in range(10):
            a0 = 2 * math.pi * k / 10
            for da, r in ((-0.16, 0.24), (-0.10, 0.30), (0.10, 0.30), (0.16, 0.24)):
                teeth.append((gx + r * math.cos(a0 + da), gy + r * math.sin(a0 + da)))
        c.polygon(teeth, W)
        c.circle(gx, gy, 0.13, fill=(0x2F, 0x3A, 0x4C, 255))
        c.circle(gx, gy, 0.055, fill=W)
        c.polyline([(0.78, 0.26), (0.78, 0.56)], W, 0.036)                                 # force arrow (pressing down), keeping a margin like the other emblems
        c.polygon([(0.71, 0.52), (0.85, 0.52), (0.78, 0.65)], W)
    elif name == "hawker_table":
        c.rrect(0.16, 0.54, 0.84, 0.61, 0.025, fill=W)                                   # table top
        c.polyline([(0.27, 0.61), (0.23, 0.84)], W, 0.045)                                 # table leg
        c.polyline([(0.73, 0.61), (0.77, 0.84)], W, 0.045)
        bowl = [(0.36 + 0.24 * k / 16, 0.50 + 0.10 * math.sin(math.pi * k / 16)) for k in range(17)]
        c.polygon([(0.33, 0.47), (0.63, 0.47)] + bowl[::-1], W)                            # a bowl of noodles
        c.polyline([(0.56, 0.44), (0.70, 0.24)], W, 0.022)                                 # chopsticks
        c.polyline([(0.60, 0.45), (0.76, 0.28)], W, 0.022)
        for x in (0.41, 0.49):                                                             # steam
            c.polyline([(x, 0.40), (x - 0.02, 0.34), (x + 0.01, 0.28), (x - 0.01, 0.22)], W2, 0.02)
    elif name == "counter_desk":
        c.rrect(0.12, 0.58, 0.88, 0.65, 0.02, fill=W)                                    # counter top
        c.rrect(0.18, 0.65, 0.82, 0.85, 0.03, fill=W2)                                   # cabinet body
        c.rrect(0.50, 0.26, 0.80, 0.48, 0.03, fill=None, outline=W, width=0.035)           # monitor
        c.polyline([(0.65, 0.48), (0.65, 0.58)], W, 0.035)
        c.polyline([(0.58, 0.58), (0.72, 0.58)], W, 0.03)
        c.circle(0.30, 0.52, 0.055, fill=W)                                                # service bell
        c.polyline([(0.22, 0.575), (0.38, 0.575)], W, 0.03)
    elif name == "handrail_gate":
        steps = [(0.14, 0.86), (0.14, 0.76), (0.32, 0.76), (0.32, 0.66), (0.50, 0.66), (0.50, 0.56), (0.68, 0.56), (0.68, 0.46),
                 (0.86, 0.46), (0.86, 0.86)]
        c.polygon(steps, W2)                                                              # steps
        c.polyline([(0.12, 0.56), (0.88, 0.22)], W, 0.05)                                  # handrail
        for x0 in (0.23, 0.50, 0.77):
            y_rail = 0.56 - (x0 - 0.12) * (0.34 / 0.76)
            y_step = 0.76 if x0 < 0.32 else (0.66 if x0 < 0.50 else (0.56 if x0 < 0.68 else 0.46))
            c.polyline([(x0, y_rail), (x0, y_step - 0.005)], W, 0.03)                      # post


def render_emblem(name: str, size: int = 1024, badge: bool = True, want_svg: bool = False) -> tuple[Image.Image, Optional[str]]:
    """Returns (RGBA image, SVG text or None)."""
    if name not in EMBLEMS:
        raise ValueError(f"unknown emblem {name!r}, choose from {EMBLEMS}")
    c0, c1 = PALETTE[name]
    c = _Canvas(size, svg=want_svg)
    draw_glyph(c, name)
    c.glow(0.02, 0.5)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    if badge:
        out.alpha_composite(_badge(size, c0, c1))
    out.alpha_composite(c.img)
    svg = None
    if want_svg:
        body = "\n".join(c.svg)
        grad = (f'<defs><linearGradient id="g" x1="0" y1="0" x2="0.45" y2="0.55"><stop offset="0" stop-color="rgb{c0}"/>'
                f'<stop offset="1" stop-color="rgb{c1}"/></linearGradient>'
                f'<filter id="glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="12" result="b"/>'
                f'<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
        bg = '<rect x="40" y="40" width="920" height="920" rx="220" fill="url(#g)"/><rect x="40" y="40" width="920" height="920" rx="220" fill="none" stroke="rgba(255,255,255,0.28)" stroke-width="8"/>' if badge else ""
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="{size}" height="{size}">\n'
               f'<!-- {name} emblem: original work for the CleanProof demo (Apache-2.0). No third-party trademarks. -->\n'
               f'{grad}\n{bg}\n<g filter="url(#glow)">\n{body}\n</g>\n</svg>\n')
    return out, svg


_CACHE: dict[tuple[str, int, bool], Image.Image] = {}


def emblem(name: str, px: int, badge: bool = True) -> Image.Image:
    """Get an emblem at the display size (downscaled from the 512 px master, cached)."""
    key = (name, int(px), badge)
    if key not in _CACHE:
        master = _CACHE.get((name, 512, badge))
        if master is None:
            master, _ = render_emblem(name, 512, badge)
            _CACHE[(name, 512, badge)] = master
        _CACHE[key] = master.resize((int(px), int(px)), Image.LANCZOS) if px != 512 else master
    return _CACHE[key]


def export_all(out_dir: Path, size: int = 1024) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in EMBLEMS:
        img, svg = render_emblem(name, size, True, want_svg=True)
        p = out_dir / f"{name}.png"
        img.save(p)
        written.append(p)
        if svg:
            q = out_dir / f"{name}.svg"
            q.write_text(svg, encoding="utf-8")
            written.append(q)
        g, _ = render_emblem(name, size, False)
        g.save(out_dir / f"{name}_glyph.png")
        written.append(out_dir / f"{name}_glyph.png")
    return written
