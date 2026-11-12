"""Write the README diagram as SVG, in a light and a dark version.

    python demos/diagram.py docs/img

Writes gpushift-light.svg and gpushift-dark.svg. GitHub picks one per theme via
<picture>. Plain SVG with a system font stack (README images can't load fonts).
"""

from __future__ import annotations

import sys
from pathlib import Path

THEMES = {
    "light": {
        "ink": "#1f2328",
        "muted": "#59636e",
        "line": "#d1d9e0",
        "panel": "#f6f8fa",
        "bad": "#cf222e",
        "a": "#0969da",
        "b": "#8250df",
        "c": "#1a7f37",
        "d": "#9a6700",
        "you": "#bc4c00",
        "track": "#e6eaef",
    },
    "dark": {
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "line": "#3d444d",
        "panel": "#151b23",
        "bad": "#f85149",
        "a": "#4493f8",
        "b": "#ab7df8",
        "c": "#3fb950",
        "d": "#d29922",
        "you": "#f0883e",
        "track": "#262c36",
    },
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Consolas, 'Liberation Mono', monospace"
W, H = 900, 440


class Svg:
    def __init__(self, t: dict):
        self.t = t
        self.parts: list[str] = []

    def add(self, s: str) -> None:
        self.parts.append(s)

    def text(self, x, y, s, size=12, color="muted", weight=None, anchor=None, mono=False):
        attrs = f'x="{x}" y="{y}" font-size="{size}" fill="{self.t[color]}"'
        if weight:
            attrs += f' font-weight="{weight}"'
        if anchor:
            attrs += f' text-anchor="{anchor}"'
        if mono:
            attrs += f' font-family="{MONO}"'
        self.add(f"<text {attrs}>{s}</text>")

    def box(self, x, y, w, h, r=12, fill="panel", stroke="line", dash=False, width=1):
        fill_v = "none" if fill is None else self.t[fill]
        extra = ' stroke-dasharray="4 4"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill_v}" '
            f'stroke="{self.t[stroke]}" stroke-width="{width}"{extra}/>'
        )

    def arrow(self, d, color="muted", dash=False, head=True):
        extra = ' stroke-dasharray="5 5"' if dash else ""
        marker = ' marker-end="url(#arr)"' if head else ""
        self.add(f'<path d="{d}" fill="none" stroke="{self.t[color]}" stroke-width="1.5"{extra}{marker}/>')

    def job(self, x, y, w, name, color, right="", progress=None):
        self.box(x, y, w, 38, r=8, fill=None, stroke=color, width=1.5)
        self.add(f'<rect x="{x + 10}" y="{y + 11}" width="4" height="16" rx="2" fill="{self.t[color]}"/>')
        self.text(x + 22, y + (20 if progress is not None else 24), name, size=13, color="ink", weight="600")
        if right:
            self.text(x + w - 12, y + 24, right, anchor="end")
        if progress is not None:
            bw = w - 34
            self.add(f'<rect x="{x + 22}" y="{y + 27}" width="{bw}" height="4" rx="2" fill="{self.t["track"]}"/>')
            self.add(
                f'<rect x="{x + 22}" y="{y + 27}" width="{bw * progress:.0f}" height="4" rx="2" fill="{self.t[color]}"/>'
            )


def svg(t: dict) -> str:
    s = Svg(t)
    s.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="{FONT}" role="img" aria-label="Claude sessions submit GPU jobs to one gpushift queue, which '
        "starts each job when a GPU is free. A hook blocks direct launches. /gpushift:free makes jobs checkpoint and step "
        'aside; /gpushift:resume continues them.">'
    )
    s.add(
        f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker></defs>'
    )

    # Column headings
    for x, label in ((40, "CLAUDE SESSIONS"), (360, "ONE SHARED QUEUE"), (670, "GPUS")):
        s.text(x, 26, label, size=11, weight="600")

    # Sessions, each submitting to the queue
    for (name, color), y in zip((("vision", "a"), ("nlp", "b"), ("evals", "c")), (44, 116, 188), strict=True):
        s.box(40, y, 170, 52, r=10)
        s.add(f'<circle cx="60" cy="{y + 26}" r="6" fill="{t[color]}"/>')
        s.text(76, y + 23, name, size=14, color="ink", weight="600")
        s.text(76, y + 40, "Claude session")
        s.arrow(f"M214,{y + 26} C282,{y + 26} 296,150 352,150")
    s.text(222, 44, "gpushift submit --one", size=11, mono=True)
    s.text(222, 59, "-- python train.py", size=11, mono=True)

    # The queue
    s.box(360, 44, 240, 196)
    s.job(376, 60, 208, "lora", "c", "starts in ~12m")
    s.job(376, 108, 208, "llama-sft", "d", "needs both GPUs")
    s.box(376, 156, 208, 38, r=8, fill=None, dash=True)
    s.text(376, 214, "In order. Short jobs may use a GPU")
    s.text(376, 230, "that would otherwise sit idle.")

    # Queue -> GPUs
    for y in (94, 196):
        s.arrow(f"M604,142 C632,142 640,{y} 662,{y}")

    # GPUs
    for g, (y, name, color, pct) in enumerate(((44, "resnet", "a", 0.62), (146, "bert", "b", 0.38))):
        s.box(670, y, 190, 94)
        s.text(686, y + 22, f"GPU {g}", size=13, color="ink", weight="600")
        s.text(844, y + 22, "24 GB", anchor="end")
        s.job(686, y + 32, 158, name, color, progress=pct)
        s.text(686, y + 86, f"CUDA_VISIBLE_DEVICES={g}", size=10.5, mono=True)
    s.text(670, 262, "One job per GPU, so no out-of-memory", size=12)
    s.text(670, 278, "crashes from two runs sharing one.", size=12)

    # The hook: a direct launch that never reaches a GPU
    s.arrow("M125,244 C125,296 420,300 640,300", color="bad", dash=True, head=False)
    s.add(
        f'<g stroke="{t["bad"]}" stroke-width="2.5"><line x1="636" y1="292" x2="652" y2="308"/>'
        '<line x1="652" y1="292" x2="636" y2="308"/></g>'
    )
    s.add(
        f'<text x="240" y="320" font-size="12" fill="{t["bad"]}">Launching '
        f'<tspan font-family="{MONO}">python train.py</tspan> directly? '
        "The gpushift hook blocks it and tells the agent how to queue it.</text>"
    )

    # You: /gpushift:free and /gpushift:resume
    s.add(f'<line x1="40" y1="344" x2="{W - 40}" y2="344" stroke="{t["line"]}"/>')
    s.box(40, 372, 170, 52, r=10)
    s.add(f'<circle cx="60" cy="398" r="6" fill="{t["you"]}"/>')
    s.text(76, 395, "you", size=14, color="ink", weight="600")
    s.text(76, 412, "need the GPUs back")
    for y, cmd, desc in (
        (372, "/gpushift:free", "Jobs that can checkpoint or restart step back into the queue; the rest pause."),
        (400, "/gpushift:resume", "Jobs continue where they stopped: from a checkpoint, or right where they paused."),
    ):
        s.arrow(f"M214,{y - 4} L250,{y - 4}", head=True)
        s.text(258, y, cmd, size=13, color="you", weight="600", mono=True)
        s.text(396, y, desc, size=12.5, color="ink")

    s.add("</svg>")
    return "\n".join(s.parts) + "\n"


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        path = out / f"gpushift-{name}.svg"
        path.write_text(svg(theme), encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
