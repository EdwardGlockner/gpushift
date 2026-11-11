"""Write an 8-second GIF of one moment: an agent launches training directly,
the gpushift hook blocks it, and the agent queues it instead. The text matches
gpushift's real hook message and submit output.

    pip install pillow
    python demos/clip.py docs/img/gpushift-hook.gif
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 880, 300
BG, INK, MUTED, LINE = (255, 255, 255), (31, 35, 40), (89, 99, 110), (209, 217, 224)
GREEN, RED, BLUE = (26, 127, 55), (207, 34, 46), (9, 105, 218)

FONT_DIRS = [
    Path("C:/Windows/Fonts"),
    Path("/System/Library/Fonts"),
    Path("/Library/Fonts"),
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/TTF"),
]


def font(names: list[str], size: int) -> ImageFont.FreeTypeFont:
    for name in names:
        for d in FONT_DIRS:
            if (d / name).exists():
                return ImageFont.truetype(str(d / name), size)
    sys.exit(f"none of these fonts found: {names}")


F_MONO = font(["consola.ttf", "Menlo.ttc", "DejaVuSansMono.ttf"], 19)
F_MONO_B = font(["consolab.ttf", "Menlo.ttc", "DejaVuSansMono-Bold.ttf"], 19)
F_HEAD = font(["seguisb.ttf", "segoeuib.ttf", "HelveticaNeue.ttc", "DejaVuSans-Bold.ttf"], 15)

CMD1 = "python train_lora.py --rank 16"
CMD2 = "gpushift submit --one -- python train_lora.py --rank 16"
BLOCK = [
    ("Blocked by gpushift: `train_lora.py` looks like a training script.", RED),
    ("Submit it instead:", RED),
]
QUEUED = [
    ("job 7 queued (one: 1 GPU, 2 slots) tag=lora", GREEN),
    ("position 1 of 1, estimated start: ~12m (14:52)", GREEN),
]


def frame(lines: list[tuple[str, tuple, bool]]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([1, 1, W - 2, H - 2], 12, outline=LINE, width=2)
    d.ellipse([28, 27, 40, 39], fill=GREEN)
    d.text((50, 24), "evals", font=F_HEAD, fill=INK)
    d.text((50 + d.textlength("evals ", font=F_HEAD), 24), "· Claude Code session, Bash tool", font=F_HEAD, fill=MUTED)
    y = 70
    for text, color, prompt in lines:
        x = 28
        if prompt:
            d.text((x, y), "$ ", font=F_MONO_B, fill=BLUE)
            x += d.textlength("$ ", font=F_MONO_B)
        d.text((x, y), text, font=F_MONO_B if prompt else F_MONO, fill=color)
        y += 34 if prompt else 30
    return img


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "gpushift-hook.gif")
    frames: list[Image.Image] = []
    durations: list[int] = []
    shown: list[tuple[str, tuple, bool]] = []

    def add(img: Image.Image, ms: int) -> None:
        frames.append(img)
        durations.append(ms)

    def typed(cmd: str) -> None:
        for k in range(0, len(cmd) + 1, 2):
            add(frame([*shown, (cmd[:k], INK, True)]), 35)
        shown.append((cmd, INK, True))
        add(frame(shown), 350)

    add(frame([("", INK, True)]), 600)
    typed(CMD1)
    shown.extend((text, color, False) for text, color in BLOCK)
    add(frame(shown), 2000)
    typed(CMD2)
    shown.extend((text, color, False) for text, color in QUEUED)
    add(frame(shown), 2800)

    # One shared palette (taken from the busiest frame) so colors don't shimmer.
    pal = frames[-1].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(out, save_all=True, append_images=gif[1:], duration=durations, loop=0, disposal=1)
    print(f"wrote {out} ({sum(durations) / 1000:.1f}s, {out.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
