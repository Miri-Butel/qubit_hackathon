"""Rasterize the generated .pptx to PNGs for visual QA.

Reads the file python-pptx produced and redraws every shape at its real
position with matplotlib, so overlaps, collisions and overflowing copy are
visible. Text wrapping is approximated (matplotlib has no box layout), so this
checks placement, not glyph-exact line breaks.
"""
import io, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from pptx import Presentation
from pptx.util import Emu

SRC = Path("presentation/dist/qubit_pitch.pptx")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "presentation/dist/slides")
OUT.mkdir(parents=True, exist_ok=True)

def inch(v): return Emu(v).inches if v is not None else 0.0

prs = Presentation(str(SRC))
W, H = inch(prs.slide_width), inch(prs.slide_height)

for idx, slide in enumerate(prs.slides, 1):
    fig = plt.figure(figsize=(W, H), dpi=96)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="#060B18", zorder=0))
    for shape in slide.shapes:
        x, y, w, h = inch(shape.left), inch(shape.top), inch(shape.width), inch(shape.height)
        if shape.__class__.__name__ == "Picture":
            img = Image.open(io.BytesIO(shape.image.blob)).convert("RGBA")
            ax.imshow(img, extent=(x, x + w, y + h, y), zorder=2, interpolation="bilinear")
            continue
        if shape.has_text_frame and shape.text_frame.text.strip():
            run = next((r for p in shape.text_frame.paragraphs for r in p.runs), None)
            size = run.font.size.pt if run and run.font.size else 15
            colour = "#CBD6E2"
            try:
                if run and run.font.color and run.font.color.rgb:
                    colour = "#" + str(run.font.color.rgb)
            except Exception:
                pass
            align = {2: "center", 3: "right"}.get(
                shape.text_frame.paragraphs[0].alignment, "left")
            tx = x + (w / 2 if align == "center" else w if align == "right" else 0)
            ax.text(tx, y + size / 72 * 0.95, shape.text_frame.text,
                    fontsize=size * 0.99, color=colour, family="Arial",
                    fontweight="bold" if (run and run.font.bold) else "normal",
                    ha=align, va="top", zorder=4, linespacing=1.2)
            continue
        try:
            fill = "#" + str(shape.fill.fore_color.rgb) if shape.fill.type == 1 else "none"
        except Exception:
            fill = "none"
        try:
            edge = "#" + str(shape.line.color.rgb)
        except Exception:
            edge = "none"
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                               linewidth=0.8, zorder=1))
    fig.savefig(OUT / f"slide{idx:02d}.png", dpi=96, facecolor="#060B18")
    plt.close(fig)
print(f"rendered {len(prs.slides._sldIdLst)} slides to {OUT}")
