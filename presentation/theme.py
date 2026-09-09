"""Brand tokens and matplotlib theming for the pitch deck.

The official QUBIT template fixes the palette (navy ground, white headlines,
cyan accent, amber for highlights only) and Arial throughout. Everything here
exists so the repo's existing figures can be re-rendered on that palette
instead of being redrawn: `routing_qaoa.viz` and `network_instance.viz` keep
their styling in module-level constants and accept an `ax`, so theming is a
matter of patching those constants and post-styling the returned axes.

Two hard constraints come from Google Slides, which is where the deck ends up:
fill and text transparency are dropped on import, so every colour here is
pre-mixed against the navy ground; and figures go in as PNGs, whose own alpha
survives untouched.
"""

from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- palette (from the official template's slide XML) ----------------------
BG = "#060B18"          # navy ground
SURFACE = "#0B1526"     # panel fill, one step up from the ground
HAIRLINE = "#22384C"    # rules and borders (cyan at ~20% over navy)
GHOST = "#25404E"       # oversized decorative numerals
BODY = "#CBD6E2"        # body copy
MUTED = "#96A8BA"       # captions, axis ticks, footers
WHITE = "#FFFFFF"       # headlines
CYAN = "#82E0F0"        # the accent
CYAN_DEEP = "#3FA8C4"   # second series
CYAN_PALE = "#C7EFF8"   # third series
AMBER = "#F5A623"       # highlights ONLY: congestion, "before", what changed
AMBER_TINT = "#2A2219"  # fill behind amber callouts
RED = "#E4572E"         # over capacity, used sparingly with amber

FONT = "Arial"

# Utilization ramp: cool where there is headroom, amber at the capacity cliff,
# so "danger" is the brand's own highlight colour rather than a stock red.
UTIL_CMAP = LinearSegmentedColormap.from_list(
    "qubit_util", ["#2C4A5A", CYAN_DEEP, CYAN, "#E8D07A", AMBER, RED]
)
try:
    mpl.colormaps.register(UTIL_CMAP, name="qubit_util")
except ValueError:
    pass  # already registered

DEMAND_COLORS = [CYAN, AMBER, CYAN_PALE, CYAN_DEEP, "#E8D07A", "#9AD1E0"]

RC = {
    "font.family": FONT,
    "font.size": 11,
    "text.color": BODY,
    "axes.labelcolor": BODY,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.facecolor": "none",
    "axes.titlecolor": WHITE,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "grid.color": MUTED,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10.5,
    "legend.frameon": False,
    "legend.labelcolor": BODY,
    "figure.facecolor": "none",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.transparent": True,
    "savefig.facecolor": "none",
    "axes.prop_cycle": mpl.cycler(color=[CYAN, AMBER, CYAN_PALE, CYAN_DEEP, WHITE]),
}


@contextmanager
def theme():
    """rcParams plus the two viz modules' style constants, brand-side."""
    import routing_qaoa.viz as rv

    saved_rv = {
        name: getattr(rv, name)
        for name in (
            "_NODE_STYLE", "_BASE_EDGE_STYLE", "_UTILIZATION_CMAP",
            "_FEASIBLE_COLOR", "_INFEASIBLE_COLOR", "_OPTIMUM_COLOR",
        )
    }
    with plt.rc_context(RC):
        rv._NODE_STYLE = {
            "node_color": SURFACE, "edgecolors": CYAN, "node_size": 700,
            "linewidths": 1.6,
        }
        rv._BASE_EDGE_STYLE = {"edge_color": HAIRLINE, "width": 1.4, "arrowsize": 13}
        rv._UTILIZATION_CMAP = "qubit_util"
        rv._FEASIBLE_COLOR = CYAN
        rv._INFEASIBLE_COLOR = "#2C3E50"
        rv._OPTIMUM_COLOR = AMBER
        try:
            yield
        finally:
            for name, value in saved_rv.items():
                setattr(rv, name, value)


def polish(ax, label_color=WHITE, label_size=10, edge_label_size=8.5):
    """Fix what the network plots leave at matplotlib defaults.

    `viz._prepare` calls `nx.draw_networkx_labels` with no font arguments, so
    node labels come out black at size 12 -- invisible on the navy ground --
    and networkx gives edge labels an opaque white bbox. Both are only
    reachable through `ax.texts` after the fact.
    """
    for text in ax.texts:
        bbox = text.get_bbox_patch()
        if bbox is not None:
            bbox.set_facecolor(BG)
            bbox.set_edgecolor("none")
            bbox.set_alpha(0.75)
            text.set_color(MUTED)
            text.set_fontsize(edge_label_size)
        else:
            text.set_color(label_color)
            text.set_fontsize(label_size)
        text.set_fontfamily(FONT)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor(SURFACE)
        legend.get_frame().set_edgecolor(HAIRLINE)
        for text in legend.get_texts():
            text.set_color(BODY)
            text.set_fontfamily(FONT)
    return ax


def style_colorbars(fig):
    """Colorbars are appended as extra axes; bring them onto the palette."""
    for cax in fig.axes:
        if cax.get_label() == "<colorbar>" or getattr(cax, "_colorbar", None):
            cax.tick_params(colors=MUTED, labelsize=9)
            cax.yaxis.label.set_color(BODY)
            cax.yaxis.label.set_fontsize(10)
            for spine in cax.spines.values():
                spine.set_edgecolor(HAIRLINE)


def save(fig, path, pad=0.08):
    """Transparent PNG so the slide's navy shows through with no seam."""
    style_colorbars(fig)
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=pad)
    plt.close(fig)
    return path
