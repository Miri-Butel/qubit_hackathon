"""Brand-styled figures for the deck, rendered from the repo's own plotters.

Nothing here redraws a figure that already exists. `routing_qaoa.viz` and
`network_instance.viz` keep their styling in module-level constants and take an
`ax`, so each figure is: patch the constants, draw into a themed axes, then
post-style what the plotters leave at matplotlib defaults.

Every number comes from a cached JSON under docs/, so a rebuild needs no cloud
access and cannot silently pick up a stale notebook output.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "network_instance"))
sys.path.insert(0, str(ROOT / "presentation"))

import theme as T  # noqa: E402
from theme import save, theme  # noqa: E402

GEN = ROOT / "presentation" / "assets" / "gen"
DOCS = ROOT / "docs"

HERO_REGION = "east10_tuned"


def load_json(name):
    path = DOCS / name
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------
# instance construction (cheap, classical, deterministic)
# --------------------------------------------------------------------------
def real_instance(region=HERO_REGION):
    import att_real as A
    import export
    import yen

    from routing_qaoa import QuboWeights, brute_force_feasible, compute_coefficients
    from routing_qaoa import decode_bitstring

    inst = A.att_backbone(region=region)
    candidates = yen.budgeted_candidate_set(inst, k=5, base=2)
    instance, _ = export.to_routing_instance(inst, candidates)
    weights = QuboWeights(congestion_profile="fortz-thorup-perlink")
    coeffs = compute_coefficients(instance, weights)
    ref = brute_force_feasible(instance, weights)
    optimal = decode_bitstring(list(ref.best_bits), instance, coeffs)

    today_routing = {
        d.id: [inst.current_routing[d.id][0]]
        for d in inst.demands
        if d.id in inst.current_routing
    }
    best_routing = export.chosen_to_routing(optimal.chosen, candidates, inst=inst)
    today_bits = [0] * instance.num_qubits
    for k in range(len(instance.demands)):
        today_bits[instance.flat_index(k, 0)] = 1
    today = decode_bitstring(today_bits, instance, coeffs)
    return {
        "inst": inst, "instance": instance, "candidates": candidates,
        "coeffs": coeffs, "today_routing": today_routing,
        "best_routing": best_routing, "today": today, "optimal": optimal,
        "num_assignments": ref.num_combinations,
    }


def _decoder_utilization(solution):
    """Map-ready utilization taken from the decoder, keyed the way the map is.

    `Instance.link_utilization` aggregates both directions of a span onto one
    undirected key, while the QUBO and the decoder treat each direction as its
    own link with its own capacity. Left alone, the map paints a span red that
    the KPI line next to it reports as within capacity. The decoder is the
    authority for every number in the deck, so the map reads its per-link KPIs
    and shows each span at the worst of its two directions.
    """
    worst = {}
    for kpi in solution.link_kpis:
        u, v = kpi.link
        key = frozenset((u, v))
        worst[key] = max(worst.get(key, 0.0), kpi.utilization)
    return worst


def _with_decoder_utilization(inst, solution):
    """Bind the decoder's utilization onto this Instance for one draw call."""
    inst.link_utilization = lambda routing, _u=_decoder_utilization(solution): _u
    return inst


def _mark_hot_links(ax, inst, solution, netviz):
    """Call out the loaded spans above the demand overlays.

    The utilization colour sits below the coloured demand paths, so a span at
    105% can be completely hidden by a route drawn over it -- the panel then
    contradicts the "1 link over capacity" in its own title. This re-draws the
    worst spans on top and labels them, which is also the only way an audience
    sees the before/after difference at a glance.
    """
    pos = netviz._geo_pos(inst)
    worst = _decoder_utilization(solution)
    over = {k: u for k, u in worst.items() if u > 1.0}
    targets = over or {max(worst, key=worst.get): max(worst.values())}
    for key, util in targets.items():
        u, v = tuple(key)
        (x0, y0), (x1, y1) = pos[u], pos[v]
        hot = util > 1.0
        colour = T.RED if hot else T.CYAN
        ax.plot([x0, x1], [y0, y1], color=colour, lw=5.0,
                linestyle=(0, (4, 2)) if hot else "solid",
                solid_capstyle="round", zorder=6, alpha=0.95)
        # offset perpendicular to the span so the badge clears the PoP dots
        dx, dy = x1 - x0, y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5 or 1.0
        ox, oy = -dy / norm, dx / norm
        ax.annotate(
            f"{util:.0%}", ((x0 + x1) / 2 + ox * 0.75,
                            (y0 + y1) / 2 + oy * 0.75), ha="center",
            color=T.BG if hot else T.BG, fontsize=11, fontweight="bold",
            zorder=7,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=colour,
                      edgecolor="none"),
        )
    return targets


def _patch_map(netviz):
    """Bring network_instance.viz's light-theme map onto the brand palette."""
    original = {
        "_paint_states": netviz._paint_states,
        "_util_style": netviz._util_style,
        "_label_pops": netviz._label_pops,
        "DEMAND_PALETTE": netviz.DEMAND_PALETTE,
        "ROUTE_PALETTE": netviz.ROUTE_PALETTE,
    }

    def paint_states(ax):
        for ring in netviz._us_states():
            ax.fill([p[0] for p in ring], [p[1] for p in ring],
                    facecolor=T.SURFACE, edgecolor=T.HAIRLINE,
                    linewidth=0.5, zorder=0)

    def util_style(val):
        if val > 1.0:
            return T.RED, (0, (4, 2))
        if val > 0.8:
            return T.AMBER, "solid"
        if val > 0.5:
            return T.CYAN, "solid"
        if val > 0.0:
            return T.CYAN_DEEP, "solid"
        return "#22384C", "solid"

    def label_pops(ax, inst, pos, nodes, node_size, font_size, color=None,
                   label_color=None, annotate=False, alpha=1.0):
        return original["_label_pops"](
            ax, inst, pos, nodes, node_size, font_size,
            color=T.BG if color is None else color,
            label_color=T.WHITE if label_color is None else label_color,
            annotate=annotate, alpha=alpha,
        )

    netviz._paint_states = paint_states
    netviz._util_style = util_style
    netviz._label_pops = label_pops
    netviz.DEMAND_PALETTE = T.DEMAND_COLORS
    netviz.ROUTE_PALETTE = T.DEMAND_COLORS
    return original


def _restore(netviz, original):
    for name, value in original.items():
        setattr(netviz, name, value)


def _style_map_axes(ax):
    for text in ax.texts:
        if text.get_color() in ("#566573", "#7f8c8d"):
            text.set_color(T.MUTED)
        text.set_fontfamily(T.FONT)
    T.polish(ax, label_color=T.WHITE, label_size=7.5, edge_label_size=7)


# --------------------------------------------------------------------------
# F1 -- the money visual: today vs optimized on the real backbone
# --------------------------------------------------------------------------
def hero_map(data):
    import viz as netviz

    with theme():
        original = _patch_map(netviz)
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
            today, best = data["today"], data["optimal"]
            netviz.draw_map_solution(
                _with_decoder_utilization(data["inst"], today),
                data["today_routing"], ax=axes[0], font_size=7, pad=3.2,
                title=(f"TODAY  ·  peak {today.max_utilization:.0%}  ·  "
                       f"{today.capacity_violations} link over capacity  ·  "
                       f"Φ {today.phi_total:,.0f}"),
            )
            netviz.draw_map_solution(
                _with_decoder_utilization(data["inst"], best),
                data["best_routing"], ax=axes[1], font_size=7, pad=3.2,
                title=(f"OPTIMIZED  ·  peak {best.max_utilization:.0%}  ·  "
                       f"{best.capacity_violations} over capacity  ·  "
                       f"Φ {best.phi_total:,.0f}"),
            )
            _mark_hot_links(axes[0], data["inst"], today, netviz)
            _mark_hot_links(axes[1], data["inst"], best, netviz)
            axes[0].title.set_color(T.AMBER)
            axes[1].title.set_color(T.CYAN)
            for ax in axes:
                ax.title.set_fontsize(10.5)
                ax.title.set_fontweight("bold")
                _style_map_axes(ax)
            fig.tight_layout(pad=0.4)
            return save(fig, GEN / "hero_map.png")
        finally:
            _restore(netviz, original)


# --------------------------------------------------------------------------
# F2 -- pipeline: the real backbone, then the slice we solve
# --------------------------------------------------------------------------
def pipeline_map(data):
    import att_real as A
    import viz as netviz

    with theme():
        original = _patch_map(netviz)
        try:
            full = A.att_backbone()
            fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
            netviz.draw_map(
                full, ax=axes[0], font_size=5.5,
                title=(f"AT&T North America MPLS backbone  ·  "
                       f"{full.n_nodes} PoPs  ·  {full.n_links} links"),
            )
            netviz.draw_map_slice(
                full, A.REGIONS[HERO_REGION], ax=axes[1], annotate=True,
                font_size=7, pad=3.4,
                title=(f"The slice we solve  ·  10 PoPs  ·  "
                       f"{len(data['instance'].demands)} demands  →  "
                       f"{data['instance'].num_qubits} qubits"),
            )
            for ax in axes:
                ax.title.set_fontsize(10.5)
                ax.title.set_color(T.WHITE)
                _style_map_axes(ax)
            fig.tight_layout(pad=0.4)
            return save(fig, GEN / "pipeline_map.png")
        finally:
            _restore(netviz, original)


# --------------------------------------------------------------------------
# F3 -- the novelty figure: why one global quadratic cannot hold the cliff
# --------------------------------------------------------------------------
def ft_curve(data):
    from routing_qaoa.qubo import (
        fortz_thorup_link_cost,
        fortz_thorup_perlink_fit,
        fortz_thorup_quadratic_fit,
        reachable_load_bounds,
    )

    instance = data["instance"]
    alpha, beta = fortz_thorup_quadratic_fit()
    lo, hi = reachable_load_bounds(instance)

    with theme():
        fig, ax = plt.subplots(figsize=(6.1, 4.3))
        u = np.linspace(0, 1.25, 400)
        exact = [fortz_thorup_link_cost(float(x), 1.0) for x in u]
        ax.plot(u, exact, color=T.WHITE, lw=2.4, zorder=4,
                label="exact Φ(u)/c  (Fortz–Thorup 2000)")
        ax.plot(u, alpha * u + beta * u ** 2, color=T.AMBER, lw=2.0, ls="--",
                zorder=3, label=f"one global quadratic  ({beta:.1f}·u²)")

        # Each link's own local fit, over the band that link can actually reach.
        shown = 0
        for key, terms in zip(
            [l.key for l in instance.links], range(len(instance.links))
        ):
            capacity = instance.link_by_key(key).capacity
            u_lo = lo.get(key, 0.0) / capacity
            u_hi = hi.get(key, 0.0) / capacity
            if u_hi <= u_lo + 1e-9:
                continue
            b, c = fortz_thorup_perlink_fit(u_lo, u_hi)
            band = np.linspace(u_lo, min(u_hi, 1.25), 60)
            ax.plot(band, b * band + c * band ** 2, color=T.CYAN, lw=1.5,
                    alpha=0.55, zorder=2,
                    label="per-link local fits (ours)" if shown == 0 else None)
            shown += 1

        ax.axvline(1.0, color=T.MUTED, lw=1.0, ls=":", zorder=1)
        ax.annotate("capacity", (1.0, 0.9), xytext=(4, 0),
                    textcoords="offset points", color=T.MUTED, fontsize=9,
                    rotation=90, va="bottom")
        ax.set_yscale("log")
        ax.set_xlabel("link utilization  u = load / capacity")
        ax.set_ylabel("link cost  Φ(u) / c   (log scale)")
        ax.set_xlim(0, 1.25)
        ax.set_ylim(1e-2, 2e3)
        ax.legend(loc="upper left", fontsize=9.5)
        T.polish(ax)
        fig.tight_layout(pad=0.3)
        return save(fig, GEN / "ft_curve.png")


# --------------------------------------------------------------------------
# F4 -- encoding economy: candidate paths vs one qubit per demand-edge
# --------------------------------------------------------------------------
def encoding_bars(data):
    instance = data["instance"]
    n_links = len(instance.links)
    n_demands = len(instance.demands)
    ours = instance.num_qubits
    theirs = n_links * n_demands

    with theme():
        fig, ax = plt.subplots(figsize=(6.1, 4.3))
        labels = ["ours\none qubit per\ncandidate route",
                  "textbook\none qubit per\ndemand × edge"]
        values = [ours, theirs]
        bars = ax.bar(labels, values, color=[T.CYAN, T.AMBER], width=0.56,
                      zorder=3)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value}", (bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 6), textcoords="offset points",
                        ha="center", color=T.WHITE, fontsize=22,
                        fontweight="bold")
        ax.set_ylabel("qubits for the same instance")
        ax.set_ylim(0, theirs * 1.22)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(False)
        ax.annotate(
            f"{theirs / ours:.0f}× fewer qubits\n"
            f"{n_demands} demands · {n_links} directed links",
            (0.5, 0.62), xycoords="axes fraction", ha="center",
            color=T.BODY, fontsize=11,
        )
        T.polish(ax)
        fig.tight_layout(pad=0.3)
        return save(fig, GEN / "encoding_bars.png")


# --------------------------------------------------------------------------
# F5 -- convergence, from a cached QAOA run
# --------------------------------------------------------------------------
def convergence(run, name):
    values = run.get("objective_values") or []
    if not values:
        return None
    with theme():
        fig, ax = plt.subplots(figsize=(6.1, 3.5))
        ax.plot(range(1, len(values) + 1), values, color=T.CYAN, lw=2.0,
                zorder=3)
        best = run.get("brute_force", {}).get("best_cost")
        if best is not None:
            ax.axhline(best, color=T.WHITE, lw=1.4, ls="--", zorder=2,
                       label=f"classical optimum  H = {best:.4f}")
            ax.legend(loc="upper right", fontsize=9.5)
        ax.set_xlabel("cost-function evaluation")
        quantile = run.get("config", {}).get("quantile", 1.0)
        ax.set_ylabel(f"CVaR$_{{{quantile:g}}}$ of sampled H")
        T.polish(ax)
        fig.tight_layout(pad=0.3)
        return save(fig, GEN / f"convergence_{name}.png")


# --------------------------------------------------------------------------
# F6 -- toy instance before/after, reusing the repo's own comparison panels
# --------------------------------------------------------------------------
def toy_comparison():
    import networkx as nx

    from routing_qaoa import (
        CandidatePath, Demand, Link, QuboWeights, RoutingInstance,
        brute_force_feasible, compute_coefficients, decode_bitstring,
        plot_link_utilization, plot_routing,
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_qaoa_study import TOY_DEMANDS, TOY_LINKS  # noqa: E402

    graph = nx.DiGraph()
    for link in TOY_LINKS:
        graph.add_edge(link.u, link.v, latency=link.latency)
    paths = {}
    for demand in TOY_DEMANDS:
        walks = nx.shortest_simple_paths(graph, demand.source, demand.target,
                                         weight="latency")
        chosen = []
        for nodes in walks:
            chosen.append(CandidatePath(tuple(zip(nodes[:-1], nodes[1:]))))
            if len(chosen) == 3:
                break
        paths[demand.name] = tuple(chosen)
    instance = RoutingInstance(TOY_LINKS, TOY_DEMANDS, paths)
    pos = {"A": (0, 0.5), "B": (1, 1), "C": (1, 0), "D": (2, 1), "E": (2, 0),
           "F": (3, 0.5)}

    configs = {
        "latency only": QuboWeights(lambda_cong=0.0),
        "congestion-aware": QuboWeights(congestion_profile="fortz-thorup-fit",
                                        cost_scale=0.1),
    }
    scoring = compute_coefficients(instance, configs["congestion-aware"])
    solutions = {}
    for label, weights in configs.items():
        ref = brute_force_feasible(instance, weights)
        solutions[label] = decode_bitstring(list(ref.best_bits), instance, scoring)
    vmax = max(1.0, *(s.max_utilization for s in solutions.values()))

    with theme():
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.4))
        for col, (label, solution) in enumerate(solutions.items()):
            plot_routing(instance, solution.chosen, pos=pos, ax=axes[0][col],
                         title=label.upper())
            plot_link_utilization(instance, solution, pos=pos, ax=axes[1][col],
                                  vmax=vmax,
                                  title=(f"peak {solution.max_utilization:.0%}"
                                         f"  ·  {solution.capacity_violations}"
                                         f" over capacity  ·  Φ* "
                                         f"{solution.phi_star:.2f}"))
            axes[0][col].title.set_color(T.AMBER if col == 0 else T.CYAN)
            axes[0][col].title.set_fontweight("bold")
            for ax in (axes[0][col], axes[1][col]):
                ax.title.set_fontsize(10.5)
                T.polish(ax, label_size=10)
        fig.tight_layout(pad=0.5)
        return save(fig, GEN / "toy_comparison.png")


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    data = real_instance()
    made = [hero_map(data), pipeline_map(data), ft_curve(data),
            encoding_bars(data), toy_comparison()]
    for name in ("toy", HERO_REGION):
        run = load_json(f"qaoa_{name}.json")
        if run:
            made.append(convergence(run, name))
    for path in made:
        if path:
            print(f"  {Path(path).name}")


if __name__ == "__main__":
    main()
