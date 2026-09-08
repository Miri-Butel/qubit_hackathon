"""Matplotlib views of routing instances, decoded solutions, and QAOA output.

Network views: `plot_network` (topology with latency/capacity edge labels),
`plot_routing` (each demand's chosen candidate path overlaid in color), and
`plot_link_utilization` (edges colored by load/capacity from decoded KPIs).

Presentation figures: `plot_solution_comparison` (two routings side by side),
`plot_energy_landscape` (cost vs sampled probability over the whole state
space), `plot_sampled_cost_distribution` (where the sampler puts its mass),
and `plot_fortz_thorup_curve` (the paper's link cost vs our quadratic fit).

Each single-panel function draws onto a provided Axes (or creates one) and
returns it, so plots compose into subplot grids.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .decode import RoutingSolution, bits_of, onehot_feasible, sample_probabilities
from .model import Link, NodeId, RoutingInstance
from .qubo import (
    CostCoefficients,
    build_cost_function,
    fortz_thorup_link_cost,
    fortz_thorup_quadratic_fit,
)

Pos = Mapping[NodeId, tuple[float, float]]

_NODE_STYLE = {"node_color": "#dddddd", "edgecolors": "#555555", "node_size": 700}
_BASE_EDGE_STYLE = {"edge_color": "#bbbbbb", "width": 1.5, "arrowsize": 15}
_DEMAND_ARC_STEP = 0.13  # curvature spacing keeps overlapping demand paths visible
_UTILIZATION_CMAP = "RdYlGn_r"
_FIGSIZE = (8, 4.5)

# Enumerating 2^N states is only for the toy instance; the real AT&T instance
# is 28 qubits (2.7e8 states), where the landscape plot is not the right tool.
MAX_ENUMERABLE_QUBITS = 16

_FEASIBLE_COLOR = "tab:blue"
_INFEASIBLE_COLOR = "#cccccc"
_OPTIMUM_COLOR = "tab:green"


def to_networkx(links: Iterable[Link]) -> nx.DiGraph:
    """Directed graph over the links, with capacity/latency edge attributes."""
    graph = nx.DiGraph()
    for link in links:
        graph.add_edge(link.u, link.v, capacity=link.capacity, latency=link.latency)
    return graph


def network_layout(instance: RoutingInstance) -> dict[NodeId, tuple[float, float]]:
    """Deterministic fallback node layout; pass an explicit `pos` to override."""
    layout = nx.kamada_kawai_layout(to_networkx(instance.links))
    return {node: (float(x), float(y)) for node, (x, y) in layout.items()}


def _prepare(
    instance: RoutingInstance, pos: Pos | None, ax: Axes | None
) -> tuple[nx.DiGraph, Pos, Axes]:
    """Resolve layout/axes and draw the nodes shared by every plot."""
    graph = to_networkx(instance.links)
    if pos is None:
        pos = network_layout(instance)
    if ax is None:
        _, ax = plt.subplots(figsize=_FIGSIZE)
    nx.draw_networkx_nodes(graph, pos, ax=ax, **_NODE_STYLE)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    ax.axis("off")
    return graph, pos, ax


def _draw_base_edges(
    graph: nx.DiGraph, instance: RoutingInstance, pos: Pos, ax: Axes
) -> None:
    """Gray topology edges with `latency/capacity` labels."""
    nx.draw_networkx_edges(
        graph, pos, node_size=_NODE_STYLE["node_size"], ax=ax, **_BASE_EDGE_STYLE
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={
            link.key: f"{link.latency:g}/{link.capacity:g}" for link in instance.links
        },
        font_size=8,
        ax=ax,
    )


def plot_network(
    instance: RoutingInstance,
    pos: Pos | None = None,
    ax: Axes | None = None,
    title: str = "edge labels: latency / capacity",
) -> Axes:
    """Draw the topology with `latency/capacity` labels on every link."""
    graph, pos, ax = _prepare(instance, pos, ax)
    _draw_base_edges(graph, instance, pos, ax)
    ax.set_title(title)
    return ax


def plot_routing(
    instance: RoutingInstance,
    chosen: Mapping[str, int | None],
    pos: Pos | None = None,
    ax: Axes | None = None,
    title: str = "selected routing",
) -> Axes:
    """Overlay each demand's chosen candidate path on the topology.

    `chosen` maps demand name -> selected path index (None = one-hot violated,
    drawn as unsatisfied), matching `RoutingSolution.chosen` and
    `BruteForceResult.best_assignment`.
    """
    graph, pos, ax = _prepare(instance, pos, ax)
    _draw_base_edges(graph, instance, pos, ax)
    ax.set_title(title)
    palette = mpl.colormaps["tab10"].colors
    handles = []
    for k, demand in enumerate(instance.demands):
        color = palette[k % len(palette)]
        route = f"{demand.name}: {demand.source}→{demand.target} (b={demand.bandwidth:g})"
        p = chosen.get(demand.name)
        if p is None:
            handles.append(Line2D([], [], color=color, lw=2.5, ls=":", label=f"{route}, unsatisfied"))
            continue
        rad = _DEMAND_ARC_STEP * (k - (len(instance.demands) - 1) / 2)
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=list(instance.paths_of(k)[p].links),
            edge_color=[color],
            width=2.5,
            arrowsize=18,
            node_size=_NODE_STYLE["node_size"],
            connectionstyle=f"arc3,rad={rad}",
            ax=ax,
        )
        handles.append(Line2D([], [], color=color, lw=2.5, label=f"{route}, path {p}"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=9)
    return ax


def plot_link_utilization(
    instance: RoutingInstance,
    solution: RoutingSolution,
    pos: Pos | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    vmax: float | None = None,
) -> Axes:
    """Color each link by utilization = load / capacity; over-capacity is dashed.

    Edge labels show `load/capacity`; the color scale saturates at
    max(1, max utilization) so 100% always maps to the same deep red. Pass
    `vmax` to force a shared scale across panels that must be comparable.
    """
    graph, pos, ax = _prepare(instance, pos, ax)
    cmap = mpl.colormaps[_UTILIZATION_CMAP]
    norm = Normalize(vmin=0.0, vmax=vmax or max(1.0, solution.max_utilization))
    for kpi in solution.link_kpis:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=[kpi.link],
            edge_color=[cmap(norm(kpi.utilization))],
            width=1.5 + 3.0 * min(kpi.utilization, 1.5),
            style="dashed" if kpi.over_capacity else "solid",
            arrowsize=15,
            node_size=_NODE_STYLE["node_size"],
            ax=ax,
        )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={
            kpi.link: f"{kpi.load:g}/{instance.link_by_key(kpi.link).capacity:g}"
            for kpi in solution.link_kpis
        },
        font_size=8,
        ax=ax,
    )
    ax.figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        label="utilization = load / capacity",
        fraction=0.046,
        pad=0.02,
    )
    if title is None:
        title = (
            f"link loads — max util {solution.max_utilization:.0%}, "
            f"{solution.capacity_violations} over capacity, "
            f"Φ* = {solution.phi_star:.2f}"
        )
    ax.set_title(title)
    return ax


# --- presentation figures ---


def plot_solution_comparison(
    instance: RoutingInstance,
    solutions: Mapping[str, RoutingSolution],
    pos: Pos | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Compare routings column by column: paths on top, link utilization below.

    The headline figure for "what does the congestion term buy us": pass
    {"latency only": sol_a, "congestion-aware": sol_b} to show one routing
    melting a link while the other stays inside capacity.
    """
    if not solutions:
        raise ValueError("solutions is empty")
    if pos is None:
        pos = network_layout(instance)  # shared layout keeps columns comparable
    # One color scale across all panels, or the same red would mean 200% in one
    # column and 100% in the next.
    vmax = max(1.0, *(s.max_utilization for s in solutions.values()))
    ncols = len(solutions)
    fig, axes = plt.subplots(
        2, ncols, figsize=figsize or (7.0 * ncols, 9.0), squeeze=False
    )
    for col, (label, solution) in enumerate(solutions.items()):
        plot_routing(instance, solution.chosen, pos=pos, ax=axes[0][col], title=label)
        plot_link_utilization(
            instance, solution, pos=pos, ax=axes[1][col], vmax=vmax
        )
    fig.tight_layout()
    return fig


def _enumerate_costs(
    instance: RoutingInstance, coeffs: CostCoefficients
) -> tuple[list[tuple[int, ...]], list[float], list[bool]]:
    """Every bitstring of the state space with its cost and one-hot feasibility."""
    if instance.num_qubits > MAX_ENUMERABLE_QUBITS:
        raise ValueError(
            f"{instance.num_qubits} qubits is too many to enumerate "
            f"(limit {MAX_ENUMERABLE_QUBITS}); use plot_sampled_cost_distribution instead"
        )
    cost_fn = build_cost_function(coeffs)
    states = [
        bits for bits in itertools.product((0, 1), repeat=instance.num_qubits)
    ]
    costs = [float(cost_fn(list(bits))) for bits in states]
    feasible = [onehot_feasible(bits, instance) for bits in states]
    return states, costs, feasible


def plot_energy_landscape(
    instance: RoutingInstance,
    coeffs: CostCoefficients,
    samples: pd.DataFrame | None = None,
    num_shots: int | None = None,
    reference_cost: float | None = None,
    ax: Axes | None = None,
    title: str = "energy landscape: where QAOA puts its probability",
) -> Axes:
    """Cost vs sampled probability for every state, feasible ones highlighted.

    Enumerates the whole 2^N space (toy instances only — see
    MAX_ENUMERABLE_QUBITS), so it shows not just the sampled states but the
    ones QAOA avoided. With `samples`, the y-axis is the sampled probability
    against the uniform-sampling baseline 1/2^N; without, states are drawn on
    the baseline so the plot still works with no cloud run.
    """
    states, costs, feasible = _enumerate_costs(instance, coeffs)
    uniform = 1.0 / 2**instance.num_qubits

    probability = dict.fromkeys(states, 0.0)
    if samples is not None:
        if num_shots is None:
            raise ValueError("num_shots is required when samples is given")
        for (_, row), prob in zip(
            samples.iterrows(), sample_probabilities(samples, num_shots)
        ):
            probability[tuple(bits_of(row))] = float(prob)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    for keep, color, label in (
        (False, _INFEASIBLE_COLOR, "infeasible (one-hot violated)"),
        (True, _FEASIBLE_COLOR, "feasible"),
    ):
        xs = [c for c, f in zip(costs, feasible) if f is keep]
        ys = [
            probability[s] if samples is not None else uniform
            for s, f in zip(states, feasible)
            if f is keep
        ]
        ax.scatter(xs, ys, s=18, c=color, label=label, alpha=0.75, zorder=2 + keep)
    ax.axhline(
        uniform,
        color="#888888",
        ls=":",
        zorder=1,
        label=f"uniform baseline = 1/2^{instance.num_qubits}",
    )
    if reference_cost is not None:
        ax.axvline(
            reference_cost,
            color=_OPTIMUM_COLOR,
            ls="--",
            zorder=1,
            label=f"brute-force optimum = {reference_cost:.3f}",
        )
    ax.set_xlabel("cost H(x)")
    ax.set_ylabel("sampled probability" if samples is not None else "uniform probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_sampled_cost_distribution(
    sample_sets: Mapping[str, pd.DataFrame],
    instance: RoutingInstance,
    coeffs: CostCoefficients,
    num_shots: int,
    reference_cost: float | None = None,
    bins: int = 40,
    ax: Axes | None = None,
    title: str = "sampled cost distribution",
) -> Axes:
    """Probability-weighted cost histogram per sample set, overlaid.

    Pass {"initial params": ..., "optimized": ...} to show the distribution
    shifting toward low cost as COBYLA converges. Unlike
    `plot_energy_landscape` this only looks at sampled states, so it scales to
    the full 28-qubit instance.
    """
    if not sample_sets:
        raise ValueError("sample_sets is empty")
    cost_fn = build_cost_function(coeffs)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for label, samples in sample_sets.items():
        costs, weights = [], []
        for (_, row), prob in zip(
            samples.iterrows(), sample_probabilities(samples, num_shots)
        ):
            costs.append(float(cost_fn(bits_of(row))))
            weights.append(float(prob))
        ax.hist(costs, bins=bins, weights=weights, alpha=0.55, label=label)
    if reference_cost is not None:
        ax.axvline(
            reference_cost,
            color=_OPTIMUM_COLOR,
            ls="--",
            label=f"brute-force optimum = {reference_cost:.3f}",
        )
    ax.set_xlabel("cost H(x)")
    ax.set_ylabel("sampled probability")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_fortz_thorup_curve(
    solution: RoutingSolution | None = None,
    instance: RoutingInstance | None = None,
    u_max: float = 1.2,
    ax: Axes | None = None,
    title: str = "Fortz–Thorup link cost vs the quadratic fit in our Hamiltonian",
) -> Axes:
    """The paper's piecewise link cost, our quadratic fit, and where links sit.

    Fortz & Thorup (INFOCOM 2000) Sec. II: Phi_e is piecewise linear with
    slopes 1, 3, 10, 70, 500, 5000 — the hockey stick that makes overload
    catastrophic. `fortz_thorup_quadratic_fit` is what the QUBO can actually
    express; plotting both shows how faithfully the Hamiltonian tracks it.
    With a `solution` (and its `instance`), each link is dropped onto the exact
    curve at its utilization.
    """
    if (solution is None) != (instance is None):
        raise ValueError("pass both `solution` and `instance`, or neither")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    steps = 400
    us = [u_max * i / steps for i in range(steps + 1)]
    exact = [fortz_thorup_link_cost(u, 1.0) for u in us]
    alpha, beta = fortz_thorup_quadratic_fit()
    fitted = [alpha * u + beta * u * u for u in us]

    ax.plot(us, exact, color="tab:red", lw=2, label="exact Φ(u)/c (piecewise, FT 2000)")
    ax.plot(
        us,
        fitted,
        color="tab:blue",
        lw=2,
        ls="--",
        label=f"quadratic fit in H: {alpha:.2g}·u + {beta:.3g}·u²",
    )
    for breakpoint in (1 / 3, 2 / 3, 9 / 10, 1.0, 11 / 10):
        if breakpoint <= u_max:
            ax.axvline(breakpoint, color="#dddddd", lw=1, zorder=0)
    ax.axvline(1.0, color="#999999", lw=1.2, ls=":", zorder=1, label="capacity (u = 1)")

    if solution is not None and instance is not None:
        for kpi in solution.link_kpis:
            if kpi.utilization > u_max:
                continue
            ax.scatter(
                kpi.utilization,
                fortz_thorup_link_cost(kpi.utilization, 1.0),
                s=45,
                c=_OPTIMUM_COLOR if not kpi.over_capacity else "tab:red",
                edgecolors="#333333",
                zorder=3,
            )
            ax.annotate(
                f"{kpi.link[0]}→{kpi.link[1]}",
                (kpi.utilization, fortz_thorup_link_cost(kpi.utilization, 1.0)),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
    ax.set_yscale("log")  # the 5000-slope tail flattens everything else on a linear axis
    ax.set_xlabel("link utilization u = load / capacity")
    ax.set_ylabel("link cost Φ(u) / c   (log scale)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return ax
