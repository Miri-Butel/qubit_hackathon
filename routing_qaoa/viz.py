"""Matplotlib views of routing instances and decoded solutions.

Three plots: `plot_network` (topology with latency/capacity edge labels),
`plot_routing` (each demand's chosen candidate path overlaid in color), and
`plot_link_utilization` (edges colored by load/capacity from decoded KPIs).
Each function draws onto a provided Axes (or creates one) and returns it,
so plots compose into subplot grids.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from .decode import RoutingSolution
from .model import Link, NodeId, RoutingInstance

Pos = Mapping[NodeId, tuple[float, float]]

_NODE_STYLE = {"node_color": "#dddddd", "edgecolors": "#555555", "node_size": 700}
_BASE_EDGE_STYLE = {"edge_color": "#bbbbbb", "width": 1.5, "arrowsize": 15}
_DEMAND_ARC_STEP = 0.13  # curvature spacing keeps overlapping demand paths visible
_UTILIZATION_CMAP = "RdYlGn_r"
_FIGSIZE = (8, 4.5)


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
) -> Axes:
    """Color each link by utilization = load / capacity; over-capacity is dashed.

    Edge labels show `load/capacity`; the color scale saturates at
    max(1, max utilization) so 100% always maps to the same deep red.
    """
    graph, pos, ax = _prepare(instance, pos, ax)
    cmap = mpl.colormaps[_UTILIZATION_CMAP]
    norm = Normalize(vmin=0.0, vmax=max(1.0, solution.max_utilization))
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
