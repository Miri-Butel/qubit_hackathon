"""Drawing and tabulating helpers for the routing instances.

Kept out of the notebook so the notebook cells stay short, and so the same
figures can be regenerated for the final write-up.
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from topologies import Instance


PRIORITY_COLOR = {"high": "#c0392b", "medium": "#e67e22", "low": "#7f8c8d"}


def _edge_width(capacity: float, max_capacity: float) -> float:
    return 1.0 + 4.0 * (capacity / max_capacity)


def draw_topology(inst: Instance, ax=None, annotate: bool = True,
                  node_size: int = 700, font_size: int = 8):
    """Physical view: what the network is, before any traffic is placed on it.

    Link thickness is proportional to capacity. The label under each link
    shows the three link attributes the brief asks us to model.
    """
    ax = ax or plt.gca()
    pos = inst.pos or nx.spring_layout(inst.graph, seed=7)
    max_cap = max(d["capacity"] for _, _, d in inst.graph.edges(data=True))

    for u, v, data in inst.graph.edges(data=True):
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color="#95a5a6", linewidth=_edge_width(data["capacity"], max_cap),
            zorder=1, solid_capstyle="round",
        )

    nx.draw_networkx_nodes(inst.graph, pos, ax=ax, node_color="#2c3e50", node_size=node_size)
    nx.draw_networkx_labels(inst.graph, pos, ax=ax, font_color="white",
                            font_size=font_size, font_weight="bold")

    if annotate:
        labels = {
            (u, v): f"{d['capacity']}G\n{d['delay']}ms  ${d['cost']}"
            for u, v, d in inst.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            inst.graph, pos, edge_labels=labels, ax=ax, font_size=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )

    ax.set_title(f"{inst.name}\n{inst.headline}", fontsize=9)
    ax.axis("off")
    return ax


def draw_congestion(inst: Instance, routing, ax=None, title: str | None = None,
                    node_size: int = 700, font_size: int = 8, annotate: bool = True):
    """Traffic view: link colour is utilization under `routing`.

    Green is comfortable, orange is tight, red-and-dashed is over capacity --
    a link that is physically impossible to run and therefore means packet
    loss in the real network.
    """
    ax = ax or plt.gca()
    pos = inst.pos or nx.spring_layout(inst.graph, seed=7)
    util = inst.link_utilization(routing)
    max_cap = max(d["capacity"] for _, _, d in inst.graph.edges(data=True))

    for u, v, data in inst.graph.edges(data=True):
        u_val = util[frozenset((u, v))]
        if u_val > 1.0:
            color, style = "#e74c3c", (0, (4, 2))
        elif u_val > 0.8:
            color, style = "#e67e22", "solid"
        elif u_val > 0.5:
            color, style = "#f1c40f", "solid"
        elif u_val > 0.0:
            color, style = "#27ae60", "solid"
        else:
            color, style = "#dfe4e6", "solid"
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=color, linewidth=_edge_width(data["capacity"], max_cap),
            linestyle=style, zorder=1, solid_capstyle="round",
        )

    nx.draw_networkx_nodes(inst.graph, pos, ax=ax, node_color="#2c3e50", node_size=node_size)
    nx.draw_networkx_labels(inst.graph, pos, ax=ax, font_color="white",
                            font_size=font_size, font_weight="bold")

    labels = {}
    for u, v in inst.graph.edges():
        val = util[frozenset((u, v))]
        if val > 0:
            labels[(u, v)] = f"{val:.0%}" + ("  OVER" if val > 1.0 else "")
    if annotate:
        nx.draw_networkx_edge_labels(
            inst.graph, pos, edge_labels=labels, ax=ax, font_size=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )

    over = sum(1 for v in util.values() if v > 1.0)
    ax.set_title(
        title or f"{inst.name}: today's routing\nmax link utilization {max(util.values()):.0%}, "
                 f"{over} link(s) over capacity",
        fontsize=9,
    )
    ax.axis("off")
    return ax


def draw_demand_overlay(inst: Instance, demand_id: str, ax=None):
    """Show one demand's currently-configured path(s) on the topology, so the
    1+1 protection (primary plus link-disjoint backup) is visible."""
    ax = ax or plt.gca()
    pos = inst.pos or nx.spring_layout(inst.graph, seed=7)
    d = next(x for x in inst.demands if x.id == demand_id)

    for u, v in inst.graph.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#dfe4e6", linewidth=1.5, zorder=1)

    styles = [("#2980b9", "solid", "primary"), ("#8e44ad", (0, (5, 2)), "backup")]
    for idx, nodes in enumerate(inst.current_routing.get(demand_id, [])):
        color, style, label = styles[min(idx, len(styles) - 1)]
        for u, v in zip(nodes[:-1], nodes[1:]):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=color, linewidth=3.0, linestyle=style, zorder=2,
                    label=label if (u, v) == (nodes[0], nodes[1]) else None)

    nx.draw_networkx_nodes(inst.graph, pos, ax=ax, node_color="#2c3e50", node_size=600)
    nx.draw_networkx_labels(inst.graph, pos, ax=ax, font_color="white", font_size=7, font_weight="bold")
    for n in (d.src, d.dst):
        ax.scatter(*pos[n], s=900, facecolors="none", edgecolors=PRIORITY_COLOR[d.priority], linewidths=2.5, zorder=3)

    protection = "1+1 protected" if d.needs_backup else "unprotected"
    ax.set_title(
        f"{d.id}: {d.src} to {d.dst}, {d.bandwidth}G, "
        f"<={d.latency_bound}ms, {d.priority} priority, {protection}",
        fontsize=8,
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.axis("off")
    return ax


def demand_table(inst: Instance) -> pd.DataFrame:
    rows = []
    for d in inst.demands:
        paths = inst.current_routing.get(d.id, [])
        rows.append({
            "demand": d.id,
            "from": d.src,
            "to": d.dst,
            "Gbps": d.bandwidth,
            "latency<=": f"{d.latency_bound}ms",
            "priority": d.priority,
            "protection": "1+1" if d.needs_backup else "none",
            "paths req.": d.paths_required,
            "today's delay": "/".join(f"{inst.path_delay(p):.0f}ms" for p in paths),
            "today's opex": "/".join(f"{inst.path_cost(p) * d.bandwidth:.0f}" for p in paths),
            "today's route": "  |  ".join("-".join(p) for p in paths),
        })
    return pd.DataFrame(rows)


def link_table(inst: Instance) -> pd.DataFrame:
    util = inst.link_utilization(inst.current_routing)
    load = inst.link_load(inst.current_routing)
    rows = []
    for u, v, data in inst.graph.edges(data=True):
        key = frozenset((u, v))
        rows.append({
            "link": f"{u}-{v}",
            "capacity (Gbps)": data["capacity"],
            "delay (ms)": data["delay"],
            "opex /Gbps": data["cost"],
            "load today": load[key],
            "utilization": f"{util[key]:.0%}",
            "status": "OVER CAPACITY" if util[key] > 1.0 else ("tight" if util[key] > 0.8 else "ok"),
        })
    return pd.DataFrame(rows).sort_values("utilization", ascending=False, key=lambda s: s.str.rstrip("%").astype(float))


_STATES_CACHE = None


def _us_states():
    """State outlines from a local GeoJSON. Drawn straight with matplotlib so
    the notebook needs no geopandas/cartopy install and works offline."""
    global _STATES_CACHE
    if _STATES_CACHE is None:
        import json
        from pathlib import Path

        path = Path(__file__).parent / "data" / "us-states.geojson"
        with open(path) as fh:
            data = json.load(fh)
        polys = []
        for feat in data["features"]:
            geom = feat["geometry"]
            chunks = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for chunk in chunks:
                polys.append(chunk[0])  # exterior ring
        _STATES_CACHE = polys
    return _STATES_CACHE


def draw_map(inst: Instance, routing=None, ax=None, title: str | None = None,
             node_size: int = 260, font_size: int = 6, pad: float = 2.5,
             annotate: bool = False):
    """The network on a real map of the United States.

    Node positions are the PoPs' true longitude and latitude from the
    Topology Zoo record, so this is genuine geography rather than a layout
    algorithm's guess. Pass `routing` to colour links by utilization.
    """
    ax = ax or plt.gca()
    pos = {n: (d["lon"], d["lat"]) for n, d in inst.graph.nodes(data=True)}

    for ring in _us_states():
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        ax.fill(xs, ys, facecolor="#f4f6f7", edgecolor="#d5dbdb", linewidth=0.6, zorder=0)

    max_cap = max(d["capacity"] for _, _, d in inst.graph.edges(data=True))
    util = inst.link_utilization(routing) if routing else None

    for u, v, data in inst.graph.edges(data=True):
        if util is None:
            color, style = "#7f8c8d", "solid"
        else:
            val = util[frozenset((u, v))]
            if val > 1.0:
                color, style = "#e74c3c", (0, (4, 2))
            elif val > 0.8:
                color, style = "#e67e22", "solid"
            elif val > 0.5:
                color, style = "#f1c40f", "solid"
            elif val > 0.0:
                color, style = "#27ae60", "solid"
            else:
                color, style = "#c8d0d2", "solid"
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color=color,
                linewidth=_edge_width(data["capacity"], max_cap), linestyle=style,
                zorder=2, solid_capstyle="round")

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    # Marker has to be wide enough for a 4-character PoP code at this font
    # size, otherwise the label spills over the edge of its own dot.
    node_size = max(node_size, (font_size ** 2) * 11)
    ax.scatter(xs, ys, s=node_size, c="#1b2631", zorder=3, edgecolors="white", linewidths=1.2)
    for n, (x, y) in pos.items():
        ax.annotate(n, (x, y), fontsize=font_size, color="white", ha="center", va="center",
                    zorder=4, fontweight="bold")
        city = inst.graph.nodes[n].get("city")
        if city and annotate:
            ax.annotate(city, (x, y - 0.9), fontsize=font_size - 0.5, color="#566573",
                        ha="center", va="top", zorder=4)

    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect(1.25)
    if title is None and util is not None:
        over = sum(1 for v in util.values() if v > 1.0)
        title = f"max link utilization {max(util.values()):.0%}, {over} link(s) over capacity"
    ax.set_title(title or inst.headline, fontsize=9)
    ax.axis("off")
    return ax


def routes_table(routes, demand=None) -> pd.DataFrame:
    """One row per candidate route, with the attribute each metric optimizes."""
    rows = []
    for r in routes:
        rows.append({
            "demand": r.demand_id,
            "route": r.label,
            "delay ms": round(r.delay, 1),
            "opex": round(r.cost, 1),
            "hops": r.hops,
            "spare Gbps": round(r.min_spare, 1),
            "legal?": "yes" if r.within_latency else "OVER LATENCY",
            "found by": ",".join(sorted(r.found_by)) if r.found_by else "",
        })
    return pd.DataFrame(rows)


def metric_table(inst: Instance, demand, k: int = 5) -> pd.DataFrame:
    """The top-k routes each metric picks, side by side.

    Reading across a row shows where the metrics disagree: the fastest route
    is frequently not the cheapest, and neither is the one with headroom.
    """
    import yen

    ranked = yen.ranked_by_metric(inst, demand, k=k)
    cols = {}
    for metric, routes in ranked.items():
        cells = []
        for r in routes:
            mark = "" if r.within_latency else "  (!)"
            cells.append(f"{r.label}  [{r.delay:.1f}ms, ${r.cost:.0f}, {r.hops}h, {r.min_spare:+.0f}G]{mark}")
        cols[f"best by {metric}"] = cells + [""] * (k - len(cells))
    return pd.DataFrame(cols, index=[f"#{i}" for i in range(1, k + 1)])


def draw_routes(inst: Instance, routes, ax=None, title: str | None = None):
    """Draw a demand's shortlisted candidate routes on the network."""
    ax = ax or plt.gca()
    pos = inst.pos or nx.spring_layout(inst.graph, seed=7)

    for u, v in inst.graph.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color="#dfe4e6", linewidth=1.5, zorder=1)

    palette = ["#2980b9", "#8e44ad", "#16a085", "#d35400", "#c0392b"]
    for idx, r in enumerate(routes):
        color = palette[idx % len(palette)]
        style = "solid" if idx == 0 else (0, (5, 2))
        for j, (u, v) in enumerate(r.edges):
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                    color=color, linewidth=3.0 - 0.4 * idx, linestyle=style, zorder=2,
                    label=f"{r.label} ({r.delay:.1f}ms)" if j == 0 else None)

    nx.draw_networkx_nodes(inst.graph, pos, ax=ax, node_color="#2c3e50", node_size=500)
    nx.draw_networkx_labels(inst.graph, pos, ax=ax, font_color="white", font_size=7, font_weight="bold")

    if routes:
        d = next((x for x in inst.demands if x.id == routes[0].demand_id), None)
        if d:
            for n in (d.src, d.dst):
                ax.scatter(*pos[n], s=800, facecolors="none",
                           edgecolors=PRIORITY_COLOR[d.priority], linewidths=2.5, zorder=3)
    ax.set_title(title or "", fontsize=8)
    ax.legend(fontsize=6, loc="lower left")
    ax.axis("off")
    return ax


def qubit_estimate(inst: Instance, k_paths: int = 2) -> dict:
    """Rough logical-qubit budget for a monolithic QUBO of this instance.

    path-selection vars + admission vars + one slack register per capacity
    constraint, the register sized by the bit-length of the capacity bound.
    This is an estimate for choosing an instance -- the synthesized circuit
    width is what actually counts.
    """
    n_path_vars = sum(min(k_paths, 3) for _ in inst.demands)
    n_admission = len(inst.demands)
    slack_bits = sum(
        int(data["capacity"]).bit_length() for _, _, data in inst.graph.edges(data=True)
    )
    return {
        "instance": inst.name,
        "nodes": inst.n_nodes,
        "links": inst.n_links,
        "demands": len(inst.demands),
        "path vars": n_path_vars,
        "admission vars": n_admission,
        "slack (all links)": slack_bits,
        "est. qubits (all links)": n_path_vars + n_admission + slack_bits,
    }
