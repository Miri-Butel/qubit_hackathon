"""Render the figures embedded in docs/presentation-visuals.md.

Uses the toy instance from qaoa_routing.ipynb and the classical brute-force
solver, so everything here runs offline — no Classiq cloud, no QAOA. The two
QAOA-output figures are rendered from a Boltzmann-weighted stand-in sample
(clearly labelled as such) so the docs show the figure's shape without
pinning a cloud run. Re-run after changing the instance or the plots:

    python scripts/render_figures.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "network_instance"))

from routing_qaoa import (  # noqa: E402
    CandidatePath,
    Demand,
    Link,
    QuboWeights,
    RoutingInstance,
    brute_force_feasible,
    compute_coefficients,
    decode_bitstring,
    plot_energy_landscape,
    plot_fortz_thorup_curve,
    plot_link_utilization,
    plot_network,
    plot_routing,
    plot_sampled_cost_distribution,
    plot_solution_comparison,
    to_networkx,
)
from routing_qaoa.qubo import build_cost_function  # noqa: E402

import att_real as A  # noqa: E402
import export  # noqa: E402
import viz as netviz  # noqa: E402
import yen  # noqa: E402

# Mirrors the toy instance in qaoa_routing.ipynb.
LINKS = (
    Link("A", "B", capacity=10, latency=1),
    Link("B", "D", capacity=5, latency=1),
    Link("A", "C", capacity=10, latency=2),
    Link("C", "D", capacity=10, latency=2),
    Link("B", "C", capacity=5, latency=1),
    Link("D", "E", capacity=10, latency=1),
    Link("D", "F", capacity=5, latency=2),
    Link("E", "F", capacity=10, latency=1),
    Link("C", "E", capacity=10, latency=3),
)
DEMANDS = (
    Demand("d1", source="A", target="F", bandwidth=4, priority=3.0),
    Demand("d2", source="A", target="E", bandwidth=3),
    Demand("d3", source="B", target="F", bandwidth=3),
)
K_PATHS = 3
POS = {"A": (0, 0.5), "B": (1, 1), "C": (1, 0), "D": (2, 1), "E": (2, 0), "F": (3, 0.5)}

LATENCY_ONLY = QuboWeights(lambda_cong=0.0)
FT_FIT = QuboWeights(congestion_profile="fortz-thorup-fit", cost_scale=0.1)

FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
DPI = 110


def build_instance() -> RoutingInstance:
    graph = to_networkx(LINKS)

    def k_shortest(demand: Demand) -> tuple[CandidatePath, ...]:
        node_paths = nx.shortest_simple_paths(
            graph, demand.source, demand.target, weight="latency"
        )
        return tuple(
            CandidatePath(tuple(zip(nodes, nodes[1:])))
            for nodes in itertools.islice(node_paths, K_PATHS)
        )

    return RoutingInstance(
        links=LINKS,
        demands=DEMANDS,
        candidate_paths={d.name: k_shortest(d) for d in DEMANDS},
    )


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def stand_in_samples(
    instance: RoutingInstance, cost_fn, sharpness: float
) -> pd.DataFrame:
    """Boltzmann-weighted sample over the state space, standing in for QAOA output.

    sharpness = 0 is the uniform distribution (what un-optimized parameters
    look like); larger values concentrate mass on low-cost states the way a
    converged QAOA does.
    """
    states = list(itertools.product((0, 1), repeat=instance.num_qubits))
    costs = np.array([float(cost_fn(list(s))) for s in states])
    weights = np.exp(-sharpness * (costs - costs.min()) / np.ptp(costs))
    probs = weights / weights.sum()
    keep = np.flatnonzero(probs > 1e-5)
    return pd.DataFrame(
        {
            "x": ["".join(map(str, states[i])) for i in keep],
            "probability": probs[keep] / probs[keep].sum(),
        }
    )


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    instance = build_instance()

    # Every routing is scored under one coefficient set so costs compare.
    coeffs = compute_coefficients(instance, FT_FIT)
    cost_fn = build_cost_function(coeffs)
    reference = brute_force_feasible(instance, FT_FIT)
    ft_solution = decode_bitstring(list(reference.best_bits), instance, coeffs)
    latency_solution = decode_bitstring(
        list(brute_force_feasible(instance, LATENCY_ONLY).best_bits), instance, coeffs
    )

    print("rendering into docs/figures:")

    plot_network(instance, pos=POS)
    save(plt.gcf(), "topology.png")

    plot_routing(
        instance,
        ft_solution.chosen,
        pos=POS,
        title="selected routing (congestion-aware objective)",
    )
    save(plt.gcf(), "routing.png")

    plot_link_utilization(instance, ft_solution, pos=POS)
    save(plt.gcf(), "utilization.png")

    fig = plot_solution_comparison(
        instance,
        {
            f"latency only (λ_cong = 0) — {latency_solution.capacity_violations} links over capacity":
                latency_solution,
            f"Fortz–Thorup congestion-aware — {ft_solution.capacity_violations} violations":
                ft_solution,
        },
        pos=POS,
    )
    save(fig, "comparison.png")

    plot_fortz_thorup_curve(solution=ft_solution, instance=instance)
    save(plt.gcf(), "fortz_thorup.png")

    optimized = stand_in_samples(instance, cost_fn, sharpness=6.0)
    initial = stand_in_samples(instance, cost_fn, sharpness=0.0)

    plot_energy_landscape(
        instance,
        coeffs,
        samples=optimized,
        num_shots=2048,
        reference_cost=reference.best_cost,
        title="energy landscape (illustrative sample — see caption)",
    )
    save(plt.gcf(), "energy_landscape.png")

    plot_sampled_cost_distribution(
        {"initial params": initial, "optimized": optimized},
        instance,
        coeffs,
        num_shots=2048,
        reference_cost=reference.best_cost,
        title="sampled cost distribution (illustrative sample — see caption)",
    )
    save(plt.gcf(), "cost_distribution.png")

    print(f"\nlatency-only : {latency_solution.chosen} "
          f"max_util={latency_solution.max_utilization:.2f} "
          f"violations={latency_solution.capacity_violations} "
          f"Φ*={latency_solution.phi_star:.2f}")
    print(f"FT-fit       : {ft_solution.chosen} "
          f"max_util={ft_solution.max_utilization:.2f} "
          f"violations={ft_solution.capacity_violations} "
          f"Φ*={ft_solution.phi_star:.2f}")

    render_att_figures()


def render_att_figures() -> None:
    """Full map → east10 slice → Yen shortlist → solved east10_dense on the US map."""
    full = A.att_backbone()
    east = A.att_backbone(region="east10")
    dense = A.att_backbone(region="east10_dense")
    east_paths = yen.budgeted_candidate_set(east, k=5, base=2, extra_for=2)
    dense_paths = yen.budgeted_candidate_set(dense, k=5, base=2)
    n_qubits = sum(len(p) for p in dense_paths.values())
    heavy = netviz.heaviest_demands(east, 2)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    netviz.draw_map(
        full, ax=axes[0][0],
        title="AT&T North America MPLS backbone: 25 PoPs, 56 links",
    )
    netviz.draw_map_slice(
        full, A.REGIONS["east10"], ax=axes[0][1], annotate=True, font_size=6,
    )
    netviz.draw_map_routes(
        east, {did: east_paths[did] for did in heavy}, ax=axes[1][0],
        annotate=True, font_size=7,
        title=f"Yen shortlist on the slice: {', '.join(heavy)} (each route is a qubit)",
    )
    netviz.draw_map(
        dense, ax=axes[1][1], annotate=True, font_size=7,
        title=(
            f"Working instance: {dense.n_nodes} PoPs, {dense.n_links} links, "
            f"{len(dense.demands)} demands → {n_qubits} qubits"
        ),
    )
    fig.tight_layout()
    save(fig, "att_pipeline.png")

    weights = QuboWeights(congestion_profile="fortz-thorup-fit")
    instance, _ = export.to_routing_instance(dense, dense_paths)
    coeffs = compute_coefficients(instance, weights)
    ref = brute_force_feasible(instance, weights)
    solved = decode_bitstring(list(ref.best_bits), instance, coeffs)
    today = {
        d.id: [dense.current_routing[d.id][0]]
        for d in dense.demands
        if d.id in dense.current_routing
    }
    chosen = export.chosen_to_routing(solved.chosen, dense_paths, inst=dense)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    netviz.draw_map_solution(
        dense, today, ax=axes[0], annotate=True, font_size=7,
        title=None,
    )
    axes[0].set_title(
        f"Today's shortest-path routing\n{axes[0].get_title()}", fontsize=9,
    )
    netviz.draw_map_solution(
        dense, chosen, ax=axes[1], annotate=True, font_size=7,
        objective=solved.cost,
    )
    axes[1].set_title(
        f"Congestion-aware assignment\n{axes[1].get_title()}", fontsize=9,
    )
    fig.tight_layout()
    save(fig, "att_solution.png")

    print(
        f"ATT dense    : {solved.chosen} "
        f"max_util={solved.max_utilization:.2f} "
        f"violations={solved.capacity_violations} "
        f"Φ*={solved.phi_star:.2f} "
        f"H={solved.cost:.2f}"
    )


if __name__ == "__main__":
    main()
