"""Build the data behind docs/dashboard.html.

Solves the toy instance under every weight configuration and writes the
topology plus per-config routings and KPIs, so the dashboard is a static page
with no Python at view time. The payload is written to dashboard_data.json
and injected into the page's DATA placeholder, keeping dashboard.html a
single self-contained file. Re-run after changing the instance or weights:

    python scripts/build_dashboard_data.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from routing_qaoa import (  # noqa: E402
    CandidatePath,
    Demand,
    Link,
    QuboWeights,
    RoutingInstance,
    RoutingSolution,
    brute_force_feasible,
    compute_coefficients,
    decode_bitstring,
    to_networkx,
)

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

WEIGHT_CONFIGS: dict[str, QuboWeights] = {
    "latency only (λ_cong = 0)": QuboWeights(lambda_cong=0.0),
    "quadratic, λ_cong = 1": QuboWeights(),
    "quadratic, λ_cong = 5": QuboWeights(lambda_cong=5.0),
    "Fortz–Thorup fit": QuboWeights(
        congestion_profile="fortz-thorup-fit", cost_scale=0.1
    ),
}
# All configs are scored under one coefficient set so their costs compare.
SCORING_WEIGHTS = WEIGHT_CONFIGS["Fortz–Thorup fit"]

DOCS = Path(__file__).resolve().parent.parent / "docs"
OUTPUT = DOCS / "dashboard_data.json"
PAGE = DOCS / "dashboard.html"
# The page ships with its data inlined; this marks the assignment to rewrite.
DATA_PREFIX = "  const DATA = "


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


def path_nodes(instance: RoutingInstance, k: int, p: int) -> list[str]:
    links = instance.paths_of(k)[p].links
    return [links[0][0]] + [v for _, v in links]


def solution_payload(
    instance: RoutingInstance, label: str, solution: RoutingSolution
) -> dict[str, Any]:
    return {
        "label": label,
        "chosen": solution.chosen,
        "paths": {
            demand.name: path_nodes(instance, k, solution.chosen[demand.name])
            for k, demand in enumerate(instance.demands)
            if solution.chosen[demand.name] is not None
        },
        "links": {
            f"{kpi.link[0]}->{kpi.link[1]}": {
                "load": kpi.load,
                "utilization": kpi.utilization,
                "over_capacity": kpi.over_capacity,
            }
            for kpi in solution.link_kpis
        },
        "kpis": {
            "max_utilization": solution.max_utilization,
            "capacity_violations": solution.capacity_violations,
            "weighted_latency": solution.weighted_latency,
            "total_latency": solution.total_latency,
            "phi_star": solution.phi_star,
            "cost": solution.cost,
        },
    }


def inline_into_page(payload: dict[str, Any]) -> None:
    """Replace the page's `const DATA = ...;` line with the fresh payload."""
    lines = PAGE.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(DATA_PREFIX):
            blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            lines[i] = f"{DATA_PREFIX}{blob};\n"
            PAGE.write_text("".join(lines))
            return
    raise SystemExit(
        f"no line starting with {DATA_PREFIX!r} in {PAGE}; cannot inject data"
    )


def main() -> None:
    instance = build_instance()
    coeffs = compute_coefficients(instance, SCORING_WEIGHTS)

    payload: dict[str, Any] = {
        "nodes": [
            {"id": node, "x": x, "y": y} for node, (x, y) in POS.items()
        ],
        "links": [
            {
                "u": link.u,
                "v": link.v,
                "capacity": link.capacity,
                "latency": link.latency,
            }
            for link in LINKS
        ],
        "demands": [
            {
                "name": d.name,
                "source": d.source,
                "target": d.target,
                "bandwidth": d.bandwidth,
                "priority": d.priority,
            }
            for d in DEMANDS
        ],
        "solutions": [],
    }

    for label, weights in WEIGHT_CONFIGS.items():
        best_bits = brute_force_feasible(instance, weights).best_bits
        solution = decode_bitstring(list(best_bits), instance, coeffs)
        payload["solutions"].append(solution_payload(instance, label, solution))

    DOCS.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote {OUTPUT.relative_to(Path.cwd())}")
    if PAGE.exists():
        inline_into_page(payload)
        print(f"wrote {PAGE.relative_to(Path.cwd())}")
    for entry in payload["solutions"]:
        kpis = entry["kpis"]
        print(
            f"  {entry['label']:<28} max_util={kpis['max_utilization']:.2f} "
            f"violations={kpis['capacity_violations']} "
            f"Φ*={kpis['phi_star']:.2f}"
        )


if __name__ == "__main__":
    main()
