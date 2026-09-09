"""Calibrate the routing Hamiltonian on the real AT&T instances.

The dense east10 instance is oversubscribed: today's shortest-path routing runs
links at 356% of capacity. That makes weight calibration a correctness issue,
not a tuning nicety -- at the default `cost_scale` the latency term dominates
and the Hamiltonian's optimum keeps today's congested routes (README warns that
congestion is divided by the used-link count while latency normalizes to
`cost_scale`).

This script enumerates every one-hot assignment once, records the physical KPIs
of each, and then asks each candidate weight configuration which assignment it
would pick. That separates two questions the pitch must not conflate:

  * what is physically achievable on this instance (the KPI frontier), and
  * whether a given Hamiltonian actually finds it.

Writes docs/real_instance_study.json for the deck and the notebook.
"""

import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "network_instance"))

from routing_qaoa import QuboWeights, decode_bitstring  # noqa: E402
from routing_qaoa.qubo import build_cost_function, compute_coefficients  # noqa: E402

import att_real as A  # noqa: E402
import export  # noqa: E402
import yen  # noqa: E402

OUT = ROOT / "docs" / "real_instance_study.json"

# Latency is normalized to `cost_scale` while the congestion term is not, so
# shrinking cost_scale is what buys the congestion term authority.
CONFIGS: dict[str, QuboWeights] = {
    "latency only": QuboWeights(lambda_cong=0.0),
    "quadratic, cs=1.0": QuboWeights(),
    "quadratic, cs=0.1": QuboWeights(cost_scale=0.1),
    "quadratic, cs=0.01": QuboWeights(cost_scale=0.01),
    "quadratic, cs=0.1, lc=5": QuboWeights(cost_scale=0.1, lambda_cong=5.0),
    "FT fit, cs=1.0": QuboWeights(congestion_profile="fortz-thorup-fit"),
    "FT fit, cs=0.1": QuboWeights(
        congestion_profile="fortz-thorup-fit", cost_scale=0.1
    ),
    "FT fit, cs=0.01": QuboWeights(
        congestion_profile="fortz-thorup-fit", cost_scale=0.01
    ),
    "FT fit, cs=0.1, lc=5": QuboWeights(
        congestion_profile="fortz-thorup-fit", cost_scale=0.1, lambda_cong=5.0
    ),
}

# KPIs are physical, so any coefficient set decodes them; costs are reported
# per configuration separately.
SCORING = QuboWeights(congestion_profile="fortz-thorup-fit", cost_scale=0.1)


def kpis(solution) -> dict:
    return {
        "max_util": round(solution.max_utilization, 4),
        "violations": solution.capacity_violations,
        "phi_star": round(solution.phi_star, 4),
        "weighted_latency": round(solution.weighted_latency, 2),
        "total_latency": round(solution.total_latency, 2),
        "demands_satisfied": solution.demands_satisfied,
    }


def study(region: str) -> dict:
    inst = A.att_backbone(region=region)
    extra = 2 if region == "east10" else 0
    candidates = yen.budgeted_candidate_set(inst, k=5, base=2, extra_for=extra)
    instance, current_routing = export.to_routing_instance(inst, candidates)
    scoring_coeffs = compute_coefficients(instance, SCORING)

    ranges = [range(len(instance.paths_of(k))) for k in range(len(instance.demands))]
    assignments = list(itertools.product(*ranges))
    print(
        f"\n=== {region}: {inst.n_nodes} PoPs, {inst.n_links} links, "
        f"{len(instance.demands)} demands, {instance.num_qubits} qubits, "
        f"{len(assignments)} feasible assignments ==="
    )

    # One enumeration pass: bits + physical KPIs for every assignment.
    records = []
    for choice in assignments:
        bits = [0] * instance.num_qubits
        for k, p in enumerate(choice):
            bits[instance.flat_index(k, p)] = 1
        solution = decode_bitstring(bits, instance, scoring_coeffs)
        records.append((choice, bits, solution))

    # today's routing = candidate 0 for every demand (build_export puts the
    # current route first), so it is the all-zeros choice.
    today = next(r for r in records if all(p == 0 for p in r[0]))
    print(f"today's routing        : {kpis(today[2])}")

    # The physical frontier: fewest violations, then lowest Phi*.
    frontier = min(records, key=lambda r: (r[2].capacity_violations, r[2].phi_star))
    best_phi = min(records, key=lambda r: r[2].phi_star)
    print(f"frontier (min viol)    : {kpis(frontier[2])}")
    print(f"frontier (min Phi*)    : {kpis(best_phi[2])}")

    rows = []
    for label, weights in CONFIGS.items():
        coeffs = compute_coefficients(instance, weights, current_routing)
        cost_fn = build_cost_function(coeffs)
        pick = min(records, key=lambda r: cost_fn(r[1]))
        row = {
            "config": label,
            "chosen": {
                instance.demands[k].name: p for k, p in enumerate(pick[0])
            },
            "keeps_today": all(p == 0 for p in pick[0]),
            "H": round(cost_fn(pick[1]), 6),
            **kpis(pick[2]),
        }
        rows.append(row)
        print(
            f"{label:24s} : max_util={row['max_util']:.2f} "
            f"viol={row['violations']:2d} Phi*={row['phi_star']:9.2f} "
            f"wl={row['weighted_latency']:8.1f} "
            f"{'(keeps today)' if row['keeps_today'] else ''}"
        )

    # Best configuration: the one whose optimum lands closest to the frontier.
    best = min(rows, key=lambda r: (r["violations"], r["phi_star"]))
    print(f"--> best configuration : {best['config']}")

    return {
        "region": region,
        "n_pops": inst.n_nodes,
        "n_links": inst.n_links,
        "n_demands": len(instance.demands),
        "num_qubits": instance.num_qubits,
        "n_assignments": len(assignments),
        "today": kpis(today[2]),
        "frontier_min_violations": kpis(frontier[2]),
        "frontier_min_phi_star": kpis(best_phi[2]),
        "configs": rows,
        "best_config": best["config"],
    }


def main() -> None:
    payload = {region: study(region) for region in ("east10", "east10_dense")}
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
