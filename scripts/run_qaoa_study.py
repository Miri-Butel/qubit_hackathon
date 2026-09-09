"""Run QAOA on an instance and cache everything the deck and notebook need.

Usage:
    python scripts/run_qaoa_study.py toy
    python scripts/run_qaoa_study.py east10_tuned [--layers 1] [--shots 2048]
                                                  [--iterations 20]

Writes docs/qaoa_<target>.json: config, circuit metrics, convergence trace,
brute-force reference, feasible probability mass, and the top sampled states.
Every number the pitch quotes should come from one of these files, so nothing
on a slide is a stale notebook output.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "network_instance"))

from classiq import synthesize  # noqa: E402

from routing_qaoa import (  # noqa: E402
    CandidatePath,
    Demand,
    Link,
    QaoaConfig,
    QuboWeights,
    RoutingInstance,
    brute_force_feasible,
    compute_coefficients,
    decode_bitstring,
    run_qaoa,
)
from routing_qaoa.decode import bits_of, sample_probabilities  # noqa: E402
from routing_qaoa.qaoa import build_qaoa_main  # noqa: E402
from routing_qaoa.qubo import build_cost_function  # noqa: E402

# The toy instance mirrors qaoa_routing.ipynb / scripts/render_figures.py.
TOY_LINKS = (
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
TOY_DEMANDS = (
    Demand("d1", source="A", target="F", bandwidth=4, priority=3.0),
    Demand("d2", source="A", target="E", bandwidth=3),
    Demand("d3", source="B", target="F", bandwidth=3),
)


def toy_instance() -> tuple[RoutingInstance, QuboWeights]:
    import networkx as nx

    graph = nx.DiGraph()
    for link in TOY_LINKS:
        graph.add_edge(link.u, link.v, latency=link.latency)
    paths = {}
    for demand in TOY_DEMANDS:
        walks = nx.shortest_simple_paths(
            graph, demand.source, demand.target, weight="latency"
        )
        chosen = []
        for nodes in walks:
            chosen.append(CandidatePath(tuple(zip(nodes[:-1], nodes[1:]))))
            if len(chosen) == 3:
                break
        paths[demand.name] = tuple(chosen)
    weights = QuboWeights(congestion_profile="fortz-thorup-fit", cost_scale=0.1)
    return RoutingInstance(TOY_LINKS, TOY_DEMANDS, paths), weights


def real_instance(region: str, diversity: float = 0.5
                  ) -> tuple[RoutingInstance, QuboWeights]:
    import qaoa_instance

    instance, _, _ = qaoa_instance.build(
        region=region, target_qubits=28, diversity=diversity
    )
    # The per-link profile is what keeps the capacity cliff in the QUBO; the
    # single global quadratic leaves the peak link over capacity.
    return instance, QuboWeights(congestion_profile="fortz-thorup-perlink")


def circuit_metrics(qprog) -> dict:
    out = {}
    data = getattr(qprog, "data", None)
    for name in ("width", "depth"):
        value = getattr(data, name, None)
        if value is not None:
            out[name] = value
    try:
        from classiq import get_transpiled_circuit_metrics

        metrics = get_transpiled_circuit_metrics(qprog)
        out["transpiled_depth"] = metrics.depth
        out["gate_counts"] = dict(metrics.count_ops)
        out["two_qubit_gates"] = sum(
            n for gate, n in metrics.count_ops.items() if gate in {"cx", "cz", "ecr"}
        )
    except Exception:  # noqa: BLE001 - metrics are a nice-to-have
        pass
    return out


def kpis(solution) -> dict:
    return {
        "chosen": solution.chosen,
        "feasible_onehot": solution.feasible_onehot,
        "max_utilization": round(solution.max_utilization, 4),
        "capacity_violations": solution.capacity_violations,
        "total_latency": round(solution.total_latency, 3),
        "weighted_latency": round(solution.weighted_latency, 3),
        "phi_total": round(solution.phi_total, 3),
        "phi_star": round(solution.phi_star, 4),
        "cost": round(solution.cost, 6),
        "demands_satisfied": solution.demands_satisfied,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--quantile", type=float, default=0.5)
    args = parser.parse_args()

    if args.target == "toy":
        instance, weights = toy_instance()
    else:
        instance, weights = real_instance(args.target)

    coeffs = compute_coefficients(instance, weights)
    cost_fn = build_cost_function(coeffs)
    n = instance.num_qubits
    print(f"{args.target}: {n} qubits, {len(instance.demands)} demands")

    # Classical ground truth over the one-hot subspace.
    t0 = time.perf_counter()
    ref = brute_force_feasible(instance, weights)
    ref_solution = decode_bitstring(list(ref.best_bits), instance, coeffs)
    print(
        f"brute force: {ref.num_combinations} assignments in "
        f"{time.perf_counter() - t0:.1f}s -> H={ref.best_cost:.6f} "
        f"peak={ref_solution.max_utilization:.2%} "
        f"viol={ref_solution.capacity_violations}"
    )

    # Today's routing = candidate 0 for every demand.
    today_bits = [0] * n
    for k in range(len(instance.demands)):
        today_bits[instance.flat_index(k, 0)] = 1
    today = decode_bitstring(today_bits, instance, coeffs)

    config = QaoaConfig(
        num_layers=args.layers,
        num_shots=args.shots,
        max_iterations=args.iterations,
        random_seed=42,
        quantile=args.quantile,
        sample_initial=True,
    )

    t0 = time.perf_counter()
    qprog = synthesize(build_qaoa_main(cost_fn, n, config.num_layers))
    metrics = circuit_metrics(qprog)
    print(f"synthesis {time.perf_counter() - t0:.1f}s -> {metrics}")

    t0 = time.perf_counter()
    result = run_qaoa(instance, weights, config, qprog=qprog)
    runtime = time.perf_counter() - t0
    print(
        f"QAOA {runtime:.1f}s, {len(result.objective_values)} cost evaluations, "
        f"CVaR {result.objective_values[0]:.4f} -> {min(result.objective_values):.4f}"
    )

    # Feasible probability mass, and the best feasible sample if there is one.
    probs = sample_probabilities(result.samples, result.num_shots)
    rows = []
    for (_, row), p in zip(result.samples.iterrows(), probs):
        bits = bits_of(row)
        solution = decode_bitstring(bits, instance, coeffs)
        rows.append({"probability": float(p), "bits": "".join(map(str, bits)),
                     **kpis(solution)})
    rows.sort(key=lambda r: -r["probability"])
    feasible = [r for r in rows if r["feasible_onehot"]]
    feasible_mass = sum(r["probability"] for r in feasible)
    uniform_mass = ref.num_combinations / (2 ** n)
    print(
        f"feasible mass {feasible_mass:.4%} vs uniform {uniform_mass:.4%} "
        f"({feasible_mass / uniform_mass:.1f}x) over {len(feasible)} feasible samples"
    )

    best = min(feasible, key=lambda r: r["cost"]) if feasible else None
    if best:
        gap = best["cost"] - ref.best_cost
        print(
            f"best feasible sample: p={best['probability']:.4%} "
            f"H={best['cost']:.6f} gap={gap:.6f} "
            f"peak={best['max_utilization']:.2%} viol={best['capacity_violations']} "
            f"matches_optimum={abs(gap) < 1e-9}"
        )
    else:
        print("no feasible sample -- see the one-hot feasible fraction above")

    payload = {
        "target": args.target,
        "num_qubits": n,
        "num_demands": len(instance.demands),
        "num_links": len(instance.links),
        "weights": {
            "congestion_profile": weights.congestion_profile,
            "lambda_cong": weights.lambda_cong,
            "lambda_onehot": weights.lambda_onehot,
            "cost_scale": weights.cost_scale,
        },
        "config": {
            "num_layers": config.num_layers, "num_shots": config.num_shots,
            "max_iterations": config.max_iterations, "quantile": config.quantile,
            "random_seed": config.random_seed,
        },
        "circuit": metrics,
        "runtime_seconds": round(runtime, 1),
        "brute_force": {
            "num_combinations": ref.num_combinations,
            "best_cost": ref.best_cost,
            "best_assignment": ref.best_assignment,
            **kpis(ref_solution),
        },
        "today": kpis(today),
        "feasible_fraction_uniform": uniform_mass,
        "feasible_mass": feasible_mass,
        "feasible_sample_count": len(feasible),
        "distinct_samples": len(rows),
        "best_feasible": best,
        "objective_values": result.objective_values,
        "top_samples": rows[:15],
    }
    out = ROOT / "docs" / f"qaoa_{args.target}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
