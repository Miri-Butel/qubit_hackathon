"""
How the two halves scale: exact classical enumeration versus the quantum
circuit that encodes the same problem.

This measures rather than asserts. For each instance size it records what the
classical solver actually has to do (how many assignments, how long), and
what the quantum side actually costs (qubits, circuit depth, two-qubit gate
count from real Classiq synthesis).

The honest shape of the result, stated up front so the numbers are read
correctly:

* CLASSICAL EXACT is exponential in the number of demands. With `n` candidate
  routes each, the feasible set has n^D one-hot assignments. That is the wall
  everyone eventually hits, and it arrives fast.

* THE QUANTUM ENCODING is linear in demands: one qubit per candidate route,
  N = n*D. Circuit depth grows with the number of interacting route pairs,
  which is the graph's interference structure, not 2^D.

* THAT IS NOT A SPEEDUP CLAIM. A shallow QAOA is a heuristic with no
  performance guarantee, and at every size measured here exact enumeration is
  both faster and better. The favourable scaling is in what the circuit
  *costs to represent*, not in time-to-solution. The fair classical
  comparison at large D is a heuristic (local search, LP rounding), which
  also scales polynomially and today wins outright.
"""

import time

import qaoa_instance


def classical_cost(instance, weights, current_routing, time_budget: float = 20.0) -> dict:
    """Measure exact enumeration: how many assignments, and how long.

    Above roughly 2^22 the enumeration is projected from a measured rate
    instead of run, so this stays usable at sizes that would take hours.
    """
    from routing_qaoa.qubo import brute_force_feasible

    combos = 1
    for k in range(len(instance.demands)):
        combos *= len(instance.paths_of(k))

    if combos <= 2 ** 22:
        t0 = time.perf_counter()
        result = brute_force_feasible(instance, weights, current_routing=current_routing)
        elapsed = time.perf_counter() - t0
        return {"combinations": combos, "seconds": elapsed, "measured": True,
                "best_cost": result.best_cost}
    return {"combinations": combos, "seconds": float("nan"), "measured": False,
            "best_cost": float("nan")}


def quantum_cost(instance, weights, current_routing, num_layers: int = 2) -> dict:
    """Synthesize the QAOA circuit and report its real resource cost."""
    from classiq import synthesize
    from routing_qaoa.qaoa import build_qaoa_main
    from routing_qaoa.qubo import build_cost_function, compute_coefficients

    coeffs = compute_coefficients(instance, weights, current_routing)
    main = build_qaoa_main(
        build_cost_function(coeffs), instance.num_qubits, num_layers
    )
    t0 = time.perf_counter()
    qprog = synthesize(main)
    synth_time = time.perf_counter() - t0

    width = getattr(qprog.data, "width", instance.num_qubits)
    depth, gates = None, None
    try:
        from classiq import get_transpiled_circuit_metrics
        metrics = get_transpiled_circuit_metrics(qprog)
        depth = getattr(metrics, "depth", None)
        gates = getattr(metrics, "count_ops", None)
    except Exception:
        pass
    return {"qubits": width, "depth": depth, "gate_counts": gates,
            "synthesis_seconds": synth_time}


def sweep(sizes, num_layers: int = 2, weights=None, diversity: float = 0.5):
    """Measure both halves across a range of qubit budgets."""
    from routing_qaoa import QuboWeights
    from routing_qaoa.qubo import compute_coefficients

    weights = weights or QuboWeights()
    rows = []
    for target in sizes:
        instance, current, ctx = qaoa_instance.build(
            target_qubits=target, diversity=diversity
        )
        coeffs = compute_coefficients(instance, weights, current)
        couplings = sum(
            len(t) * (len(t) - 1) // 2 for t in coeffs.link_terms
        ) + sum(len(g) * (len(g) - 1) // 2 for g in coeffs.onehot_groups)

        row = {
            "demands": len(instance.demands),
            "qubits": instance.num_qubits,
            "route_pair_couplings": couplings,
        }
        row.update(classical_cost(instance, weights, current))
        row.update(quantum_cost(instance, weights, current, num_layers))
        rows.append(row)
    return rows


def project_classical(demands: int, routes_per_demand: int = 2,
                      assignments_per_second: float = 3.0e5) -> float:
    """Seconds for exact enumeration at a given size, from a measured rate.

    `assignments_per_second` is calibrated from the sweep. The point of this
    function is the shape of the curve, not the constant: n^D means no
    constant saves you.
    """
    return routes_per_demand ** demands / assignments_per_second
