"""
One call that turns the real AT&T backbone into a `routing_qaoa` problem
sized to a chosen qubit budget.

This is the join between the two halves of the pipeline. Where the two sides
modelled something differently, **routing_qaoa's model wins** and this module
adapts to it. Three places that matters:

* LATENCY IS PRICED, NOT BOUNDED. Our generator carries a per-demand
  `latency_bound` and treats routes exceeding it as illegal. `routing_qaoa`
  has no such notion: latency enters the Hamiltonian as a cost term weighted
  by bandwidth and priority, so a slow route is expensive rather than
  forbidden. So candidates here are *not* filtered by the bound. The bound
  travels along as metadata (`route_notes`) for reporting, nothing more.

* CAPACITY IS PRICED, NOT ENFORCED. Same story. Their congestion term is a
  soft quadratic on utilization, so we do not prune, cap, or admission-control
  anything. Every demand gets routed; the objective decides where.

* PROTECTION DOES NOT SURVIVE. Their model selects exactly one path per
  demand. A demand we generated as 1+1 protected is exported as an ordinary
  demand. `protected_demands()` in `export.py` lists which ones lost that
  guarantee, so nobody discovers it by accident in the results.

What does carry over cleanly: priority. `routing_qaoa.Demand.priority`
multiplies the latency cost, and our high/medium/low classes map onto it via
`export.PRIORITY_TO_COEFFICIENT`.
"""

import att_real
import export
import yen


def _paths_per_demand(demands, target_qubits: int, min_paths: int = 2) -> dict[str, int]:
    """Split a qubit budget across demands: one qubit per candidate path.

    Everyone gets `min_paths` so every demand is a real decision, then the
    remainder goes to the largest demands first, since a spare route is worth
    most where the most bandwidth rides on the choice.
    """
    ranked = sorted(demands, key=lambda d: -d.bandwidth)
    if min_paths * len(ranked) > target_qubits:
        # The budget cannot seat every demand with a real choice, so carry the
        # heaviest demands and drop the rest. The alternative, giving everyone
        # a single forced route, spends a qubit per demand on no decision at
        # all: a one-hot group of size 1 is pinned to 1 by its own penalty.
        ranked = ranked[: target_qubits // min_paths]
    counts = {d.id: min_paths for d in ranked}
    for d in ranked[: target_qubits - min_paths * len(ranked)]:
        counts[d.id] += 1
    return counts


def build(region: str = "east10_dense", target_qubits: int = 28, k: int = 5,
          diversity: float = 0.5):
    """Build a `routing_qaoa` instance from the real AT&T backbone.

    Returns `(instance, current_routing, context)` where `instance` is a
    `RoutingInstance` with exactly `target_qubits` candidate paths,
    `current_routing` maps each demand to the index of the route it runs
    today (what their `H_switch` term measures churn against), and `context`
    carries the generator-side detail their model has no field for.
    """
    net = att_real.att_backbone(region=region)
    counts = _paths_per_demand(net.demands, target_qubits)

    # A tight budget may have dropped demands; keep the instance consistent
    # with what the budget actually seats.
    net.demands[:] = [d for d in net.demands if d.id in counts]
    for demand_id in list(net.current_routing):
        if demand_id not in counts:
            del net.current_routing[demand_id]

    # Latency bounds are deliberately NOT applied as a filter here: downstream
    # prices latency instead of forbidding it. Candidates are the best routes
    # under all four metrics (delay, opex, hops, spare capacity), merged.
    candidates = yen.build_candidate_set(net, k=k, n=counts, diversity=diversity)

    instance, current_routing = export.to_routing_instance(net, candidates)

    context = {
        "region": region,
        "source_instance": net,
        "candidates": candidates,
        "protected_demands": export.protected_demands(net),
        "route_notes": {
            d.id: [
                {
                    "route": "-".join(r.nodes),
                    "delay_ms": round(r.delay, 1),
                    "opex": r.cost,
                    "within_generator_latency_bound": r.within_latency,
                }
                for r in candidates[d.id]
            ]
            for d in net.demands
        },
    }
    return instance, current_routing, context


def summary(instance, current_routing, context) -> str:
    net = context["source_instance"]
    over_bound = sum(
        1 for notes in context["route_notes"].values()
        for n in notes if not n["within_generator_latency_bound"]
    )
    lines = [
        f"region            : {context['region']}",
        f"PoPs / links      : {net.n_nodes} / {net.n_links}  (real AT&T backbone)",
        f"demands           : {len(instance.demands)}",
        f"candidate paths   : {instance.num_qubits}  = qubits",
        f"routes past our latency bound (kept: downstream prices latency): {over_bound}",
        f"demands that lost 1+1 protection downstream: {context['protected_demands'] or 'none'}",
    ]
    return "\n".join(lines)
