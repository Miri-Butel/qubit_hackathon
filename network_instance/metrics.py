"""
Shared scoring for any routing assignment (classical greedy, exact ILP, or
QAOA sample) so every solver in this folder is judged on the exact same
yardstick -- the five bullets from the challenge brief's "What to measure"
section, plus the approximation ratio against the exact ILP optimum.

Modeling choice worth stating explicitly: capacity is a hard physical limit
that must never be exceeded, but *admission* is a decision, not a given --
a demand is either carried with its full required redundancy or it is
declined outright, never carried with only partial protection (that would
silently violate the brief's "no violations of resiliency rules" bullet).
So "demand satisfaction" here means bandwidth-weighted admission rate, and
`redundancy_violations` flags only the case a solver should never produce
by construction (a demand partially, rather than fully, protected) -- for
the exact ILP that count is always 0; for QAOA's penalty-based sampling it
is a genuine signal of how far a given sample is from feasible.
"""

from dataclasses import dataclass

import networkx as nx

from topology import CandidatePath, Demand


Assignment = dict[str, list[int]]  # demand_id -> list of chosen candidate-path indices

DEFAULT_ALPHA = 1.0    # weight on link-congestion term
DEFAULT_BETA = 0.02    # weight on latency term
DEFAULT_GAMMA = 5.0    # weight on the per-demand rejection penalty


@dataclass
class Metrics:
    max_link_utilization: float  # 0..1+ (values > 1 mean an overloaded link)
    avg_latency_ms: float
    demand_satisfaction_pct: float  # bandwidth-weighted: % of requested Gbps actually carried
    rejected_demands: int
    route_changes_vs_baseline: int | None
    capacity_violations: int
    redundancy_violations: int  # demands carried with 0 < paths < required (should never happen)
    objective_value: float

    def is_feasible(self) -> bool:
        return self.capacity_violations == 0 and self.redundancy_violations == 0

    def report(self, label: str) -> str:
        lines = [f"--- {label} ---"]
        lines.append(f"  max link utilization      : {self.max_link_utilization:6.1%}")
        lines.append(f"  avg latency (chosen paths): {self.avg_latency_ms:6.2f} ms")
        lines.append(f"  demand satisfaction (Gbps): {self.demand_satisfaction_pct:6.1f} %")
        lines.append(f"  demands rejected          : {self.rejected_demands}")
        if self.route_changes_vs_baseline is not None:
            lines.append(f"  route changes vs baseline : {self.route_changes_vs_baseline}")
        lines.append(f"  capacity violations       : {self.capacity_violations}")
        lines.append(f"  redundancy violations     : {self.redundancy_violations}")
        lines.append(f"  objective (lower=better)  : {self.objective_value:.4f}")
        feasible = "FEASIBLE" if self.is_feasible() else "INFEASIBLE"
        lines.append(f"  -> {feasible}")
        return "\n".join(lines)


def _edge_load(
    assignment: Assignment, g: nx.Graph, demands: list[Demand], candidates: dict[str, list[CandidatePath]]
) -> dict[frozenset, float]:
    load = {frozenset(e): 0.0 for e in g.edges()}
    for d in demands:
        for idx in assignment.get(d.id, []):
            for e in candidates[d.id][idx].edges:
                load[frozenset(e)] += d.bandwidth
    return load


def objective_value(
    assignment: Assignment,
    g: nx.Graph,
    demands: list[Demand],
    candidates: dict[str, list[CandidatePath]],
    priority_weight: dict[str, float],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """alpha*congestion + beta*latency + gamma*sum_d weight[d]*(1 - admitted[d]).

    `admitted[d]` is 1 only when d received exactly its required number of
    paths -- a partially-protected demand counts as *not* admitted here, so
    there is never an incentive to half-protect a demand instead of
    rejecting or fully protecting it. Same formula is used by the classical
    ILP baseline and the quantum QUBO -- this is what "best" means for
    every solver in this folder.
    """
    load = _edge_load(assignment, g, demands, candidates)
    latency_term = 0.0
    rejection_term = 0.0
    for d in demands:
        chosen = assignment.get(d.id, [])
        for idx in chosen:
            latency_term += priority_weight[d.priority] * candidates[d.id][idx].delay
        admitted = 1.0 if len(chosen) == d.paths_required else 0.0
        rejection_term += priority_weight[d.priority] * (1.0 - admitted)
    util_term = sum(load[frozenset((u, v))] / data["capacity"] for u, v, data in g.edges(data=True))
    return alpha * util_term + beta * latency_term + gamma * rejection_term


def compute_metrics(
    assignment: Assignment,
    g: nx.Graph,
    demands: list[Demand],
    candidates: dict[str, list[CandidatePath]],
    priority_weight: dict[str, float],
    baseline: Assignment | None = None,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> Metrics:
    load = _edge_load(assignment, g, demands, candidates)
    delays: list[float] = []
    carried_bw = 0.0
    total_bw = sum(d.bandwidth for d in demands)
    rejected = 0
    redundancy_violations = 0

    for d in demands:
        chosen = assignment.get(d.id, [])
        for idx in chosen:
            delays.append(candidates[d.id][idx].delay)
        if len(chosen) == d.paths_required:
            carried_bw += d.bandwidth
        elif len(chosen) == 0:
            rejected += 1
        else:
            redundancy_violations += 1  # partially protected -- should never happen by construction

    capacity_violations = 0
    max_util = 0.0
    for u, v, data in g.edges(data=True):
        util = load[frozenset((u, v))] / data["capacity"]
        max_util = max(max_util, util)
        if util > 1.0 + 1e-9:
            capacity_violations += 1

    route_changes = None
    if baseline is not None:
        route_changes = sum(
            1 for d in demands if sorted(assignment.get(d.id, [])) != sorted(baseline.get(d.id, []))
        )

    return Metrics(
        max_link_utilization=max_util,
        avg_latency_ms=sum(delays) / len(delays) if delays else float("nan"),
        demand_satisfaction_pct=100.0 * carried_bw / total_bw if total_bw else 100.0,
        rejected_demands=rejected,
        route_changes_vs_baseline=route_changes,
        capacity_violations=capacity_violations,
        redundancy_violations=redundancy_violations,
        objective_value=objective_value(assignment, g, demands, candidates, priority_weight, alpha, beta, gamma),
    )


def approximation_ratio(candidate_obj: float, optimal_obj: float) -> float:
    """Both objectives are minimized and non-negative; ratio in (0, 1], 1.0 = optimal."""
    if candidate_obj <= 1e-9:
        return 1.0 if optimal_obj <= 1e-9 else 0.0
    return min(1.0, optimal_obj / candidate_obj)
