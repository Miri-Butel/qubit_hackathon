"""Decode sampled bitstrings into routing solutions with classical KPIs.

KPIs are reported in raw units (real latency, utilization, Fortz-Thorup Phi);
the normalization used inside the Hamiltonian never leaks into KPIs.

Phi* = Phi / Phi_Uncap follows Fortz & Thorup (INFOCOM 2000), eqs. 10-11:
Phi_Uncap = sum_k b_k * hop_dist(s_k, t_k). Phi* = 1 is ideal (all traffic on
unit-weight shortest paths, all utilizations < 1/3); Phi* >= 32/3 (~10.67)
means the network is congested.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from .model import LinkKey, NodeId, RoutingInstance
from .qubo import CostCoefficients, build_cost_function, fortz_thorup_link_cost

CONGESTION_THRESHOLD_PHI_STAR = 32 / 3


@dataclass(frozen=True)
class LinkKpi:
    link: LinkKey
    load: float
    utilization: float
    over_capacity: bool


@dataclass(frozen=True)
class RoutingSolution:
    chosen: dict[str, int | None]
    feasible_onehot: bool
    link_kpis: tuple[LinkKpi, ...]
    max_utilization: float
    total_latency: float
    weighted_latency: float
    capacity_violations: int
    demands_satisfied: int
    phi_total: float
    phi_star: float
    cost: float


def hop_distance(instance: RoutingInstance, source: NodeId, target: NodeId) -> int:
    """Unit-weight (hop count) shortest-path distance over all links, via BFS."""
    successors: dict[NodeId, list[NodeId]] = {}
    for link in instance.links:
        successors.setdefault(link.u, []).append(link.v)
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            return dist[node]
        for nxt in successors.get(node, ()):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                queue.append(nxt)
    raise ValueError(f"no route from {source!r} to {target!r} in the link graph")


def phi_uncap(instance: RoutingInstance) -> float:
    """Fortz-Thorup uncapacitated lower bound: sum_k b_k * hop_dist(s_k, t_k)."""
    return sum(
        demand.bandwidth * hop_distance(instance, demand.source, demand.target)
        for demand in instance.demands
    )


def chosen_paths(bits: Sequence[int], instance: RoutingInstance) -> dict[str, int | None]:
    """Per demand: the selected path index, or None if one-hot is violated."""
    chosen: dict[str, int | None] = {}
    for k, demand in enumerate(instance.demands):
        selected = [
            p
            for p in range(len(instance.paths_of(k)))
            if bits[instance.flat_index(k, p)]
        ]
        chosen[demand.name] = selected[0] if len(selected) == 1 else None
    return chosen


def onehot_feasible(bits: Sequence[int], instance: RoutingInstance) -> bool:
    return all(p is not None for p in chosen_paths(bits, instance).values())


def decode_bitstring(
    bits: Sequence[int], instance: RoutingInstance, coeffs: CostCoefficients
) -> RoutingSolution:
    """Decode one bitstring: chosen routes, link loads/utilization, latency, Phi KPIs.

    Link loads count only demands whose one-hot constraint holds (a demand with
    zero or multiple selected paths is unsatisfied and carries no traffic).
    """
    chosen = chosen_paths(bits, instance)
    loads: dict[LinkKey, float] = {link.key: 0.0 for link in instance.links}
    total_latency = 0.0
    weighted_latency = 0.0
    for k, demand in enumerate(instance.demands):
        p = chosen[demand.name]
        if p is None:
            continue
        latency = instance.path_latency(k, p)
        total_latency += latency
        weighted_latency += demand.bandwidth * latency
        for key in instance.paths_of(k)[p].links:
            loads[key] += demand.bandwidth

    link_kpis = tuple(
        LinkKpi(
            link=link.key,
            load=loads[link.key],
            utilization=loads[link.key] / link.capacity,
            over_capacity=loads[link.key] > link.capacity,
        )
        for link in instance.links
    )
    phi_total = sum(
        fortz_thorup_link_cost(loads[link.key], link.capacity)
        for link in instance.links
    )
    return RoutingSolution(
        chosen=chosen,
        feasible_onehot=all(p is not None for p in chosen.values()),
        link_kpis=link_kpis,
        max_utilization=max(kpi.utilization for kpi in link_kpis),
        total_latency=total_latency,
        weighted_latency=weighted_latency,
        capacity_violations=sum(kpi.over_capacity for kpi in link_kpis),
        demands_satisfied=sum(p is not None for p in chosen.values()),
        phi_total=phi_total,
        phi_star=phi_total / phi_uncap(instance),
        cost=float(build_cost_function(coeffs)(list(bits))),
    )


def _probabilities(samples: pd.DataFrame, num_shots: int) -> pd.Series:
    if "probability" in samples.columns:
        return samples["probability"]
    if "counts" in samples.columns:
        return samples["counts"] / num_shots
    raise ValueError(
        f"samples has neither 'probability' nor 'counts'; columns: {list(samples.columns)}"
    )


def _bits_of(row: pd.Series) -> list[int]:
    if "x" not in row:
        raise ValueError(f"samples row has no 'x' column; columns: {list(row.index)}")
    return [int(b) for b in row["x"]]


def top_solutions(
    samples: pd.DataFrame,
    instance: RoutingInstance,
    coeffs: CostCoefficients,
    num_shots: int,
    k: int = 10,
) -> pd.DataFrame:
    """Decode the k most-sampled bitstrings into a KPI table, most probable first."""
    ranked = samples.assign(_prob=_probabilities(samples, num_shots))
    ranked = ranked.sort_values("_prob", ascending=False).head(k)
    rows = []
    for _, row in ranked.iterrows():
        bits = _bits_of(row)
        sol = decode_bitstring(bits, instance, coeffs)
        rows.append(
            {
                "probability": row["_prob"],
                "bits": "".join(map(str, bits)),
                **{f"path[{name}]": p for name, p in sol.chosen.items()},
                "feasible": sol.feasible_onehot,
                "cap_violations": sol.capacity_violations,
                "max_util": sol.max_utilization,
                "weighted_latency": sol.weighted_latency,
                "phi_star": sol.phi_star,
                "cost": sol.cost,
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


def feasible_probability(
    samples: pd.DataFrame, instance: RoutingInstance, num_shots: int
) -> float:
    """Total probability mass on one-hot-feasible bitstrings."""
    probs = _probabilities(samples, num_shots)
    return float(
        sum(
            prob
            for (_, row), prob in zip(samples.iterrows(), probs)
            if onehot_feasible(_bits_of(row), instance)
        )
    )
