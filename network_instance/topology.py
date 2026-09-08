"""
Small backbone-style network instance for the AT&T traffic-routing challenge.

This is a hand-built, clearly-labeled synthetic instance (NOT real AT&T or
SNDlib data -- we have no organizer-provided dataset and did not want to
mis-recall exact real topology data from memory). It is deliberately built in
the same spirit as SNDlib instances (a capacitated backbone graph + a demand
matrix with bandwidth/latency/priority per demand, and a 1+1 dedicated
protection model for high-priority demands, mirroring SNDlib's own
survivability model). Swap `build_network`/`build_demands` for a loader over
a real SNDlib/Topology Zoo file before the final submission if there's time.

Kept small on purpose: 8 nodes / 12 links / 6 demands / up to `k` candidate
paths each keeps the QUBO to a qubit count that Classiq's simulator can
optimize quickly for a live demo.
"""

from dataclasses import dataclass, field
from itertools import islice

import networkx as nx


NODES = ["SEA", "DEN", "CHI", "NYC", "ATL", "DAL", "LAX", "MIA"]

# (u, v, capacity, delay_ms). Capacity is in units of 5 Gbps (e.g. 20 -> 100
# Gbps) rather than raw Gbps -- purely a rescaling for the quantum encoding
# (see `qaoa_model.py`): every inequality constraint's capacity bound needs
# its own slack register sized in bits by ceil(log2(capacity)), so smaller
# integers buy real qubit headroom on a NISQ-era simulator/device without
# changing which combinations of demands are feasible (dividing every
# bandwidth and capacity by the same constant is exact, not an approximation).
EDGES = [
    ("SEA", "DEN", 20, 8),
    ("DEN", "CHI", 16, 7),
    ("CHI", "NYC", 20, 6),
    ("NYC", "ATL", 12, 5),
    ("ATL", "MIA", 10, 4),
    ("ATL", "DAL", 8, 6),
    ("DAL", "LAX", 18, 9),
    ("LAX", "SEA", 20, 10),
    ("DEN", "DAL", 10, 5),
    ("CHI", "ATL", 14, 5),
    ("SEA", "CHI", 12, 12),  # chord -- fewer hops, longer physical delay
    ("NYC", "DAL", 8, 11),  # chord
]


@dataclass
class Demand:
    id: str
    src: str
    dst: str
    bandwidth: float  # Gbps
    latency_bound: float  # ms
    priority: str  # "high" | "medium" | "low"
    needs_backup: bool  # True -> 1+1 dedicated protection (2 disjoint-ish paths)

    @property
    def paths_required(self) -> int:
        return 2 if self.needs_backup else 1


# Kept to 4 demands (not 6) so the QUBO encoding -- x/y decision qubits plus
# one inequality-constraint slack register per genuinely contested link --
# fits comfortably under Classiq's 28-qubit simulator limit for tonight's
# live demo. D4 (NYC-LAX) and D6 (DAL-CHI) touch the same bottleneck edges
# and are kept in `EXTRA_DEMANDS` below to extend the instance once a wider
# backend or a decomposed/multi-run approach is in play (see RESEARCH_NOTES.md).
DEMANDS = [
    Demand("D1", "SEA", "ATL", bandwidth=7, latency_bound=30, priority="high", needs_backup=True),
    Demand("D2", "DEN", "MIA", bandwidth=5, latency_bound=40, priority="medium", needs_backup=False),
    Demand("D3", "CHI", "DAL", bandwidth=6, latency_bound=25, priority="medium", needs_backup=False),
    Demand("D5", "SEA", "NYC", bandwidth=8, latency_bound=35, priority="high", needs_backup=True),
]

EXTRA_DEMANDS = [
    Demand("D4", "NYC", "LAX", bandwidth=4, latency_bound=45, priority="low", needs_backup=False),
    Demand("D6", "DAL", "CHI", bandwidth=3, latency_bound=25, priority="low", needs_backup=False),
]

PRIORITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


def build_network() -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(NODES)
    for u, v, cap, delay in EDGES:
        g.add_edge(u, v, capacity=cap, delay=delay)
    return g


def build_demands() -> list[Demand]:
    return list(DEMANDS)


@dataclass
class CandidatePath:
    demand_id: str
    nodes: list[str]
    edges: list[tuple[str, str]] = field(default_factory=list)
    delay: float = 0.0
    within_latency: bool = True

    def __post_init__(self):
        if not self.edges:
            self.edges = list(zip(self.nodes[:-1], self.nodes[1:]))


def _path_delay(g: nx.Graph, nodes: list[str]) -> float:
    return sum(g[u][v]["delay"] for u, v in zip(nodes[:-1], nodes[1:]))


def generate_candidate_paths(
    g: nx.Graph, demands: list[Demand], k: int = 2
) -> dict[str, list[CandidatePath]]:
    """k shortest (by delay) simple paths per demand, preferring latency-feasible ones.

    If fewer than `k` feasible paths exist we fall back to the shortest
    available paths anyway (flagged via `within_latency=False`) rather than
    leaving a demand with zero candidates -- an unroutable-within-SLA demand
    is itself a useful finding to report, not a reason to crash.
    """
    candidates: dict[str, list[CandidatePath]] = {}
    for d in demands:
        gen = nx.shortest_simple_paths(g, d.src, d.dst, weight="delay")
        all_paths = list(islice(gen, max(k * 3, 6)))  # oversample, then filter
        scored = []
        for nodes in all_paths:
            delay = _path_delay(g, nodes)
            scored.append(CandidatePath(d.id, nodes, delay=delay, within_latency=delay <= d.latency_bound))
        scored.sort(key=lambda p: (not p.within_latency, p.delay))
        chosen = scored[:k]
        if not any(p.within_latency for p in chosen) and scored:
            # keep the best one anyway so the demand isn't silently dropped
            chosen = scored[:max(k, 1)]
        candidates[d.id] = chosen
    return candidates


# D3's two candidate paths (CHI-ATL-DAL vs CHI-DEN-DAL) differ by only 1ms
# of delay and load a similar part of the network, so capping it to its
# single best path barely constrains the search -- unlike D2, whose two
# candidates route through genuinely different bottleneck edges and whose
# choice materially changes which demands the network can afford to admit.
# Keeping full 2-path choice for every demand pushed the QUBO to 33 logical
# qubits, just over Classiq's 28-qubit simulator cap (see RESEARCH_NOTES.md);
# this cap buys back qubits without removing the alternative that matters.
# Used by classical and quantum solvers alike so both see the same instance.
CANDIDATE_CAP_OVERRIDE = {"D3": 1}


def default_candidates(g: nx.Graph, demands: list[Demand], k: int = 2) -> dict[str, list[CandidatePath]]:
    candidates = generate_candidate_paths(g, demands, k=k)
    for demand_id, cap in CANDIDATE_CAP_OVERRIDE.items():
        if demand_id in candidates:
            candidates[demand_id] = candidates[demand_id][:cap]
    return candidates


if __name__ == "__main__":
    g = build_network()
    demands = build_demands()
    cands = generate_candidate_paths(g, demands, k=2)
    for d in demands:
        print(f"\n{d.id}: {d.src}->{d.dst}  bw={d.bandwidth}Gbps  "
              f"latency<={d.latency_bound}ms  priority={d.priority}  "
              f"paths_required={d.paths_required}")
        for p in cands[d.id]:
            flag = "" if p.within_latency else "  [!] exceeds latency bound"
            print(f"    {'-'.join(p.nodes)}  delay={p.delay}ms{flag}")
