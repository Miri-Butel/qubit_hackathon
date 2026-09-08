"""
Candidate network instances for the AT&T traffic-routing challenge.

Every element the brief asks for is represented here:

  brief says                                    -> where it lives
  ------------------------------------------------------------------------
  "graph of nodes and links"                    -> Instance.graph
  "finite capacities"                           -> edge attr `capacity` (Gbps)
  "delays"                                      -> edge attr `delay` (ms)
  "operational costs"                           -> edge attr `cost` (per Gbps carried)
  "multiple source-destination demands"         -> Instance.demands
  "different bandwidth ... requirements"        -> Demand.bandwidth
  "... latency ... requirements"                -> Demand.latency_bound
  "... priority requirements"                   -> Demand.priority
  "redundancy requirements"                     -> Demand.needs_backup (1+1 protection)
  "route changes against the current state"     -> Instance.current_routing

`current_routing` is what the network is running today: plain shortest-path
routing plus a backup path for each protected demand. Under the demand sets
below it is *already overloaded*, which is the premise of the whole exercise
-- traffic grew past what the legacy config can carry, and the job is to
re-optimize it with as few route changes as possible.

Three scales, so we pick deliberately rather than by accident. All three are
synthetic instances we built ourselves, in the style of SNDlib fixed-telecom
benchmarks (capacitated graph + demand matrix + 1+1 dedicated protection,
which is SNDlib's own survivability model). They are NOT real AT&T or real
SNDlib data -- we have no organizer-provided dataset, and inventing "real"
topology numbers from memory would be worse than saying plainly that these
are ours. Swapping in a real SNDlib or Topology Zoo file later just means
writing a loader that returns the same `Instance` shape.

Units: capacity and bandwidth in Gbps, delay in ms, cost in arbitrary
per-Gbps operational units (higher on long-haul / leased spans). The quantum
encoding rescales capacities via `scaled_copy`, because every inequality
constraint's slack register costs qubits proportional to the bit-length of
its capacity bound.
"""

from dataclasses import dataclass, field, replace

import networkx as nx

from topology import Demand


@dataclass
class Instance:
    name: str
    headline: str
    graph: nx.Graph
    demands: list[Demand]
    current_routing: dict[str, list[list[str]]] = field(default_factory=dict)
    pos: dict[str, tuple[float, float]] = field(default_factory=dict)
    unit_gbps: float = 1.0  # 1 capacity unit == this many Gbps

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_links(self) -> int:
        return self.graph.number_of_edges()

    @property
    def total_demand(self) -> float:
        """Gbps requested, counting a protected demand once per required path."""
        return sum(d.bandwidth * d.paths_required for d in self.demands)

    @property
    def total_capacity(self) -> float:
        return sum(data["capacity"] for _, _, data in self.graph.edges(data=True))

    def path_delay(self, nodes: list[str]) -> float:
        return sum(self.graph[u][v]["delay"] for u, v in zip(nodes[:-1], nodes[1:]))

    def path_cost(self, nodes: list[str]) -> float:
        return sum(self.graph[u][v]["cost"] for u, v in zip(nodes[:-1], nodes[1:]))

    def link_load(self, routing: dict[str, list[list[str]]]) -> dict[frozenset, float]:
        """Gbps on each link under a routing given as demand_id -> node paths."""
        bw = {d.id: d.bandwidth for d in self.demands}
        load = {frozenset((u, v)): 0.0 for u, v in self.graph.edges()}
        for demand_id, paths in routing.items():
            for nodes in paths:
                for e in zip(nodes[:-1], nodes[1:]):
                    load[frozenset(e)] += bw[demand_id]
        return load

    def link_utilization(self, routing: dict[str, list[list[str]]]) -> dict[frozenset, float]:
        load = self.link_load(routing)
        return {
            frozenset((u, v)): load[frozenset((u, v))] / data["capacity"]
            for u, v, data in self.graph.edges(data=True)
        }


def _build(nodes, edges, positions) -> tuple[nx.Graph, dict]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for u, v, cap, delay, cost in edges:
        g.add_edge(u, v, capacity=cap, delay=delay, cost=cost)
    return g, positions


# --------------------------------------------------------------------------
# Option A -- mini6: smallest instance that still has a real routing conflict.
# For fast debugging, and for QAOA runs where you would rather spend qubits
# on more layers than on more problem.
# --------------------------------------------------------------------------

_MINI6_NODES = ["N1", "N2", "N3", "N4", "N5", "N6"]
# (u, v, capacity Gbps, delay ms, opex per Gbps)
_MINI6_EDGES = [
    ("N1", "N2", 100, 4, 2),
    ("N2", "N3", 60, 5, 2),
    ("N3", "N4", 80, 4, 2),
    ("N4", "N5", 60, 6, 3),
    ("N5", "N6", 100, 5, 2),
    ("N6", "N1", 80, 4, 2),
    ("N1", "N4", 50, 9, 5),   # chord: short in hops, long and pricey in reality
    ("N2", "N5", 40, 8, 4),   # chord
]
_MINI6_POS = {
    "N1": (0.0, 0.5), "N2": (0.5, 1.0), "N3": (1.0, 0.5),
    "N4": (1.0, -0.5), "N5": (0.5, -1.0), "N6": (0.0, -0.5),
}
_MINI6_DEMANDS = [
    Demand("A1", "N1", "N4", bandwidth=40, latency_bound=15, priority="high", needs_backup=True),
    Demand("A2", "N2", "N5", bandwidth=35, latency_bound=18, priority="medium", needs_backup=False),
    Demand("A3", "N6", "N3", bandwidth=30, latency_bound=20, priority="medium", needs_backup=False),
]
_MINI6_CURRENT = {
    "A1": [["N1", "N4"], ["N1", "N2", "N3", "N4"]],  # primary + 1+1 backup
    "A2": [["N2", "N5"]],
    "A3": [["N6", "N1", "N2", "N3"]],
}


# --------------------------------------------------------------------------
# Option B -- backbone8: US-backbone-flavoured, 8 PoPs. Sized so the full
# QUBO (path vars + admission vars + capacity slack registers) fits inside
# Classiq's 28-qubit simulator limit after light candidate pruning.
# --------------------------------------------------------------------------

_BB8_NODES = ["SEA", "DEN", "CHI", "NYC", "ATL", "DAL", "LAX", "MIA"]
_BB8_EDGES = [
    ("SEA", "DEN", 100, 8, 3),
    ("DEN", "CHI", 80, 7, 2),
    ("CHI", "NYC", 100, 6, 2),
    ("NYC", "ATL", 60, 5, 2),
    ("ATL", "MIA", 50, 4, 2),
    ("ATL", "DAL", 40, 6, 3),
    ("DAL", "LAX", 90, 9, 3),
    ("LAX", "SEA", 100, 10, 4),
    ("DEN", "DAL", 50, 5, 2),
    ("CHI", "ATL", 70, 5, 2),
    ("SEA", "CHI", 60, 12, 5),  # leased long-haul chord: few hops, high delay AND high opex
    ("NYC", "DAL", 40, 11, 5),  # leased long-haul chord
]
_BB8_POS = {
    "SEA": (0.00, 1.00), "LAX": (0.05, 0.25), "DEN": (0.35, 0.60),
    "DAL": (0.45, 0.15), "CHI": (0.65, 0.80), "ATL": (0.75, 0.30),
    "NYC": (1.00, 0.75), "MIA": (0.90, 0.00),
}
_BB8_DEMANDS = [
    Demand("D1", "SEA", "ATL", bandwidth=35, latency_bound=30, priority="high", needs_backup=True),
    Demand("D2", "DEN", "MIA", bandwidth=25, latency_bound=40, priority="medium", needs_backup=False),
    Demand("D3", "CHI", "DAL", bandwidth=30, latency_bound=25, priority="medium", needs_backup=False),
    Demand("D4", "NYC", "LAX", bandwidth=20, latency_bound=45, priority="low", needs_backup=False),
    Demand("D5", "SEA", "NYC", bandwidth=40, latency_bound=35, priority="high", needs_backup=True),
    Demand("D6", "DAL", "CHI", bandwidth=15, latency_bound=25, priority="low", needs_backup=False),
]
_BB8_CURRENT = {
    "D1": [["SEA", "CHI", "ATL"], ["SEA", "DEN", "DAL", "ATL"]],  # primary + 1+1 backup
    "D2": [["DEN", "DAL", "ATL", "MIA"]],
    "D3": [["CHI", "ATL", "DAL"]],
    "D4": [["NYC", "DAL", "LAX"]],
    # primary + 1+1 backup, link-disjoint: the obvious second-shortest path
    # SEA-DEN-CHI-NYC would reuse CHI-NYC and give no real protection.
    "D5": [["SEA", "CHI", "NYC"], ["SEA", "DEN", "DAL", "NYC"]],
    "D6": [["DAL", "ATL", "CHI"]],
}


# --------------------------------------------------------------------------
# Option C -- metro12: 12 nodes / 18 links, the scale of a small SNDlib
# reference network. Too large for one monolithic QAOA circuit on today's
# simulator -- this is the instance that motivates decomposition, and the
# one to quote when arguing about scaling.
# --------------------------------------------------------------------------

_M12_NODES = ["PDX", "SLC", "PHX", "OMA", "MSP", "STL", "NSH", "CLT", "JAX", "WDC", "PHL", "BOS"]
_M12_EDGES = [
    ("PDX", "SLC", 80, 7, 3),
    ("PDX", "MSP", 60, 14, 5),
    ("SLC", "PHX", 60, 6, 3),
    ("SLC", "OMA", 70, 8, 3),
    ("PHX", "STL", 50, 11, 4),
    ("PHX", "JAX", 40, 16, 5),
    ("MSP", "OMA", 80, 4, 2),
    ("MSP", "STL", 60, 6, 2),
    ("OMA", "STL", 70, 5, 2),
    ("STL", "NSH", 60, 4, 2),
    ("STL", "WDC", 50, 9, 3),
    ("NSH", "CLT", 70, 4, 2),
    ("NSH", "JAX", 50, 6, 2),
    ("CLT", "WDC", 80, 4, 2),
    ("CLT", "JAX", 60, 5, 2),
    ("WDC", "PHL", 100, 3, 1),
    ("PHL", "BOS", 90, 4, 1),
    ("MSP", "BOS", 50, 13, 5),
]
_M12_POS = {
    "PDX": (0.00, 1.00), "SLC": (0.20, 0.70), "PHX": (0.20, 0.25),
    "OMA": (0.45, 0.65), "MSP": (0.50, 0.90), "STL": (0.60, 0.55),
    "NSH": (0.68, 0.40), "CLT": (0.80, 0.35), "JAX": (0.85, 0.10),
    "WDC": (0.92, 0.55), "PHL": (0.96, 0.68), "BOS": (1.00, 0.85),
}
_M12_DEMANDS = [
    Demand("C1", "PDX", "BOS", bandwidth=30, latency_bound=40, priority="high", needs_backup=True),
    Demand("C2", "SLC", "WDC", bandwidth=25, latency_bound=35, priority="high", needs_backup=True),
    Demand("C3", "PHX", "PHL", bandwidth=20, latency_bound=40, priority="medium", needs_backup=False),
    Demand("C4", "MSP", "JAX", bandwidth=25, latency_bound=30, priority="medium", needs_backup=False),
    Demand("C5", "OMA", "CLT", bandwidth=30, latency_bound=25, priority="medium", needs_backup=False),
    Demand("C6", "STL", "BOS", bandwidth=20, latency_bound=30, priority="low", needs_backup=False),
    Demand("C7", "NSH", "PDX", bandwidth=15, latency_bound=45, priority="low", needs_backup=False),
    Demand("C8", "JAX", "WDC", bandwidth=25, latency_bound=20, priority="medium", needs_backup=False),
]
_M12_CURRENT = {
    "C1": [["PDX", "MSP", "BOS"], ["PDX", "SLC", "OMA", "STL", "WDC", "PHL", "BOS"]],
    "C2": [["SLC", "OMA", "STL", "WDC"], ["SLC", "PHX", "STL", "NSH", "CLT", "WDC"]],
    "C3": [["PHX", "STL", "WDC", "PHL"]],
    "C4": [["MSP", "STL", "NSH", "JAX"]],
    "C5": [["OMA", "STL", "NSH", "CLT"]],
    "C6": [["STL", "WDC", "PHL", "BOS"]],
    "C7": [["NSH", "STL", "OMA", "SLC", "PDX"]],
    "C8": [["JAX", "CLT", "WDC"]],
}


def mini6() -> Instance:
    g, pos = _build(_MINI6_NODES, _MINI6_EDGES, _MINI6_POS)
    return Instance(
        name="mini6",
        headline="6 nodes / 8 links / 3 demands -- debugging scale, plenty of qubit headroom",
        graph=g, demands=list(_MINI6_DEMANDS),
        current_routing={k: [list(p) for p in v] for k, v in _MINI6_CURRENT.items()},
        pos=pos,
    )


def backbone8() -> Instance:
    g, pos = _build(_BB8_NODES, _BB8_EDGES, _BB8_POS)
    return Instance(
        name="backbone8",
        headline="8 PoPs / 12 links / 6 demands -- fits a 28-qubit simulator after pruning",
        graph=g, demands=list(_BB8_DEMANDS),
        current_routing={k: [list(p) for p in v] for k, v in _BB8_CURRENT.items()},
        pos=pos,
    )


def metro12() -> Instance:
    g, pos = _build(_M12_NODES, _M12_EDGES, _M12_POS)
    return Instance(
        name="metro12",
        headline="12 nodes / 18 links / 8 demands -- SNDlib reference scale, needs decomposition",
        graph=g, demands=list(_M12_DEMANDS),
        current_routing={k: [list(p) for p in v] for k, v in _M12_CURRENT.items()},
        pos=pos,
    )


ALL_INSTANCES = {"mini6": mini6, "backbone8": backbone8, "metro12": metro12}


def get(name: str) -> Instance:
    return ALL_INSTANCES[name]()


def scaled_copy(inst: Instance, divisor: int) -> Instance:
    """Divide every capacity and bandwidth by the same integer.

    Exact, not an approximation: scaling both sides of every capacity
    constraint by one constant leaves the set of feasible routings
    unchanged, and utilization ratios are scale-invariant. It exists purely
    to shrink the integers the quantum encoding has to represent.
    """
    g = nx.Graph()
    g.add_nodes_from(inst.graph.nodes)
    for u, v, data in inst.graph.edges(data=True):
        g.add_edge(
            u, v,
            capacity=round(data["capacity"] / divisor),
            delay=data["delay"],
            cost=data["cost"],
        )
    demands = [replace(d, bandwidth=round(d.bandwidth / divisor)) for d in inst.demands]
    return Instance(
        name=f"{inst.name}/{divisor}",
        headline=inst.headline,
        graph=g, demands=demands,
        current_routing={k: [list(p) for p in v] for k, v in inst.current_routing.items()},
        pos=dict(inst.pos), unit_gbps=inst.unit_gbps * divisor,
    )


def validate(inst: Instance) -> list[str]:
    """Sanity-check an instance against the brief's own rules. Returns
    human-readable problems, empty list if the instance is coherent."""
    problems = []
    ids = {d.id for d in inst.demands}
    for demand_id, paths in inst.current_routing.items():
        if demand_id not in ids:
            problems.append(f"current_routing has unknown demand {demand_id}")
            continue
        d = next(x for x in inst.demands if x.id == demand_id)
        if len(paths) != d.paths_required:
            problems.append(
                f"{demand_id}: current routing has {len(paths)} path(s), "
                f"but priority={d.priority}/backup={d.needs_backup} requires {d.paths_required}"
            )
        for nodes in paths:
            for u, v in zip(nodes[:-1], nodes[1:]):
                if not inst.graph.has_edge(u, v):
                    problems.append(f"{demand_id}: current path uses missing link {u}-{v}")
            if nodes[0] != d.src or nodes[-1] != d.dst:
                problems.append(f"{demand_id}: current path {nodes} does not run {d.src}->{d.dst}")
            delay = inst.path_delay(nodes)
            if delay > d.latency_bound:
                problems.append(
                    f"{demand_id}: current path {'-'.join(nodes)} takes {delay}ms, "
                    f"over its {d.latency_bound}ms bound"
                )
        if d.needs_backup and len(paths) == 2:
            e0 = {frozenset(e) for e in zip(paths[0][:-1], paths[0][1:])}
            e1 = {frozenset(e) for e in zip(paths[1][:-1], paths[1][1:])}
            shared = e0 & e1
            if shared:
                names = ", ".join("-".join(sorted(s)) for s in shared)
                problems.append(
                    f"{demand_id}: 1+1 backup is not link-disjoint, shares {names}"
                )
    for d in inst.demands:
        if d.id not in inst.current_routing:
            problems.append(f"{d.id} has no current routing")
    return problems
