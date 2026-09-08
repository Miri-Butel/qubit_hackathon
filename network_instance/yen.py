"""
Yen's K-shortest loopless paths, plus candidate-route generation for the
routing problem.

Why this matters here: the quantum formulation does not search over all
possible routes, it picks one route per demand out of a small precomputed
candidate set. Yen's algorithm is what builds that set. Garbage candidates
mean the optimizer cannot find a good answer no matter how good the solver
is, so this step decides the ceiling on solution quality.

The brief gives each link three attributes -- capacity, delay and
operational cost -- and those pull in different directions. A route that is
fastest is often not cheapest, and neither is necessarily the one with room
to spare. So we run Yen's once per metric and take the union, which gives
the optimizer a candidate set that contains the best answer under any of the
competing objectives rather than just one of them.
"""

from dataclasses import dataclass, field

import networkx as nx


# --------------------------------------------------------------------------
# Yen's algorithm
# --------------------------------------------------------------------------

def _path_weight(g: nx.Graph, nodes: list[str], weight) -> float:
    return sum(weight(u, v, g[u][v]) for u, v in zip(nodes[:-1], nodes[1:]))


def yen_k_shortest(
    g: nx.Graph, source: str, target: str, k: int = 5, weight=None
) -> list[tuple[list[str], float]]:
    """K shortest loopless paths from source to target.

    Yen's algorithm. The first path is just the shortest path. Each
    subsequent path is found by taking each already-accepted path, treating
    every node along it as a "spur node", banning the links that would
    retrace an existing path from that point, and finding the best detour
    from there. The cheapest such detour becomes the next path.

    Returns (nodes, total_weight) pairs, ascending by weight. Fewer than `k`
    come back if the graph simply does not contain that many loopless paths.
    """
    if weight is None:
        weight = lambda u, v, d: 1.0
    if source == target or source not in g or target not in g:
        return []

    try:
        first = nx.shortest_path(g, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return []

    accepted: list[list[str]] = [first]
    candidates: list[tuple[float, list[str]]] = []

    while len(accepted) < k:
        previous = accepted[-1]

        for i in range(len(previous) - 1):
            spur_node = previous[i]
            root_path = previous[: i + 1]

            work = g.copy()

            # Ban the next hop of every accepted path sharing this root, so
            # we are forced to find a genuinely different continuation.
            for p in accepted:
                if len(p) > i and p[: i + 1] == root_path:
                    if work.has_edge(p[i], p[i + 1]):
                        work.remove_edge(p[i], p[i + 1])

            # Drop the root's interior nodes so the detour cannot loop back.
            for node in root_path[:-1]:
                if node in work:
                    work.remove_node(node)

            try:
                spur_path = nx.shortest_path(work, spur_node, target, weight=weight)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            total = root_path[:-1] + spur_path
            if total in accepted or any(total == c[1] for c in candidates):
                continue
            candidates.append((_path_weight(g, total, weight), total))

        if not candidates:
            break
        candidates.sort(key=lambda c: (c[0], len(c[1])))
        accepted.append(candidates.pop(0)[1])

    return [(p, _path_weight(g, p, weight)) for p in accepted]


# --------------------------------------------------------------------------
# The competing metrics, one per constraint the brief names
# --------------------------------------------------------------------------

def delay_weight(u, v, d):
    """Latency constraint: minimize end-to-end delay."""
    return d["delay"]


def cost_weight(u, v, d):
    """Operational cost: minimize opex per Gbps carried."""
    return d["cost"]


def hop_weight(u, v, d):
    """Resource usage: fewest links touched, the classic routing default."""
    return 1.0


def make_spare_capacity_weight(load: dict[frozenset, float]):
    """Capacity constraint: prefer links that still have room.

    This is the minimum-interference idea from the brief's own reading list
    -- route where you do least damage to everyone else's remaining
    headroom. A link running near its limit is expensive; a saturated one is
    effectively impassable.
    """
    def weight(u, v, d):
        used = load.get(frozenset((u, v)), 0.0)
        spare = d["capacity"] - used
        if spare <= 0:
            return 1e6
        return d["capacity"] / spare
    return weight


def metrics_for(inst, load=None):
    """The four routing metrics, named for the constraint each serves."""
    load = inst.link_load(inst.current_routing) if load is None else load
    return {
        "delay": delay_weight,
        "cost": cost_weight,
        "hops": hop_weight,
        "spare": make_spare_capacity_weight(load),
    }


# --------------------------------------------------------------------------
# Candidate routes per demand
# --------------------------------------------------------------------------

@dataclass
class Route:
    demand_id: str
    nodes: list[str]
    delay: float
    cost: float          # opex per Gbps carried, summed over links
    hops: int
    min_spare: float     # Gbps of headroom on this route's tightest link today
    within_latency: bool
    found_by: set[str] = field(default_factory=set)   # which metrics ranked it
    best_rank: dict[str, int] = field(default_factory=dict)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(zip(self.nodes[:-1], self.nodes[1:]))

    @property
    def label(self) -> str:
        return "-".join(self.nodes)

    def is_link_disjoint_from(self, other: "Route") -> bool:
        return not ({frozenset(e) for e in self.edges} & {frozenset(e) for e in other.edges})


def describe_route(inst, demand, nodes: list[str], load=None) -> Route:
    load = inst.link_load(inst.current_routing) if load is None else load
    g = inst.graph
    delay = sum(g[u][v]["delay"] for u, v in zip(nodes[:-1], nodes[1:]))
    cost = sum(g[u][v]["cost"] for u, v in zip(nodes[:-1], nodes[1:]))
    spare = min(
        (g[u][v]["capacity"] - load.get(frozenset((u, v)), 0.0)
         for u, v in zip(nodes[:-1], nodes[1:])),
        default=0.0,
    )
    return Route(
        demand_id=demand.id, nodes=list(nodes), delay=delay, cost=cost,
        hops=len(nodes) - 1, min_spare=spare,
        within_latency=delay <= demand.latency_bound,
    )


def candidate_routes(inst, demand, k: int = 5, load=None) -> list[Route]:
    """Top-k routes under every metric, merged into one candidate set.

    A route found by more than one metric is kept once, tagged with each
    metric that ranked it and where it placed. Routes that break the demand's
    latency bound are kept but flagged, because "no legal route exists here"
    is itself a finding worth seeing rather than silently hiding.
    """
    load = inst.link_load(inst.current_routing) if load is None else load
    metrics = metrics_for(inst, load)

    merged: dict[tuple[str, ...], Route] = {}
    for metric_name, weight in metrics.items():
        for rank, (nodes, _) in enumerate(
            yen_k_shortest(inst.graph, demand.src, demand.dst, k=k, weight=weight), start=1
        ):
            key = tuple(nodes)
            if key not in merged:
                merged[key] = describe_route(inst, demand, nodes, load)
            merged[key].found_by.add(metric_name)
            merged[key].best_rank[metric_name] = rank

    return sorted(merged.values(), key=lambda r: (not r.within_latency, r.delay))


def ranked_by_metric(inst, demand, k: int = 5, load=None) -> dict[str, list[Route]]:
    """The top-k list each metric produces, kept separate so the
    disagreement between metrics is visible."""
    load = inst.link_load(inst.current_routing) if load is None else load
    out = {}
    for metric_name, weight in metrics_for(inst, load).items():
        out[metric_name] = [
            describe_route(inst, demand, nodes, load)
            for nodes, _ in yen_k_shortest(inst.graph, demand.src, demand.dst, k=k, weight=weight)
        ]
    return out


def shared_links(a: "Route", b: "Route") -> int:
    return len({frozenset(e) for e in a.edges} & {frozenset(e) for e in b.edges})


def select_candidates(inst, demand, pool: list["Route"], n: int = 2) -> list["Route"]:
    """Cut the Yen candidate pool down to the `n` routes the optimizer gets.

    This is the step that decides the qubit budget: every candidate kept is
    one more qubit in the QUBO, so the pool of ~8 distinct routes Yen's finds
    has to become a shortlist. Which `n` we keep sets the ceiling on solution
    quality, so the rule is not "take the n fastest" -- two near-identical
    routes down the same corridor give the optimizer no real decision and
    waste a qubit.

    For a 1+1 protected demand, the two must be link-disjoint, since that is
    what protection means; we take the best legal disjoint pair.

    Otherwise we take the best route by delay, then repeatedly add whichever
    remaining route shares the fewest links with everything already chosen,
    breaking ties by delay. That buys genuine diversity: alternatives that
    fail over to different parts of the network.
    """
    legal = [r for r in pool if r.within_latency]
    if not legal:
        return sorted(pool, key=lambda r: r.delay)[:n]

    if demand.needs_backup and n >= 2:
        pairs = disjoint_pairs(legal, top=1)
        if pairs:
            chosen = list(pairs[0])
            remaining = [r for r in legal if r not in chosen]
            while len(chosen) < n and remaining:
                nxt = min(remaining, key=lambda r: (sum(shared_links(r, c) for c in chosen), r.delay))
                chosen.append(nxt)
                remaining.remove(nxt)
            return chosen

    chosen = [min(legal, key=lambda r: r.delay)]
    remaining = [r for r in legal if r is not chosen[0]]
    while len(chosen) < n and remaining:
        nxt = min(remaining, key=lambda r: (sum(shared_links(r, c) for c in chosen), r.delay))
        chosen.append(nxt)
        remaining.remove(nxt)
    return chosen


def build_candidate_set(inst, k: int = 5, n=2, load=None) -> dict[str, list["Route"]]:
    """End to end: Yen's top-k under every metric, merged, then shortlisted to
    `n` routes per demand. This dict is exactly what the QUBO encodes.

    `n` may be an int for a uniform shortlist, or a dict keyed by demand id
    for an uneven one. Uneven is often the better spend of a fixed qubit
    budget: give the biggest, most contended demands a third route to choose
    from, and leave the small ones with two, rather than paying for a third
    option on a demand that was never going to be the problem.
    """
    load = inst.link_load(inst.current_routing) if load is None else load
    out = {}
    for d in inst.demands:
        pool = candidate_routes(inst, d, k=k, load=load)
        want = n.get(d.id, 2) if isinstance(n, dict) else n
        out[d.id] = select_candidates(inst, d, pool, n=want)
    return out


def budgeted_candidate_set(
    inst, k: int = 5, base: int = 2, extra_for: int = 0, load=None
) -> dict[str, list["Route"]]:
    """Shortlist with `base` routes each, then hand a spare route to the
    `extra_for` largest demands -- a simple way to spend a fixed qubit
    budget where it buys the most decision."""
    ranked = sorted(inst.demands, key=lambda d: -d.bandwidth)
    n = {d.id: base + (1 if i < extra_for else 0) for i, d in enumerate(ranked)}
    return build_candidate_set(inst, k=k, n=n, load=load)


def disjoint_pairs(routes: list[Route], top: int = 5) -> list[tuple[Route, Route]]:
    """Best link-disjoint route pairs, for demands needing 1+1 protection.

    A protected demand needs two paths that share no link, so one cut cannot
    take down both. Ranked by combined delay, and only pairs where both legs
    meet the latency bound are returned -- during a failover the backup is
    carrying the service, so it has to be within SLA too.
    """
    legal = [r for r in routes if r.within_latency]
    pairs = []
    for i, a in enumerate(legal):
        for b in legal[i + 1:]:
            if a.is_link_disjoint_from(b):
                pairs.append((a, b))
    pairs.sort(key=lambda ab: (ab[0].delay + ab[1].delay))
    return pairs[:top]
