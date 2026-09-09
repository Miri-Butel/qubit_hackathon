"""
The real AT&T North America MPLS backbone, loaded from the Internet Topology Zoo.

Source: `data/AttMpls.gml`, from the Internet Topology Zoo (Knight, Nguyen,
Falkner, Bowden and Roughan, "The Internet Topology Zoo", IEEE JSAC 2011).
The Zoo built this record from AT&T's own published network map,
`Domestic_OC-768_Network.pdf`, for the 2007-2008 build-out. 25 PoPs, 56
links, with AT&T's real PoP codes (NY54, DLLS, CHCG, LA03 ...) and real
geographic coordinates.

BE PRECISE ABOUT WHAT IS REAL HERE, because the distinction matters when
presenting results:

  REAL, from the published AT&T map:
    - which PoPs exist and where they physically are (lat/long)
    - which PoPs are directly connected by a backbone link

  DERIVED from that real data:
    - link delay. Great-circle distance between the two real PoPs, times a
      1.5 detour factor because fibre does not run straight, times 5us/km
      (light in glass at ~200,000 km/s). This reproduces well-known
      real-world figures: it puts NY-to-LA near 30ms one way, which is what
      that path actually measures.

  SYNTHESIZED by us, with the method stated so it can be challenged:
    - link capacity. The source document is AT&T's OC-768 network map, so
      capacity is issued in OC-768 units of ~40 Gbps, with more wavelengths
      lit between higher-degree hubs than out to leaf PoPs.
    - operational cost per Gbps, taken proportional to distance, since
      long-haul transport is what actually costs money.
    - the traffic matrix. A standard gravity model: traffic between two
      PoPs is proportional to the product of their metro populations. This
      is the usual way traffic matrices are synthesized in the traffic-
      engineering literature when real customer traffic is not public,
      which it never is.

Nothing here claims to be AT&T's real traffic or real provisioned capacity.
Those are trade secrets. The topology and geography, which drive the shape
of the optimization problem, are real.
"""

import math
import re
from dataclasses import replace
from pathlib import Path

import networkx as nx

from topology import Demand
from topologies import Instance


DATA_FILE = Path(__file__).parent / "data" / "AttMpls.gml"

OC768_GBPS = 40  # OC-768 is 39.813 Gbps; round to 40 for readability
FIBRE_DETOUR = 1.5  # fibre route length vs great-circle distance
MS_PER_KM = 0.005  # light in glass, ~200,000 km/s -> 5 microseconds per km

# Human-readable names for AT&T's PoP codes.
POP_NAMES = {
    "NY54": "New York", "CMBR": "Cambridge/Boston", "CHCG": "Chicago",
    "CLEV": "Cleveland", "RLGH": "Raleigh", "ATLN": "Atlanta",
    "PHLA": "Philadelphia", "WASH": "Washington DC", "NSVL": "Nashville",
    "STLS": "St. Louis", "NWOR": "New Orleans", "HSTN": "Houston",
    "SNAN": "San Antonio", "DLLS": "Dallas", "ORLD": "Orlando",
    "DNVR": "Denver", "KSCY": "Kansas City", "SNFN": "San Francisco",
    "SCRM": "Sacramento", "PTLD": "Portland", "STTL": "Seattle",
    "SLKC": "Salt Lake City", "LA03": "Los Angeles", "SNDG": "San Diego",
    "PHNX": "Phoenix",
}

# Approximate metro-area populations in millions, used only to shape the
# synthetic gravity-model traffic matrix. Rounded public figures.
METRO_POP_M = {
    "NY54": 20.1, "CMBR": 4.9, "CHCG": 9.6, "CLEV": 2.1, "RLGH": 1.4,
    "ATLN": 6.1, "PHLA": 6.2, "WASH": 6.3, "NSVL": 2.0, "STLS": 2.8,
    "NWOR": 1.3, "HSTN": 7.1, "SNAN": 2.6, "DLLS": 7.6, "ORLD": 2.7,
    "DNVR": 3.0, "KSCY": 2.2, "SNFN": 4.7, "SCRM": 2.4, "PTLD": 2.5,
    "STTL": 4.0, "SLKC": 1.3, "LA03": 13.2, "SNDG": 3.3, "PHNX": 5.0,
}

# Regional slices of the real backbone. Useful because the full 25-PoP
# network is far too large for one monolithic QUBO, but a real region of it
# is still real AT&T topology, not a toy graph we invented.
REGIONS = {
    # The working instance: 10 real PoPs across AT&T's east and midwest.
    # Picked by scanning candidate 10-node slices for path diversity -- every
    # one of its 45 node pairs has at least 5 distinct loopless routes, so
    # Yen's always returns a full candidate set and the optimizer always has
    # a real choice to make.
    "east10": ["CHCG", "CLEV", "NY54", "CMBR", "PHLA", "WASH", "RLGH", "ATLN", "NSVL", "STLS"],
    # Same 10 PoPs, loaded with many more demands. See REGION_DEFAULTS.
    "east10_dense": ["CHCG", "CLEV", "NY54", "CMBR", "PHLA", "WASH", "RLGH", "ATLN", "NSVL", "STLS"],
    # Same 10 PoPs and demand count as east10_dense at a routable traffic
    # level -- the instance the QAOA results are reported on.
    "east10_tuned": ["CHCG", "CLEV", "NY54", "CMBR", "PHLA", "WASH", "RLGH", "ATLN", "NSVL", "STLS"],
    "west": ["STTL", "PTLD", "SCRM", "SNFN", "SLKC", "DNVR", "LA03", "SNDG", "PHNX"],
    "northeast": ["NY54", "CMBR", "PHLA", "WASH", "CLEV", "CHCG", "RLGH"],
    "south": ["DLLS", "HSTN", "SNAN", "NWOR", "ATLN", "ORLD", "NSVL", "STLS", "KSCY"],
}

# How many demands, and how much traffic, per region. Chosen by sweeping the
# traffic scale until today's shortest-path routing is genuinely congested --
# several links over capacity and others close to it. An instance nobody
# struggles to route is not a test of anything.
REGION_DEFAULTS = {
    # max_per_pop=1 so no city appears in more than one demand. With 2, New
    # York's metro is heavy enough to claim half the traffic matrix, and the
    # instance stops being a network problem and becomes a New York problem.
    # Readable demo instance. 5 demands is the most this region supports at
    # one demand per PoP, since five demands consume all ten PoPs as
    # endpoints. Good for figures and for reading route tables by eye, but
    # too sparse to be a hard optimization: the demands barely share links,
    # so today's shortest-path routing is already close to optimal.
    "east10": {"top_n": 5, "scale": 2.5, "max_per_pop": 1},
    # Handoff instance for the QAOA pipeline. Deliberately dense: 14 demands
    # over the same 10 PoPs, up to 4 demands per PoP, so their shortest paths
    # collide and rerouting genuinely pays. Exactly 28 candidate paths, which
    # is 28 qubits in routing_qaoa's encoding. Verified non-degenerate: an
    # exhaustive search over all 16384 assignments beats today's routing,
    # cutting total congestion sum(u^2) by about 11%.
    "east10_dense": {"top_n": 14, "scale": 4.0, "max_per_pop": 4, "fill": 0.9},
    # The instance the QAOA results are reported on. Same 10 PoPs and 14
    # demands as east10_dense, but `fill` drops from 0.9 to 0.3.
    #
    # That one number decides whether the problem is solvable at all. `fill`
    # caps each demand at that fraction of the widest bottleneck available to
    # it, so at 0.9 any two demands sharing a link overload it no matter how
    # they are routed: an exhaustive search over all 16384 assignments finds
    # *no* assignment without capacity violations, and the best one trades 9
    # violations for 6 while making the peak worse. Congestion that no routing
    # can relieve is a capacity-planning result, not an optimization problem.
    #
    # At 0.3 the demands compete instead of individually overflowing. Today's
    # shortest-path routing drives one link to 105% of capacity; a clean
    # assignment exists and peaks at 90%; and only 112 of the 16384 assignments
    # are clean at all (0.7%), so finding one is real work. Same 28 qubits.
    #
    # Measured limit, worth knowing before promising a quantum result here:
    # with D demands choosing between 2 routes each, the one-hot feasible
    # subspace is 2^D of 2^(2D) states -- 0.006% at D=14. A depth-1 X-mixer
    # QAOA run samples no feasible state at all (see docs/qaoa_east10_tuned.json),
    # even though its CVaR objective drops from 47.1 to 11.8. Reaching this
    # subspace needs a feasibility-preserving XY mixer or a binary encoding;
    # dropping to 8 demands makes it reachable but leaves today's routing
    # already optimal, so there is nothing left to solve.
    "east10_tuned": {"top_n": 14, "scale": 2.0, "max_per_pop": 4, "fill": 0.3},
    "west": {"top_n": 6, "scale": 1.5},
    "northeast": {"top_n": 5, "scale": 2.0},
    "south": {"top_n": 6, "scale": 1.5},
    None: {"top_n": 10, "scale": 1.5},
}


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _parse_gml(path: Path):
    """Hand-rolled parser: the Zoo file has one duplicated edge (LA03-PHNX
    appears twice), which networkx's strict GML reader rejects outright."""
    txt = path.read_text()
    nodes = {}
    for blk in re.findall(r"node \[(.*?)\]", txt, re.S):
        nid = int(re.search(r"id (\d+)", blk).group(1))
        label = re.search(r'label "(.*?)"', blk).group(1)
        lat = re.search(r"Latitude ([-\d.]+)", blk)
        lon = re.search(r"Longitude ([-\d.]+)", blk)
        nodes[nid] = {
            "label": label,
            "lat": float(lat.group(1)) if lat else None,
            "lon": float(lon.group(1)) if lon else None,
        }
    edges = set()
    for blk in re.findall(r"edge \[(.*?)\]", txt, re.S):
        s = int(re.search(r"source (\d+)", blk).group(1))
        t = int(re.search(r"target (\d+)", blk).group(1))
        edges.add(tuple(sorted((s, t))))  # dedupe the repeated parallel link
    return nodes, sorted(edges)


def load_graph(nodes_subset: list[str] | None = None) -> tuple[nx.Graph, dict]:
    """Real topology and geography; capacity/delay/cost derived as documented."""
    raw_nodes, raw_edges = _parse_gml(DATA_FILE)
    id_to_label = {nid: d["label"] for nid, d in raw_nodes.items()}

    g = nx.Graph()
    for nid, d in raw_nodes.items():
        label = d["label"]
        if nodes_subset and label not in nodes_subset:
            continue
        g.add_node(label, lat=d["lat"], lon=d["lon"],
                   city=POP_NAMES.get(label, label), pop_m=METRO_POP_M.get(label, 1.0))

    raw_degree = {label: 0 for label in id_to_label.values()}
    for s, t in raw_edges:
        raw_degree[id_to_label[s]] += 1
        raw_degree[id_to_label[t]] += 1

    for s, t in raw_edges:
        u, v = id_to_label[s], id_to_label[t]
        if u not in g or v not in g:
            continue
        km = haversine_km(g.nodes[u]["lat"], g.nodes[u]["lon"],
                          g.nodes[v]["lat"], g.nodes[v]["lon"])
        delay = round(km * FIBRE_DETOUR * MS_PER_KM, 1)
        # More wavelengths lit between the big hubs than out to leaf PoPs.
        waves = max(1, min(6, round((raw_degree[u] + raw_degree[v]) / 4)))
        g.add_edge(u, v,
                   capacity=waves * OC768_GBPS,
                   delay=max(delay, 0.5),
                   cost=max(1, round(km / 400)),
                   km=round(km))

    pos = {n: (d["lon"], d["lat"]) for n, d in g.nodes(data=True)}
    return g, pos


def gravity_demands(
    g: nx.Graph, top_n: int = 8, scale: float = 1.0, seed_priority: bool = True,
    max_per_pop: int = 2,
) -> list[Demand]:
    """Synthesize a demand set with a gravity model: traffic between two PoPs
    grows with the product of their metro populations.

    `max_per_pop` caps how many demands any single PoP may appear in. Without
    it the model degenerates: New York's metro is so much larger than
    everyone else's that it wins essentially every top-weighted pair, and the
    "network-wide" instance turns into a star centred on one node. Capping
    keeps the traffic spread over the real topology, which is both more
    realistic and a far more interesting routing problem.

    Bandwidth is scaled so the largest flows are a meaningful fraction of a
    single OC-768, which is what makes the instance contended rather than
    trivially satisfiable. Latency bounds are set a little above each pair's
    best achievable delay, so they bind on the detour paths but not on the
    direct one. The biggest flows get high priority and 1+1 protection,
    which is how carriers actually treat their largest customers.
    """
    pairs = []
    nodes = list(g.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            weight = g.nodes[u]["pop_m"] * g.nodes[v]["pop_m"]
            pairs.append((weight, u, v))
    pairs.sort(reverse=True)

    used: dict[str, int] = {}
    chosen = []
    for weight, u, v in pairs:
        if len(chosen) >= top_n:
            break
        if used.get(u, 0) >= max_per_pop or used.get(v, 0) >= max_per_pop:
            continue
        chosen.append((weight, u, v))
        used[u] = used.get(u, 0) + 1
        used[v] = used.get(v, 0) + 1
    if not chosen:
        return []

    heaviest = chosen[0][0]
    demands = []
    for idx, (weight, u, v) in enumerate(chosen):
        # Bandwidth follows the square root of the gravity weight, not the
        # weight itself. Raw population products span orders of magnitude, so
        # a linear map pins everything below the top pair at the floor and
        # leaves the network uncontended; the square root compresses that
        # spread into a range where many demands genuinely compete.
        bw = max(10, round(scale * 60 * math.sqrt(weight / heaviest) / 5) * 5)
        rank = idx / max(1, len(chosen) - 1)
        if seed_priority and rank < 0.25:
            priority, backup = "high", True
        elif rank < 0.65:
            priority, backup = "medium", False
        else:
            priority, backup = "low", False
        demands.append(Demand(
            id=f"T{idx + 1}", src=u, dst=v, bandwidth=bw,
            latency_bound=0,  # filled in by `set_latency_bounds` once routes are known
            priority=priority, needs_backup=backup,
        ))
    return demands


def set_latency_bounds(
    inst_graph: nx.Graph, demands: list[Demand],
    routing: dict[str, list[list[str]]], headroom: float = 1.6,
    min_slack_ms: float = 5.0,
) -> list[Demand]:
    """Set each demand's latency bound from the routes it actually runs today.

    The bound has to cover every path the demand currently uses, including
    the longer link-disjoint backup of a 1+1 protected service -- deriving it
    from the primary alone declares today's own configuration illegal, which
    is nonsense.

    Bound = worst path in use today, widened by whichever is more generous:
    a proportional `headroom`, or a flat `min_slack_ms`. The flat floor
    matters. A purely proportional bound on a 1.5ms metro hop grants under a
    millisecond of slack, which makes every alternative route illegal and
    leaves the optimizer with a single legal option and nothing to decide.
    Real SLAs are quoted with absolute margins for exactly this reason.
    """
    out = []
    for d in demands:
        paths = routing.get(d.id, [])
        worst = max(
            (sum(inst_graph[a][b]["delay"] for a, b in zip(p[:-1], p[1:])) for p in paths),
            default=0.0,
        )
        bound = max(worst * headroom, worst + min_slack_ms)
        out.append(replace(d, latency_bound=max(1, round(bound))))
    return out


def cap_demands_to_capacity(
    g: nx.Graph, demands: list[Demand], fill: float = 0.7
) -> list[Demand]:
    """Clamp each demand so at least one path through the network can carry it.

    Without this the gravity model happily emits a demand bigger than the
    bottleneck of every route available to it. That looks like heavy
    congestion but is actually a degenerate instance: no assignment can
    relieve the overload, so the optimum is "change nothing" and the problem
    teaches nothing. Real contention has to come from demands *competing*
    for shared links, not from one demand that fits nowhere.

    Each demand is capped at `fill` of the widest bottleneck available to it,
    where a path's bottleneck is its narrowest link.
    """
    out = []
    for d in demands:
        widest = 0.0
        for nodes in nx.shortest_simple_paths(g, d.src, d.dst, weight="delay"):
            bottleneck = min(g[u][v]["capacity"] for u, v in zip(nodes[:-1], nodes[1:]))
            widest = max(widest, bottleneck)
            if widest >= d.bandwidth:
                break
        cap = max(5, int(widest * fill))
        out.append(replace(d, bandwidth=min(d.bandwidth, cap)) if d.bandwidth > cap else d)
    return out


def shortest_path_routing(g: nx.Graph, demands: list[Demand]) -> dict[str, list[list[str]]]:
    """The routing the network runs today: lowest-delay path for every demand,
    plus a link-disjoint backup for each protected one, found by removing the
    primary's links and re-running the shortest path (Suurballe's idea in its
    simplest form)."""
    routing = {}
    for d in demands:
        primary = nx.shortest_path(g, d.src, d.dst, weight="delay")
        paths = [primary]
        if d.needs_backup:
            h = g.copy()
            h.remove_edges_from(list(zip(primary[:-1], primary[1:])))
            try:
                paths.append(nx.shortest_path(h, d.src, d.dst, weight="delay"))
            except nx.NetworkXNoPath:
                pass  # no disjoint backup exists; validate() will flag it
        routing[d.id] = paths
    return routing


def att_backbone(
    region: str | None = None, top_n: int | None = None, scale: float | None = None,
    latency_headroom: float = 1.6,
) -> Instance:
    """Build an `Instance` from the real AT&T backbone, optionally a region of it."""
    defaults = REGION_DEFAULTS.get(region, {"top_n": 8, "scale": 1.5})
    top_n = defaults["top_n"] if top_n is None else top_n
    scale = defaults["scale"] if scale is None else scale
    max_per_pop = defaults.get("max_per_pop", 2)
    subset = REGIONS[region] if region else None
    g, pos = load_graph(subset)
    demands = gravity_demands(g, top_n=top_n, scale=scale, max_per_pop=max_per_pop)
    demands = cap_demands_to_capacity(g, demands, fill=defaults.get("fill", 0.7))
    routing = shortest_path_routing(g, demands)
    demands = set_latency_bounds(g, demands, routing, headroom=latency_headroom)
    where = f"{region} region" if region else "North America"
    return Instance(
        name=f"att-{region}" if region else "att-full",
        headline=(f"REAL AT&T MPLS backbone, {where}: {g.number_of_nodes()} PoPs / "
                  f"{g.number_of_edges()} links / {len(demands)} demands"),
        graph=g, demands=demands, current_routing=routing, pos=pos,
    )
