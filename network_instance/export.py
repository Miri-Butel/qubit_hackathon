"""
Bridge from this package (pipeline steps 1-2: build the network, generate
candidate routes) to `routing_qaoa` (steps 3-4: QUBO encoding and QAOA).

`routing_qaoa.RoutingInstance` wants three things -- directed links carrying
capacity and latency, demands carrying bandwidth, and a tuple of candidate
paths per demand expressed as chains of (u, v) link keys. This module emits
exactly that, plus the `current_routing` index map their switch-penalty term
needs.

Four things are worth knowing about the translation, because the two models
do not line up perfectly:

1. LINKS BECOME DIRECTED. Our graph is undirected; theirs keys links by
   (u, v). Every undirected link is emitted twice, once per direction, with
   the same capacity. That matches real full-duplex fibre, where each
   direction has its own capacity and traffic flowing opposite ways does not
   contend. It does mean the two directions are independent capacity
   constraints, which is correct but worth saying out loud.

2. OPERATIONAL COST IS NOT CARRIED OVER. Their `Link` has capacity and
   latency only. Cost still does real work upstream -- it is one of the four
   metrics Yen's ranks routes under, so cheap routes reach the shortlist --
   but once handed over, the QUBO cannot price it. Adding a `cost` field to
   their `Link` and a term to the Hamiltonian is the natural extension.

3. PRIORITY, LATENCY BOUNDS AND 1+1 PROTECTION ARE ENFORCED HERE, NOT THERE.
   Their model selects exactly one path per demand, with no notion of an SLA
   bound or a protected demand needing two disjoint paths. So we enforce the
   latency bound by only ever exporting routes that satisfy it, and priority
   by ranking. Protection cannot be expressed downstream at all; a protected
   demand is exported as an ordinary one, and `protected_demands()` reports
   which ones lost that guarantee.

4. TODAY'S ROUTE IS ALWAYS INCLUDED as a candidate, and `current_routing`
   gives its index. Their `H_switch` term penalizes every path that is not
   the current one, so if today's route were missing from the shortlist the
   penalty would be measured against a route the optimizer cannot pick, and
   the "minimize unnecessary route changes" metric would be meaningless.
"""

import json
from collections.abc import Mapping
from pathlib import Path

import yen

# Upstream tags a demand's service class as a string; `Demand.priority`
# downstream is a latency multiplier, so the classes need numeric weights.
# High-priority traffic is worth 3x a best-effort demand's latency, which is
# what makes premium demands win a contested link in the QUBO.
PRIORITY_WEIGHTS = {"high": 3.0, "medium": 1.5, "low": 1.0}


def priority_weight(value: str | float) -> float:
    """Map a service-class tag to the numeric priority `routing_qaoa` wants."""
    if isinstance(value, (int, float)):
        return float(value)
    return PRIORITY_WEIGHTS.get(str(value).lower(), 1.0)


def _directed_links(inst) -> list[dict]:
    links = []
    for u, v, data in inst.graph.edges(data=True):
        for a, b in ((u, v), (v, u)):
            links.append({
                "u": a, "v": b,
                "capacity": float(data["capacity"]),
                "latency": float(data["delay"]),
                # not consumed by routing_qaoa today; carried so the QUBO can
                # price operational cost later without regenerating anything
                "cost": float(data["cost"]),
            })
    return links


def _path_link_keys(nodes: list[str]) -> list[list[str]]:
    return [[u, v] for u, v in zip(nodes[:-1], nodes[1:])]


def build_export(inst, candidates: dict) -> dict:
    """Serialize an instance plus its candidate routes into a plain dict.

    Today's route is inserted as candidate 0 for every demand if the
    shortlist does not already contain it, so `current_routing` always points
    at a real candidate.
    """
    demands, paths, current = [], {}, {}

    for d in inst.demands:
        routes = list(candidates[d.id])

        today = inst.current_routing.get(d.id, [])
        primary = today[0] if today else None
        if primary is not None:
            idx = next((i for i, r in enumerate(routes) if r.nodes == primary), None)
            if idx is None:
                routes.insert(0, yen.describe_route(inst, d, primary))
                idx = 0
            current[d.id] = idx

        demands.append({
            "name": d.id,
            "source": d.src,
            "target": d.dst,
            "bandwidth": float(d.bandwidth),
            # context their model has no field for, kept so nothing is lost
            "latency_bound": float(d.latency_bound),
            "priority": d.priority,
            "needs_backup": bool(d.needs_backup),
        })
        paths[d.id] = [
            {
                "nodes": r.nodes,
                "links": _path_link_keys(r.nodes),
                "delay": round(r.delay, 3),
                "cost": round(r.cost, 3),
                "hops": r.hops,
                "within_latency": r.within_latency,
                "found_by": sorted(r.found_by),
            }
            for r in routes
        ]

    return {
        "name": inst.name,
        "description": inst.headline,
        "provenance": {
            "topology": "Internet Topology Zoo, AttMpls.gml (AT&T North America "
                        "MPLS backbone, from AT&T's published Domestic_OC-768_Network.pdf)",
            "real": ["PoP identities", "PoP coordinates", "which links exist"],
            "derived": ["link delay, from great-circle distance x 1.5 fibre detour x 5us/km"],
            "synthesized": ["link capacity (OC-768 units)", "operational cost",
                            "traffic matrix (gravity model on metro population)"],
        },
        "links": _directed_links(inst),
        "demands": demands,
        "candidate_paths": paths,
        "current_routing": current,
        "notes": {
            "links_directed": "each undirected fibre span is emitted in both "
                              "directions with equal capacity (full duplex)",
            "latency_enforced_upstream": "only routes within each demand's "
                                         "latency_bound are exported",
            "protection_not_representable": "routing_qaoa selects one path per "
                                            "demand, so 1+1 protection is dropped; "
                                            "see needs_backup flags",
        },
    }


def write_json(inst, candidates: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_export(inst, candidates), indent=2))
    return path


def protected_demands(inst) -> list[str]:
    """Demands that require 1+1 protection upstream but cannot keep it once
    exported, because the downstream model routes each demand over one path."""
    return [d.id for d in inst.demands if d.needs_backup]


def instance_from_export(data: dict):
    """Build a live `routing_qaoa.RoutingInstance` from an export dict.

    Returns (instance, current_routing) ready to hand to
    `routing_qaoa.qubo.compute_coefficients`.
    """
    from routing_qaoa import CandidatePath, Demand, Link, RoutingInstance

    links = tuple(
        Link(l["u"], l["v"], capacity=l["capacity"], latency=l["latency"])
        for l in data["links"]
    )
    demands = tuple(
        Demand(
            d["name"],
            source=d["source"],
            target=d["target"],
            bandwidth=d["bandwidth"],
            priority=priority_weight(d["priority"]),
        )
        for d in data["demands"]
    )
    paths = {
        name: tuple(
            CandidatePath(tuple((u, v) for u, v in p["links"])) for p in plist
        )
        for name, plist in data["candidate_paths"].items()
    }
    return RoutingInstance(links, demands, paths), data["current_routing"]


def _path_nodes(entry) -> list[str]:
    """Accept a Yen Route, an export path dict, or a bare node list."""
    if hasattr(entry, "nodes"):
        return list(entry.nodes)
    if isinstance(entry, Mapping) and "nodes" in entry:
        return list(entry["nodes"])
    return list(entry)


def chosen_to_routing(chosen: Mapping[str, int | None], candidates, inst=None):
    """Turn a decoded path-index assignment into `Instance` routing.

    `chosen` maps demand id -> selected candidate index (`None` = unsatisfied).
    `candidates` is demand id -> sequence of Route objects, export path dicts,
    or node lists. Indices must match the `RoutingInstance` that produced
    `chosen`. Pass `inst` to run the same `build_export` reordering that
    `to_routing_instance` uses (today's route inserted as candidate 0).
    """
    if inst is not None:
        candidates = build_export(inst, candidates)["candidate_paths"]
    routing = {}
    for demand_id, idx in chosen.items():
        if idx is None:
            continue
        routing[demand_id] = [_path_nodes(candidates[demand_id][idx])]
    return routing


def to_routing_instance(inst, candidates: dict):
    """Build a live `routing_qaoa.RoutingInstance` (requires that package)."""
    return instance_from_export(build_export(inst, candidates))


def load_routing_instance(path: str | Path):
    """Build a live `routing_qaoa.RoutingInstance` from a written export JSON.

    Lets the QAOA stages run against the committed instances without
    re-deriving the topology or re-running Yen's.
    """
    return instance_from_export(json.loads(Path(path).read_text()))
