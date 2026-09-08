# network_instance — pipeline steps 1-2

Builds the network and generates the candidate routes that `routing_qaoa`
selects between. This is the classical half of the pipeline: the graph, the
traffic, the constraints, and K-shortest-paths. It does no QUBO and no QAOA.

`routing_qaoa/model.py` describes the handoff as *"candidate paths per demand
are generated classically (pipeline step 2, e.g. K-shortest paths); this
package selects one path per demand with QAOA (steps 3-4)"*. This is that
step 2.

## Quick start

```python
import sys; sys.path.insert(0, "network_instance")
import att_real, yen, export

net   = att_real.att_backbone(region="east10_dense")   # real AT&T topology
paths = yen.budgeted_candidate_set(net, k=5, base=2)   # Yen's K-shortest
instance, current_routing = export.to_routing_instance(net, paths)

# hand straight to routing_qaoa
from routing_qaoa.qubo import compute_coefficients, QuboWeights
coeffs = compute_coefficients(instance, QuboWeights(), current_routing)
```

`export.to_routing_instance` imports `routing_qaoa`, so it only runs once this
branch and `feature/qaoa-routing` are merged. Everything else here, including
the JSON export, is standalone:

```python
export.write_json(net, paths, "instance.json")   # no routing_qaoa needed
```

Prebuilt exports are already in `data/*.json` if you would rather not run the
generator at all. `data/att_east10_dense_instance.json` is the 28-qubit one.

## The two instances

| region | demands | qubits | use |
| --- | --- | --- | --- |
| `east10` | 5 | 10 | readable: small enough to check route tables by eye |
| `east10_dense` | 14 | 28 | the real test case: fills the 28-qubit budget |

Use **`east10_dense`** for QAOA work. `east10` is too sparse to be a real
optimization: its demands barely share links, so today's shortest-path
routing is already optimal and the solver has nothing to find.

`east10_dense` is verified non-degenerate. An exhaustive search over all
16384 assignments beats today's routing, rerouting 2 demands and cutting
total congestion `sum(u^2)` from 36.1 to 32.1, about 11%.

## Where the data comes from

Topology is **AT&T's real North America MPLS backbone** from the
[Internet Topology Zoo](http://www.topology-zoo.org/) (Knight, Nguyen, Falkner,
Bowden, Roughan, IEEE JSAC 2011), which built it from AT&T's own published
`Domestic_OC-768_Network.pdf`. 25 PoPs, 56 links, real PoP codes, real
coordinates. `east10` is a 10-PoP slice of it.

Be precise about this when presenting:

- **Real**: which PoPs exist, where they are, which links exist.
- **Derived** from that: link delay, from great-circle distance x 1.5 fibre
  detour x 5us/km. This reproduces reality well; it puts New York to Los
  Angeles at 30.3ms, which is what that path actually measures.
- **Synthesized** by us, methods documented in `att_real.py`: link capacity
  (in OC-768 units, since the source is AT&T's OC-768 map), operational cost,
  and the traffic matrix (gravity model on metro populations). No carrier
  publishes real capacity or real traffic.

## Candidate route generation

`yen.py` implements Yen's K-shortest-loopless-paths from scratch. It is
verified against networkx's own implementation across all 300 PoP pairs of
the full AT&T network under three metrics: 900 comparisons, zero mismatches.

The brief gives each link three attributes that pull against each other, so
Yen's runs once per metric and the results are merged:

| metric | constraint it serves |
| --- | --- |
| `delay` | latency bound |
| `cost` | operational cost per Gbps |
| `hops` | resource usage |
| `spare` | link capacity — prefer links with headroom. This is the minimum-interference idea from the challenge's own reading list |

The metrics genuinely disagree. The fastest route is routinely one already
oversubscribed, while the route with real headroom is slower or breaks the
latency bound. That disagreement is the optimization.

Shortlisting to fit the qubit budget is in `yen.select_candidates`. The rule
is not "keep the fastest": two near-identical routes down the same corridor
give the solver no real decision and waste a qubit. It keeps the fastest
route, then repeatedly adds whichever remaining route shares the fewest links
with those already chosen.

## Four things to know about the handoff

`export.py` converts to `routing_qaoa`'s `RoutingInstance`. The two models do
not line up perfectly:

1. **Links become directed.** Our graph is undirected; `Link` is keyed
   `(u, v)`. Every span is emitted in both directions at equal capacity,
   which matches full-duplex fibre. Consequence worth knowing: the two
   directions are independent capacity constraints, so an instance that looks
   285% congested in an undirected view arrives materially less congested.

2. **Operational cost is not carried into the QUBO.** `Link` has capacity and
   latency only. Cost still does real work upstream, since it is one of the
   four metrics Yen's ranks under, but the Hamiltonian cannot price it.
   Adding a `cost` field and a term is the natural extension. The JSON export
   carries `cost` on every link already, so nothing needs regenerating.

3. **Latency bounds and protection are enforced here, not there.** The
   downstream model picks one path per demand with no SLA bound and no notion
   of a protected demand needing two disjoint paths. So latency is enforced by
   only ever exporting routes that satisfy it. 1+1 protection cannot be
   expressed downstream at all; `export.protected_demands()` reports which
   demands lose that guarantee.

4. **Today's route is always candidate-included**, and `current_routing` gives
   its index. `H_switch` penalizes every path that is not the current one, so
   if today's route were missing from the shortlist the penalty would be
   measured against something unpickable and the route-churn metric would be
   meaningless.

## One tuning note

At default `QuboWeights()`, congestion is divided by `m_used` (the number of
used links) while latency is normalized to `cost_scale`. On a 16-link
instance that dilutes congestion roughly 16x relative to latency. Combined
with candidate 0 being the delay-optimal route, the latency term can dominate
hard enough that the optimum is trivially "keep today's routing". Worth
sweeping `lambda_cong` rather than assuming the default is balanced.

## Files

| file | what it does |
| --- | --- |
| `att_real.py` | loads the real AT&T topology, derives delays, synthesizes capacity and traffic |
| `yen.py` | Yen's K-shortest paths, the four metrics, candidate shortlisting |
| `export.py` | bridge to `routing_qaoa.RoutingInstance` and JSON |
| `topologies.py` | `Instance` container, validation, hand-built synthetic instances |
| `topology.py` | `Demand` dataclass and candidate helpers |
| `viz.py` | draws the network on a real US map, plus route and congestion views |
| `metrics.py` | scores a routing on the brief's five measures |
| `data/` | the Topology Zoo source file, US state boundaries, prebuilt JSON exports |

`validate()` in `topologies.py` checks an instance against the brief's
"no violations of link capacity or resiliency rules" rule. It has already
caught two real bugs: a backup path that quietly reused a link from its own
primary, and a latency bound that made today's own routing illegal.
