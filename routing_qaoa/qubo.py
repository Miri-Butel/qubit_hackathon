"""QUBO / cost-Hamiltonian construction for path-based traffic routing.

H = lambda_onehot * H_onehot + H_lat + lambda_cong * H_cong + lambda_switch * H_switch

with x_{k,p} in {0,1} selecting candidate path p for demand k:
    H_onehot = sum_k (sum_p x_{k,p} - 1)^2                  (exactly one path per demand)
    H_lat    = sum_{k,p} scale_lat * L_{k,p} * x_{k,p}      (L_{k,p} = pi_k * b_k * lat(p) by default)
    H_cong   = (1/m_used) * sum_e g(u_e),  u_e = sum_{(k,p): e in p} (b_k / c_e) * x_{k,p}
    H_switch = sum_{k in current} sum_{p != p_cur(k)} x_{k,p}

g(u) is u^2 ("quadratic", default) or alpha*u + beta*u^2 fitted to the
Fortz-Thorup link cost ("fortz-thorup-fit"). Latency is normalized so its
feasible-space spread equals cost_scale; congestion is O(cost_scale) for
utilizations O(1) — keeping phase(gamma * H) away from 2*pi wrap-around
(normalization approach per Classiq's network_traffic_optimization notebook).

References:
    B. Fortz, M. Thorup, "Internet Traffic Engineering by Optimizing OSPF
        Weights", IEEE INFOCOM 2000 (link cost Phi, normalized cost Phi*).
    E. Farhi, J. Goldstone, S. Gutmann, "A Quantum Approximate Optimization
        Algorithm", arXiv:1411.4028.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from .model import LinkKey, RoutingInstance

# Fortz-Thorup 2000, Sec. II: piecewise-linear convex link cost Phi_e(load).
# Derivative (slope) per utilization segment:
#   [0,1/3): 1   [1/3,2/3): 3   [2/3,9/10): 10   [9/10,1): 70   [1,11/10): 500   [11/10,inf): 5000
FT_BREAKPOINTS: tuple[float, ...] = (1 / 3, 2 / 3, 9 / 10, 1.0, 11 / 10)
FT_SLOPES: tuple[float, ...] = (1.0, 3.0, 10.0, 70.0, 500.0, 5000.0)

CongestionProfile = Literal["quadratic", "fortz-thorup-fit", "fortz-thorup-perlink"]


def fortz_thorup_link_cost(load: float, capacity: float) -> float:
    """Exact piecewise-linear Fortz-Thorup link cost Phi_e(load); Phi_e(c_e) = 32/3 * c_e."""
    if load < 0:
        raise ValueError("load must be >= 0")
    if capacity <= 0:
        raise ValueError("capacity must be > 0")
    cost = 0.0
    seg_start = 0.0
    for breakpoint, slope in zip(FT_BREAKPOINTS, FT_SLOPES):
        seg_end = breakpoint * capacity
        if load <= seg_end:
            return cost + slope * (load - seg_start)
        cost += slope * (seg_end - seg_start)
        seg_start = seg_end
    return cost + FT_SLOPES[-1] * (load - seg_start)


def fortz_thorup_quadratic_fit(
    u_max: float = 1.1, num_samples: int = 111
) -> tuple[float, float]:
    """Non-negative least-squares (alpha, beta) for alpha*u + beta*u^2 ~ Phi(u)/c on [0, u_max].

    Phi's hockey-stick tail drives an unconstrained linear coefficient negative
    (which would *reward* lightly loading links), so alpha is clamped >= 0 —
    in practice the fit is a tail-calibrated pure quadratic (alpha = 0,
    beta ~ 16 for u_max = 1.1), i.e. ~16x the default u^2 pressure.
    """
    import numpy as np
    from scipy.optimize import nnls

    u = np.linspace(0.0, u_max, num_samples)
    y = np.array([fortz_thorup_link_cost(float(ui), 1.0) for ui in u])
    basis = np.stack([u, u * u], axis=1)
    (alpha, beta), _ = nnls(basis, y)
    return float(alpha), float(beta)


def fortz_thorup_perlink_fit(
    u_lo: float, u_hi: float, num_samples: int = 64
) -> tuple[float, float]:
    """Least-squares (b, c) for b*u + c*u^2 ~ Phi(u)/c over [u_lo, u_hi].

    One *global* quadratic cannot be simultaneously cheap at u = 0.3 and
    catastrophic at u = 1.05, so `fortz-thorup-fit` minimizes total squared
    load and leaves a link sitting just over capacity: on the real AT&T
    instance its optimum keeps today's 105% peak, while the exact Phi clears it
    to 90%.

    Fitting each link over the utilization interval it can actually reach
    recovers the cliff. H already carries a per-link linear term and u_e is
    linear in x, so the fit stays a QUBO. The constant term is dropped: it is
    identical for every assignment and cannot move the argmin. b and c are
    unconstrained here -- a steep local segment needs a negative linear part,
    and clamping it (as the global fit must, to avoid rewarding empty links)
    is exactly what flattens the cliff away.
    """
    import numpy as np

    hi = max(u_hi, u_lo + 1e-6)
    u = np.linspace(u_lo, hi, num_samples)
    y = np.array([fortz_thorup_link_cost(float(x), 1.0) for x in u])
    basis = np.stack([np.ones_like(u), u, u * u], axis=1)
    coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return float(coeffs[1]), float(coeffs[2])


def reachable_load_bounds(
    instance: RoutingInstance,
) -> tuple[dict[LinkKey, float], dict[LinkKey, float]]:
    """Per-link (min, max) load over all one-hot assignments.

    A demand contributes its full bandwidth to a link at minimum when *every*
    one of its candidate paths crosses that link, and at maximum when *any*
    of them does. Derived from the candidate sets only -- never from a
    solution -- so the per-link fits stay independent of the answer.
    """
    lo: dict[LinkKey, float] = {}
    hi: dict[LinkKey, float] = {}
    for k, demand in enumerate(instance.demands):
        paths = instance.paths_of(k)
        crossings: dict[LinkKey, int] = {}
        for path in paths:
            for key in set(path.links):
                crossings[key] = crossings.get(key, 0) + 1
        for key, count in crossings.items():
            hi[key] = hi.get(key, 0.0) + demand.bandwidth
            if count == len(paths):
                lo[key] = lo.get(key, 0.0) + demand.bandwidth
    return lo, hi


@dataclass(frozen=True)
class QuboWeights:
    """Term weights and normalization knobs for the routing Hamiltonian."""

    lambda_onehot: float = 4.0
    lambda_cong: float = 1.0
    lambda_switch: float = 0.0
    cost_scale: float = 1.0
    bandwidth_weighted_latency: bool = True
    congestion_profile: CongestionProfile = "quadratic"


@dataclass(frozen=True)
class CostCoefficients:
    """Precomputed float coefficients of H over flat variables x_0..x_{N-1}.

    H(x) = sum_i linear[i]*x_i
         + cong_scale * sum_e (sum_{(i,a) in link_terms[e]} a*x_i)^2
         + lambda_onehot_scaled * sum_k (sum_{i in onehot_groups[k]} x_i - 1)^2
    """

    linear: tuple[float, ...]
    link_terms: tuple[tuple[tuple[int, float], ...], ...]
    onehot_groups: tuple[tuple[int, ...], ...]
    lambda_onehot_scaled: float
    cong_scale: float
    # Per-link quadratic weights, parallel to `link_terms`, used by the
    # `fortz-thorup-perlink` profile where each link carries its own local fit.
    # None means every link shares `cong_scale`.
    cong_weights: tuple[float, ...] | None = None


def compute_coefficients(
    instance: RoutingInstance,
    weights: QuboWeights = QuboWeights(),
    current_routing: dict[str, int] | None = None,
) -> CostCoefficients:
    """Precompute all Hamiltonian coefficients for `instance` as plain floats."""
    n = instance.num_qubits
    linear = [0.0] * n

    # Latency: L_{k,p} = pi_k * b_k * lat(p) (or pi_k * lat(p)), scaled so the
    # feasible-space spread [sum_k min_p, sum_k max_p] equals cost_scale. The
    # priority pi_k enters before lo/hi accumulate, so the spread invariant
    # holds for any priorities and a global rescale of all pi_k cancels.
    lat_raw = [0.0] * n
    lo, hi = 0.0, 0.0
    for k, demand in enumerate(instance.demands):
        weight = demand.priority * (
            demand.bandwidth if weights.bandwidth_weighted_latency else 1.0
        )
        costs = [
            weight * instance.path_latency(k, p)
            for p in range(len(instance.paths_of(k)))
        ]
        for p, cost in enumerate(costs):
            lat_raw[instance.flat_index(k, p)] = cost
        lo += min(costs)
        hi += max(costs)
    spread = hi - lo
    scale_lat = weights.cost_scale / (spread if spread > 1e-12 else max(hi, 1.0))
    for i in range(n):
        linear[i] += scale_lat * lat_raw[i]

    # Congestion: per used link e, u_e = sum (b_k / c_e) x_i; contributes
    # cong_scale * u_e^2 (+ folded linear alpha term for the Fortz-Thorup fit).
    # Priority is deliberately absent here: u_e is physical link utilization,
    # and the Fortz-Thorup fit is calibrated against real loads.
    incidence: dict[LinkKey, list[tuple[int, float]]] = {}
    for k, demand in enumerate(instance.demands):
        for p, path in enumerate(instance.paths_of(k)):
            i = instance.flat_index(k, p)
            for key in path.links:
                a = demand.bandwidth / instance.link_by_key(key).capacity
                incidence.setdefault(key, []).append((i, a))
    link_keys = tuple(incidence.keys())
    link_terms = tuple(tuple(terms) for terms in incidence.values())
    m_used = max(len(link_terms), 1)
    cong_base = weights.lambda_cong * weights.cost_scale / m_used
    cong_weights: tuple[float, ...] | None = None
    if weights.congestion_profile == "quadratic":
        alpha, beta = 0.0, 1.0
    elif weights.congestion_profile == "fortz-thorup-fit":
        alpha, beta = fortz_thorup_quadratic_fit()
    elif weights.congestion_profile == "fortz-thorup-perlink":
        # Each link gets a local fit of the exact Phi over the utilization
        # band it can reach, so the capacity cliff survives into the QUBO.
        alpha, beta = 0.0, 1.0
        lo_load, hi_load = reachable_load_bounds(instance)
        per_link: list[float] = []
        for key, terms in zip(link_keys, link_terms):
            capacity = instance.link_by_key(key).capacity
            b, c = fortz_thorup_perlink_fit(
                lo_load.get(key, 0.0) / capacity, hi_load.get(key, 0.0) / capacity
            )
            per_link.append(cong_base * c)
            for i, a in terms:
                linear[i] += cong_base * b * a
        cong_weights = tuple(per_link)
    else:
        raise ValueError(f"unknown congestion_profile {weights.congestion_profile!r}")
    cong_scale = cong_base * beta
    if alpha:
        for terms in link_terms:
            for i, a in terms:
                linear[i] += cong_base * alpha * a

    # Route-change penalty: +lambda_switch*cost_scale on every non-current path
    # of demands present in current_routing (skipped entirely when off).
    if current_routing and weights.lambda_switch:
        names = {demand.name: k for k, demand in enumerate(instance.demands)}
        for name, p_cur in current_routing.items():
            if name not in names:
                raise ValueError(f"current_routing has unknown demand {name!r}")
            k = names[name]
            if not 0 <= p_cur < len(instance.paths_of(k)):
                raise ValueError(f"current_routing[{name!r}] = {p_cur} out of range")
            for p in range(len(instance.paths_of(k))):
                if p != p_cur:
                    linear[instance.flat_index(k, p)] += (
                        weights.lambda_switch * weights.cost_scale
                    )

    onehot_groups = tuple(
        tuple(
            instance.flat_index(k, p) for p in range(len(instance.paths_of(k)))
        )
        for k in range(len(instance.demands))
    )
    return CostCoefficients(
        linear=tuple(linear),
        link_terms=link_terms,
        onehot_groups=onehot_groups,
        lambda_onehot_scaled=weights.lambda_onehot * weights.cost_scale,
        cong_scale=cong_scale,
        cong_weights=cong_weights,
    )


def build_cost_function(coeffs: CostCoefficients) -> Callable[[Sequence[Any]], Any]:
    """Dual-use cost H(x): symbolic on a QArray inside phase(), classical on list[int].

    Uses only +, *, **2 and precomputed floats so the identical expression
    builds the phase gadget and evaluates sampled bitstrings.
    """
    linear = coeffs.linear
    link_terms = coeffs.link_terms
    onehot_groups = coeffs.onehot_groups
    lambda_onehot = coeffs.lambda_onehot_scaled
    cong_scale = coeffs.cong_scale
    cong_weights = coeffs.cong_weights

    def cost(x: Sequence[Any]) -> Any:
        lat = sum(c * x[i] for i, c in enumerate(linear) if c != 0.0)
        onehot = sum((sum(x[i] for i in grp) - 1) ** 2 for grp in onehot_groups)
        if cong_weights is None:
            cong = sum(sum(a * x[i] for i, a in terms) ** 2 for terms in link_terms)
            cong_total = cong_scale * cong
        else:
            cong_total = sum(
                w * sum(a * x[i] for i, a in terms) ** 2
                for w, terms in zip(cong_weights, link_terms)
            )
        return lat + cong_total + lambda_onehot * onehot

    return cost


@dataclass(frozen=True)
class BruteForceResult:
    best_assignment: dict[str, int]
    best_cost: float
    best_bits: tuple[int, ...]
    num_combinations: int


def brute_force_feasible(
    instance: RoutingInstance,
    weights: QuboWeights = QuboWeights(),
    current_routing: dict[str, int] | None = None,
) -> BruteForceResult:
    """Exhaustive minimum of H over all one-hot (feasible) assignments."""
    coeffs = compute_coefficients(instance, weights, current_routing)
    cost = build_cost_function(coeffs)
    path_ranges = [range(len(instance.paths_of(k))) for k in range(len(instance.demands))]
    best_bits: tuple[int, ...] | None = None
    best_cost = float("inf")
    best_choice: tuple[int, ...] = ()
    count = 0
    for choice in itertools.product(*path_ranges):
        count += 1
        bits = [0] * instance.num_qubits
        for k, p in enumerate(choice):
            bits[instance.flat_index(k, p)] = 1
        value = cost(bits)
        if value < best_cost:
            best_cost, best_bits, best_choice = value, tuple(bits), choice
    assert best_bits is not None
    assignment = {
        demand.name: best_choice[k] for k, demand in enumerate(instance.demands)
    }
    return BruteForceResult(
        best_assignment=assignment,
        best_cost=best_cost,
        best_bits=best_bits,
        num_combinations=count,
    )
