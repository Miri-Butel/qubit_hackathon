import itertools
import math

import pytest

from routing_qaoa import (
    QuboWeights,
    RoutingInstance,
    brute_force_feasible,
    build_cost_function,
    compute_coefficients,
    decode_bitstring,
    fortz_thorup_link_cost,
    fortz_thorup_quadratic_fit,
    hop_distance,
    onehot_feasible,
    phi_uncap,
)
from tests.conftest import all_bitstrings

# ---- independent (from-first-principles) reference cost for micro2 ----


def manual_cost(
    bits: list[int],
    instance: RoutingInstance,
    weights: QuboWeights,
    ft_fit: tuple[float, float] | None = None,
) -> float:
    lat_raw, lo, hi = {}, 0.0, 0.0
    for k, demand in enumerate(instance.demands):
        b = demand.bandwidth if weights.bandwidth_weighted_latency else 1.0
        costs = [b * instance.path_latency(k, p) for p in range(len(instance.paths_of(k)))]
        for p, c in enumerate(costs):
            lat_raw[instance.flat_index(k, p)] = c
        lo, hi = lo + min(costs), hi + max(costs)
    scale_lat = weights.cost_scale / (hi - lo)
    lat = sum(scale_lat * lat_raw[i] * bits[i] for i in lat_raw)

    loads: dict[tuple[str, str], float] = {}
    for k, demand in enumerate(instance.demands):
        for p, path in enumerate(instance.paths_of(k)):
            if bits[instance.flat_index(k, p)]:
                for key in path.links:
                    loads[key] = loads.get(key, 0.0) + demand.bandwidth
    used_links = {
        key for k in range(len(instance.demands)) for path in instance.paths_of(k) for key in path.links
    }
    alpha, beta = ft_fit if ft_fit is not None else (0.0, 1.0)
    cong = 0.0
    for key in used_links:
        u = loads.get(key, 0.0) / instance.link_by_key(key).capacity
        cong += alpha * u + beta * u * u
    cong *= weights.lambda_cong * weights.cost_scale / len(used_links)

    onehot = weights.lambda_onehot * weights.cost_scale * sum(
        (sum(bits[instance.flat_index(k, p)] for p in range(len(instance.paths_of(k)))) - 1) ** 2
        for k in range(len(instance.demands))
    )
    return lat + cong + onehot


# ---- cost function ----


def test_cost_matches_manual_on_all_bitstrings(micro2: RoutingInstance) -> None:
    weights = QuboWeights()
    cost = build_cost_function(compute_coefficients(micro2, weights))
    for bits in all_bitstrings(micro2.num_qubits):
        assert cost(bits) == pytest.approx(manual_cost(bits, micro2, weights))


def test_onehot_penalty_on_empty_assignment(micro2: RoutingInstance) -> None:
    cost = build_cost_function(compute_coefficients(micro2))
    # all-zero: no latency, no congestion, each of the 2 demands violates by (0-1)^2
    assert cost([0, 0, 0, 0]) == pytest.approx(4.0 * 2)


def test_latency_spread_equals_cost_scale(micro2: RoutingInstance) -> None:
    for cost_scale in (1.0, 2.5):
        weights = QuboWeights(lambda_cong=0.0, cost_scale=cost_scale)
        cost = build_cost_function(compute_coefficients(micro2, weights))
        values = [
            cost(bits)
            for bits in all_bitstrings(micro2.num_qubits)
            if onehot_feasible(bits, micro2)
        ]
        assert max(values) - min(values) == pytest.approx(cost_scale)


def test_bandwidth_weighting_toggle(micro2: RoutingInstance) -> None:
    weighted = compute_coefficients(micro2, QuboWeights())
    unweighted = compute_coefficients(
        micro2, QuboWeights(bandwidth_weighted_latency=False)
    )
    # weighted: d1 L = 4*2 = 8, spread = (8+9)-(8+6) = 3 -> linear[0] = 8/3
    assert weighted.linear[0] == pytest.approx(8 / 3)
    # unweighted: d1 L = 2, spread = (2+3)-(2+2) = 1 -> linear[0] = 2
    assert unweighted.linear[0] == pytest.approx(2.0)


def test_switch_penalty(micro2: RoutingInstance) -> None:
    weights = QuboWeights(lambda_cong=0.0, lambda_switch=0.5)
    cost = build_cost_function(
        compute_coefficients(micro2, weights, current_routing={"d1": 0})
    )
    # d1's two paths have equal (weighted) latency, so with congestion off the
    # only difference between staying and switching is the switch penalty.
    stay, switch = cost([1, 0, 1, 0]), cost([0, 1, 1, 0])
    assert switch - stay == pytest.approx(0.5 * weights.cost_scale)


def test_switch_penalty_validates_input(micro2: RoutingInstance) -> None:
    weights = QuboWeights(lambda_switch=0.5)
    with pytest.raises(ValueError, match="unknown demand"):
        compute_coefficients(micro2, weights, current_routing={"ghost": 0})
    with pytest.raises(ValueError, match="out of range"):
        compute_coefficients(micro2, weights, current_routing={"d1": 5})


# ---- brute force ----


def test_brute_force_is_global_minimum(micro2: RoutingInstance) -> None:
    weights = QuboWeights()
    cost = build_cost_function(compute_coefficients(micro2, weights))
    result = brute_force_feasible(micro2, weights)
    global_min = min(cost(bits) for bits in all_bitstrings(micro2.num_qubits))
    # lambda_onehot = 4 dominates: the unconstrained minimum is one-hot feasible
    assert result.best_cost == pytest.approx(global_min)
    assert onehot_feasible(list(result.best_bits), micro2)
    assert result.num_combinations == 4
    # latency-optimal AND congestion-safe choice: d1 direct (p0), d2 via M (p0)
    assert result.best_assignment == {"d1": 0, "d2": 0}


# ---- Fortz-Thorup ----


def test_fortz_thorup_link_cost_breakpoints() -> None:
    for c in (1.0, 5.0, 10.0):
        assert fortz_thorup_link_cost(0.0, c) == 0.0
        assert fortz_thorup_link_cost(c / 3, c) == pytest.approx(c / 3)
        assert fortz_thorup_link_cost(c, c) == pytest.approx(32 * c / 3)
        assert fortz_thorup_link_cost(1.2 * c, c) == pytest.approx(32 * c / 3 + 550 * c)
    samples = [i / 20 for i in range(31)]  # utilizations 0 .. 1.5
    values = [fortz_thorup_link_cost(u, 1.0) for u in samples]
    assert all(b > a for a, b in zip(values, values[1:]))  # strictly increasing
    slopes = [(b - a) for a, b in zip(values, values[1:])]
    assert all(s2 >= s1 - 1e-9 for s1, s2 in zip(slopes, slopes[1:]))  # convex


def test_fortz_thorup_quadratic_fit() -> None:
    alpha, beta = fortz_thorup_quadratic_fit()
    assert alpha >= 0.0
    assert 10.0 < beta < 25.0  # tail-calibrated quadratic, ~16 for u_max = 1.1
    fitted = [alpha * u + beta * u * u for u in (0.0, 0.3, 0.7, 1.1)]
    assert all(b > a for a, b in zip(fitted, fitted[1:]))


def test_fortz_thorup_profile_cost(micro2: RoutingInstance) -> None:
    weights = QuboWeights(congestion_profile="fortz-thorup-fit")
    cost = build_cost_function(compute_coefficients(micro2, weights))
    fit = fortz_thorup_quadratic_fit()
    for bits in ([1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1], [0, 0, 0, 0]):
        assert cost(bits) == pytest.approx(manual_cost(bits, micro2, weights, ft_fit=fit))


# ---- decode / KPIs ----


def test_decode_known_assignment(micro2: RoutingInstance) -> None:
    coeffs = compute_coefficients(micro2)
    sol = decode_bitstring([0, 1, 1, 0], micro2, coeffs)  # d1 -> p1, d2 -> p0
    assert sol.chosen == {"d1": 1, "d2": 0}
    assert sol.feasible_onehot and sol.demands_satisfied == 2
    kpis = {kpi.link: kpi for kpi in sol.link_kpis}
    assert kpis[("M", "T")].load == pytest.approx(7)  # shared e3: 4 + 3
    assert kpis[("M", "T")].utilization == pytest.approx(1.4)
    assert kpis[("M", "T")].over_capacity
    assert kpis[("A", "T")].load == 0
    assert sol.max_utilization == pytest.approx(1.4)
    assert sol.capacity_violations == 1
    assert sol.total_latency == pytest.approx(4)  # 2 + 2
    assert sol.weighted_latency == pytest.approx(14)  # 4*2 + 3*2
    # Phi: e2 -> Phi(4,10) = 16/3, e3 -> Phi(7,5) = 40/3 + 7790, e4 -> Phi(3,10) = 3
    assert sol.phi_total == pytest.approx(16 / 3 + 40 / 3 + 7790 + 3)
    assert sol.phi_star == pytest.approx(sol.phi_total / 7)
    assert sol.cost == pytest.approx(
        build_cost_function(coeffs)([0, 1, 1, 0])
    )


def test_decode_infeasible_assignment(micro2: RoutingInstance) -> None:
    coeffs = compute_coefficients(micro2)
    sol = decode_bitstring([1, 1, 0, 0], micro2, coeffs)  # d1 double, d2 none
    assert sol.chosen == {"d1": None, "d2": None}
    assert not sol.feasible_onehot
    assert sol.demands_satisfied == 0
    assert sol.total_latency == 0
    assert all(kpi.load == 0 for kpi in sol.link_kpis)


def test_phi_uncap_and_hop_distance(micro2: RoutingInstance) -> None:
    assert hop_distance(micro2, "A", "T") == 1  # direct link e1
    assert hop_distance(micro2, "B", "T") == 1  # direct link e5
    assert phi_uncap(micro2) == pytest.approx(4 * 1 + 3 * 1)
    with pytest.raises(ValueError, match="no route"):
        hop_distance(micro2, "T", "A")
