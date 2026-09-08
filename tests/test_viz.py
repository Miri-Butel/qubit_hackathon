"""Tests for the matplotlib visualization helpers (headless Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from routing_qaoa import (
    QuboWeights,
    RoutingInstance,
    compute_coefficients,
    decode_bitstring,
)
from routing_qaoa.viz import (
    MAX_ENUMERABLE_QUBITS,
    network_layout,
    plot_energy_landscape,
    plot_fortz_thorup_curve,
    plot_link_utilization,
    plot_network,
    plot_routing,
    plot_sampled_cost_distribution,
    plot_solution_comparison,
    to_networkx,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _samples(*bit_prob: tuple[str, float]) -> pd.DataFrame:
    """A minimal stand-in for a classiq sample frame ('x' bits + 'probability')."""
    return pd.DataFrame(
        [{"x": bits, "probability": prob} for bits, prob in bit_prob]
    )


def test_to_networkx_carries_link_attributes(micro2: RoutingInstance) -> None:
    graph = to_networkx(micro2.links)
    assert graph.number_of_edges() == len(micro2.links)
    assert graph.edges["A", "T"] == {"capacity": 10, "latency": 2}


def test_network_layout_places_every_node(micro2: RoutingInstance) -> None:
    pos = network_layout(micro2)
    assert set(pos) == {"A", "B", "M", "T"}
    assert all(len(xy) == 2 for xy in pos.values())


def test_plot_network_draws_on_provided_axes(micro2: RoutingInstance) -> None:
    _, axes = plt.subplots(1, 2)
    assert plot_network(micro2, ax=axes[0]) is axes[0]
    assert axes[0].get_title() == "edge labels: latency / capacity"


def test_plot_routing_legend_lists_every_demand(micro2: RoutingInstance) -> None:
    ax = plot_routing(micro2, {"d1": 0, "d2": None})
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert len(labels) == len(micro2.demands)
    assert "path 0" in labels[0]
    assert "unsatisfied" in labels[1]


def test_plot_link_utilization_reports_overload(micro2: RoutingInstance) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    # d1 path 1 (qubit 1) + d2 path 0 (qubit 2) load shared link e3 to 7 > c=5.
    solution = decode_bitstring([0, 1, 1, 0], micro2, coeffs)
    assert solution.capacity_violations == 1
    ax = plot_link_utilization(micro2, solution)
    assert "1 over capacity" in ax.get_title()
    assert f"Φ* = {solution.phi_star:.2f}" in ax.get_title()


# --- presentation figures ---


def test_plot_solution_comparison_grids_two_solutions(micro2: RoutingInstance) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    overloaded = decode_bitstring([0, 1, 1, 0], micro2, coeffs)
    clean = decode_bitstring([1, 0, 0, 1], micro2, coeffs)
    fig = plot_solution_comparison(
        micro2, {"overloaded": overloaded, "clean": clean}
    )
    assert len(fig.axes) >= 4  # 2 routings x (paths + utilization); plus colorbars
    assert fig.axes[0].get_title() == "overloaded"


def test_plot_energy_landscape_marks_feasible_and_optimum(
    micro2: RoutingInstance,
) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    # micro2 has 4 qubits -> 16 states, 4 of them one-hot feasible.
    ax = plot_energy_landscape(
        micro2,
        coeffs,
        samples=_samples(("1001", 0.7), ("0110", 0.3)),
        num_shots=100,
        reference_cost=0.5,
    )
    feasible_points = ax.collections[1].get_offsets()
    assert len(feasible_points) == 4
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any("optimum" in label for label in labels)


def test_plot_energy_landscape_rejects_large_instances(
    micro2: RoutingInstance, monkeypatch: pytest.MonkeyPatch
) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    monkeypatch.setattr(type(micro2), "num_qubits", property(lambda self: 28))
    with pytest.raises(ValueError, match="too many to enumerate"):
        plot_energy_landscape(micro2, coeffs)


def test_plot_energy_landscape_without_samples(micro2: RoutingInstance) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    ax = plot_energy_landscape(micro2, coeffs)
    assert ax.get_ylabel() == "uniform probability"


def test_plot_sampled_cost_distribution_labels_each_set(
    micro2: RoutingInstance,
) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    ax = plot_sampled_cost_distribution(
        {
            "initial params": _samples(("0000", 0.5), ("1111", 0.5)),
            "optimized": _samples(("1001", 0.9), ("0110", 0.1)),
        },
        micro2,
        coeffs,
        num_shots=100,
        reference_cost=0.5,
    )
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "initial params" in labels and "optimized" in labels


def test_plot_fortz_thorup_curve_is_log_scaled(micro2: RoutingInstance) -> None:
    coeffs = compute_coefficients(micro2, QuboWeights())
    solution = decode_bitstring([0, 1, 1, 0], micro2, coeffs)
    ax = plot_fortz_thorup_curve(solution=solution, instance=micro2)
    assert ax.get_yscale() == "log"
    assert MAX_ENUMERABLE_QUBITS >= micro2.num_qubits


def test_plot_fortz_thorup_curve_requires_both_or_neither(
    micro2: RoutingInstance,
) -> None:
    plot_fortz_thorup_curve()  # curve alone is valid
    with pytest.raises(ValueError, match="both"):
        plot_fortz_thorup_curve(instance=micro2)
