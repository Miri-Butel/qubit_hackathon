"""Tests for the matplotlib visualization helpers (headless Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from routing_qaoa import (
    QuboWeights,
    RoutingInstance,
    compute_coefficients,
    decode_bitstring,
)
from routing_qaoa.viz import (
    network_layout,
    plot_link_utilization,
    plot_network,
    plot_routing,
    to_networkx,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


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
