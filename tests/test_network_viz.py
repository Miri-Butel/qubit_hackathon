"""Smoke tests for the AT&T geographic pipeline figures and the chosen-path bridge."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

ROOT = Path(__file__).resolve().parent.parent
NET = ROOT / "network_instance"
if str(NET) not in sys.path:
    sys.path.insert(0, str(NET))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import att_real as A  # noqa: E402
import export  # noqa: E402
import viz  # noqa: E402
import yen  # noqa: E402
from routing_qaoa import QuboWeights, brute_force_feasible, compute_coefficients, decode_bitstring  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def east10():
    return A.att_backbone(region="east10")


@pytest.fixture(scope="module")
def full_backbone():
    return A.att_backbone()


@pytest.fixture(scope="module")
def east10_solved(east10):
    paths = yen.budgeted_candidate_set(east10, k=5, base=2, extra_for=2)
    instance, _ = export.to_routing_instance(east10, paths)
    weights = QuboWeights(congestion_profile="fortz-thorup-fit")
    ref = brute_force_feasible(instance, weights)
    coeffs = compute_coefficients(instance, weights)
    sol = decode_bitstring(list(ref.best_bits), instance, coeffs)
    routing = export.chosen_to_routing(sol.chosen, paths, inst=east10)
    return paths, instance, sol, routing


def test_chosen_to_routing_accepts_export_dicts_and_routes(east10):
    paths = yen.budgeted_candidate_set(east10, k=5, base=2)
    data = export.build_export(east10, paths)
    demand = east10.demands[0].id

    from_export = export.chosen_to_routing({demand: 0}, data["candidate_paths"])
    from_routes = export.chosen_to_routing({demand: 0}, paths)
    from_nodes = export.chosen_to_routing({demand: 0}, {demand: [r.nodes for r in paths[demand]]})

    assert from_export[demand][0] == data["candidate_paths"][demand][0]["nodes"]
    assert from_routes[demand][0] == paths[demand][0].nodes
    assert from_nodes[demand][0] == paths[demand][0].nodes


def test_chosen_to_routing_skips_unsatisfied():
    assert export.chosen_to_routing({"T1": None, "T2": 0}, {"T2": [["A", "B"]]}) == {
        "T2": [["A", "B"]]
    }


def test_chosen_to_routing_matches_export_order(east10):
    paths = yen.budgeted_candidate_set(east10, k=5, base=2)
    data = export.build_export(east10, paths)
    via_inst = export.chosen_to_routing(data["current_routing"], paths, inst=east10)
    for demand_id, idx in data["current_routing"].items():
        assert via_inst[demand_id][0] == data["candidate_paths"][demand_id][idx]["nodes"]
        assert via_inst[demand_id][0] == east10.current_routing[demand_id][0]


def test_draw_map_slice_uses_provided_axes(full_backbone):
    _, ax = plt.subplots()
    keep = A.REGIONS["east10"]
    assert viz.draw_map_slice(full_backbone, keep, ax=ax) is ax
    assert "10 of 25" in ax.get_title()


def test_draw_map_routes_lists_candidates(east10):
    paths = yen.budgeted_candidate_set(east10, k=5, base=2)
    heavy = viz.heaviest_demands(east10, 2)
    _, ax = plt.subplots()
    viz.draw_map_routes(east10, {did: paths[did] for did in heavy}, ax=ax)
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert any(heavy[0] in label for label in labels)
    assert any(heavy[1] in label for label in labels)


def test_draw_map_solution_labels_highlighted_demands(east10, east10_solved):
    _, _, sol, routing = east10_solved
    highlight = viz.heaviest_demands(east10, 3)
    _, ax = plt.subplots()
    viz.draw_map_solution(
        east10, routing, ax=ax, highlight=highlight, objective=sol.cost,
    )
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    for did in highlight:
        assert any(did in label for label in labels)
    assert "Objective=" in ax.get_title()
    assert "max util" in ax.get_title()
