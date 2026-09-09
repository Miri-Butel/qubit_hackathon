import pytest

from routing_qaoa import CandidatePath, Demand, Link, RoutingInstance
from tests.conftest import E1, E2, E3, MICRO2_DEMANDS, MICRO2_LINKS, MICRO2_PATHS


def test_index_map_roundtrip(micro2: RoutingInstance) -> None:
    assert micro2.num_qubits == 4
    assert micro2.offsets == (0, 2)
    for k in range(len(micro2.demands)):
        for p in range(len(micro2.paths_of(k))):
            assert micro2.demand_path_at(micro2.flat_index(k, p)) == (k, p)
    assert [micro2.flat_index(*micro2.demand_path_at(i)) for i in range(4)] == [0, 1, 2, 3]
    with pytest.raises(IndexError):
        micro2.demand_path_at(4)
    with pytest.raises(IndexError):
        micro2.flat_index(0, 2)


def test_path_latency(micro2: RoutingInstance) -> None:
    assert micro2.path_latency(0, 0) == 2  # d1 via e1
    assert micro2.path_latency(0, 1) == 2  # d1 via e2 + e3
    assert micro2.path_latency(1, 1) == 3  # d2 via e5


def _build(links=MICRO2_LINKS, demands=MICRO2_DEMANDS, paths=MICRO2_PATHS) -> RoutingInstance:
    return RoutingInstance(links, demands, dict(paths))


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="duplicate link keys"):
        _build(links=MICRO2_LINKS + (Link(*E1, capacity=1, latency=1),))
    with pytest.raises(ValueError, match="capacity must be > 0"):
        _build(links=MICRO2_LINKS[1:] + (Link(*E1, capacity=0, latency=2),))
    with pytest.raises(ValueError, match="latency must be >= 0"):
        _build(links=MICRO2_LINKS[1:] + (Link(*E1, capacity=10, latency=-1),))
    with pytest.raises(ValueError, match="duplicate demand names"):
        _build(
            demands=MICRO2_DEMANDS + (Demand("d1", "A", "T", 1),),
            paths={**MICRO2_PATHS},
        )
    with pytest.raises(ValueError, match="bandwidth must be > 0"):
        _build(demands=(MICRO2_DEMANDS[0], Demand("d2", "B", "T", 0)))
    with pytest.raises(ValueError, match="keys must match demand names"):
        _build(paths={**MICRO2_PATHS, "ghost": (CandidatePath((E1,)),)})
    with pytest.raises(ValueError, match="has no candidate paths"):
        _build(paths={**MICRO2_PATHS, "d2": ()})
    with pytest.raises(ValueError, match="path 0 is empty"):
        _build(paths={**MICRO2_PATHS, "d2": (CandidatePath(()),)})
    with pytest.raises(ValueError, match="unknown link"):
        _build(paths={**MICRO2_PATHS, "d2": (CandidatePath((("B", "Z"),)),)})
    with pytest.raises(ValueError, match="do not chain"):
        _build(paths={**MICRO2_PATHS, "d1": (CandidatePath((E2, E1)),)})
    with pytest.raises(ValueError, match="does not start at source"):
        _build(paths={**MICRO2_PATHS, "d2": (CandidatePath((E2, E3)),)})
    with pytest.raises(ValueError, match="does not end at target"):
        _build(paths={**MICRO2_PATHS, "d1": (CandidatePath((E2,)),)})
