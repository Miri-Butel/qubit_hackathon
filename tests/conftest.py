"""Shared micro-instances for tests.

micro2 topology (4 qubits: d1 -> {0, 1}, d2 -> {2, 3}):

    A --e1(c=10, lat=2)--> T           d1: A->T, b=4, paths [e1] / [e2, e3]
    A --e2(c=10, lat=1)--> M           d2: B->T, b=3, paths [e4, e3] / [e5]
    M --e3(c=5,  lat=1)--> T
    B --e4(c=10, lat=1)--> M           e3 is shared: d1 path 1 + d2 path 0
    B --e5(c=10, lat=3)--> T           load it 4 + 3 = 7 > c = 5.
"""

import pytest

from routing_qaoa import CandidatePath, Demand, Link, RoutingInstance

E1, E2, E3, E4, E5 = ("A", "T"), ("A", "M"), ("M", "T"), ("B", "M"), ("B", "T")

MICRO2_LINKS = (
    Link(*E1, capacity=10, latency=2),
    Link(*E2, capacity=10, latency=1),
    Link(*E3, capacity=5, latency=1),
    Link(*E4, capacity=10, latency=1),
    Link(*E5, capacity=10, latency=3),
)
MICRO2_DEMANDS = (
    Demand("d1", source="A", target="T", bandwidth=4),
    Demand("d2", source="B", target="T", bandwidth=3),
)
MICRO2_PATHS = {
    "d1": (CandidatePath((E1,)), CandidatePath((E2, E3))),
    "d2": (CandidatePath((E4, E3)), CandidatePath((E5,))),
}


@pytest.fixture
def micro2() -> RoutingInstance:
    return RoutingInstance(MICRO2_LINKS, MICRO2_DEMANDS, dict(MICRO2_PATHS))


def all_bitstrings(n: int) -> list[list[int]]:
    return [[(value >> i) & 1 for i in range(n)] for value in range(2**n)]
