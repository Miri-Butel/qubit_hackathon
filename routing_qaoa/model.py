"""Data model for path-based network traffic routing.

Pipeline context (AT&T routing challenge): candidate paths per demand are
generated classically (pipeline step 2, e.g. K-shortest paths); this package
selects one path per demand with QAOA (steps 3-4).

Decision variable convention: x_{k,p} = 1 iff demand k routes on its
candidate path p. Flat qubit index i = offsets[k] + p, N = sum_k |P_k|.
"""

from __future__ import annotations

from dataclasses import dataclass

NodeId = str
LinkKey = tuple[NodeId, NodeId]


@dataclass(frozen=True)
class Link:
    """Directed network link with capacity c_e and latency lat_e."""

    u: NodeId
    v: NodeId
    capacity: float
    latency: float

    @property
    def key(self) -> LinkKey:
        return (self.u, self.v)


@dataclass(frozen=True)
class Demand:
    """Traffic demand k: route `bandwidth` (b_k) from `source` to `target`."""

    name: str
    source: NodeId
    target: NodeId
    bandwidth: float


@dataclass(frozen=True)
class CandidatePath:
    """Ordered chain of link keys from a demand's source to its target."""

    links: tuple[LinkKey, ...]


@dataclass(frozen=True)
class RoutingInstance:
    """A validated routing problem: network, demands, and candidate paths."""

    links: tuple[Link, ...]
    demands: tuple[Demand, ...]
    candidate_paths: dict[str, tuple[CandidatePath, ...]]

    def __post_init__(self) -> None:
        links_by_key = {link.key: link for link in self.links}
        self._validate(links_by_key)
        object.__setattr__(self, "_links_by_key", links_by_key)
        offsets: list[int] = []
        total = 0
        for demand in self.demands:
            offsets.append(total)
            total += len(self.candidate_paths[demand.name])
        object.__setattr__(self, "_offsets", tuple(offsets))
        object.__setattr__(self, "_num_qubits", total)

    # --- flat index map: qubit i = offsets[k] + p ---

    @property
    def offsets(self) -> tuple[int, ...]:
        return self._offsets  # type: ignore[attr-defined]

    @property
    def num_qubits(self) -> int:
        return self._num_qubits  # type: ignore[attr-defined]

    def paths_of(self, k: int) -> tuple[CandidatePath, ...]:
        return self.candidate_paths[self.demands[k].name]

    def flat_index(self, k: int, p: int) -> int:
        if not 0 <= p < len(self.paths_of(k)):
            raise IndexError(f"demand {k} has no candidate path {p}")
        return self.offsets[k] + p

    def demand_path_at(self, i: int) -> tuple[int, int]:
        """Inverse of flat_index: qubit index -> (demand index k, path index p)."""
        if not 0 <= i < self.num_qubits:
            raise IndexError(f"flat index {i} out of range [0, {self.num_qubits})")
        for k in reversed(range(len(self.demands))):
            if i >= self.offsets[k]:
                return k, i - self.offsets[k]
        raise AssertionError("unreachable")

    def link_by_key(self, key: LinkKey) -> Link:
        return self._links_by_key[key]  # type: ignore[attr-defined]

    def path_latency(self, k: int, p: int) -> float:
        return sum(self.link_by_key(key).latency for key in self.paths_of(k)[p].links)

    # --- validation ---

    def _validate(self, links_by_key: dict[LinkKey, Link]) -> None:
        if len(links_by_key) != len(self.links):
            raise ValueError("duplicate link keys (u, v) in links")
        for link in self.links:
            if link.capacity <= 0:
                raise ValueError(f"link {link.key} capacity must be > 0")
            if link.latency < 0:
                raise ValueError(f"link {link.key} latency must be >= 0")

        names = [demand.name for demand in self.demands]
        if len(set(names)) != len(names):
            raise ValueError("duplicate demand names")
        for demand in self.demands:
            if demand.bandwidth <= 0:
                raise ValueError(f"demand {demand.name!r} bandwidth must be > 0")

        if set(self.candidate_paths) != set(names):
            raise ValueError("candidate_paths keys must match demand names exactly")
        for demand in self.demands:
            paths = self.candidate_paths[demand.name]
            if not paths:
                raise ValueError(f"demand {demand.name!r} has no candidate paths")
            for p, path in enumerate(paths):
                self._validate_path(demand, p, path, links_by_key)

    @staticmethod
    def _validate_path(
        demand: Demand, p: int, path: CandidatePath, links_by_key: dict[LinkKey, Link]
    ) -> None:
        where = f"demand {demand.name!r} path {p}"
        if not path.links:
            raise ValueError(f"{where} is empty")
        for key in path.links:
            if key not in links_by_key:
                raise ValueError(f"{where} uses unknown link {key}")
        for (_, v), (u, _) in zip(path.links, path.links[1:]):
            if v != u:
                raise ValueError(f"{where} links do not chain: {v} != {u}")
        if path.links[0][0] != demand.source:
            raise ValueError(f"{where} does not start at source {demand.source!r}")
        if path.links[-1][1] != demand.target:
            raise ValueError(f"{where} does not end at target {demand.target!r}")
