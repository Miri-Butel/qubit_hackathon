"""Classiq QAOA solver for the routing QUBO (the only module importing classiq).

Ansatz and execution follow the Classiq combinatorial-optimization idiom
(classiq-library: applications/telecom/network_traffic_optimization):
|+>^N initial state, alternating phase(H, gamma_l) / RX-mixer(beta_l) layers,
ExecutionSession + scipy COBYLA over estimate_cost, final sample.

QAOA: Farhi, Goldstone, Gutmann, arXiv:1411.4028.

NOTE: no `from __future__ import annotations` here — @qfunc needs the size
annotations (CArray[CReal, 2 * num_layers], QArray[QBit, num_qubits]) to be
evaluated eagerly at decoration time, not deferred to strings.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from classiq import (
    CArray,
    CReal,
    ExecutionSession,
    Output,
    QArray,
    QBit,
    RX,
    allocate,
    apply_to_all,
    hadamard_transform,
    phase,
    qfunc,
    repeat,
    synthesize,
)

from .model import RoutingInstance
from .qubo import QuboWeights, build_cost_function, compute_coefficients

OUTPUT_VAR = "x"  # main's output name; must match state[OUTPUT_VAR] in estimate_cost


@dataclass(frozen=True)
class QaoaConfig:
    num_layers: int = 3
    num_shots: int = 2048
    max_iterations: int = 60
    random_seed: int | None = 42
    # CVaR fraction for estimate_cost (Barkoutsos et al., arXiv:1907.04769):
    # optimize the mean of the best `quantile` of sampled energies; 1.0 = plain mean.
    quantile: float = 1.0


def build_qaoa_main(
    cost_fn: Callable[[Sequence[Any]], Any], num_qubits: int, num_layers: int
):
    """Build the QAOA `main` qfunc for a cost function over `num_qubits` bits.

    Sizes are baked into the annotations at decoration time, so a new `main`
    is built per problem instance / layer count.
    """

    @qfunc
    def mixer_layer(beta: CReal, qba: QArray[QBit]) -> None:
        apply_to_all(lambda q: RX(beta, q), qba)

    @qfunc
    def main(
        params: CArray[CReal, 2 * num_layers],
        x: Output[QArray[QBit, num_qubits]],
    ) -> None:
        allocate(x)
        hadamard_transform(x)
        repeat(
            count=num_layers,
            iteration=lambda i: (
                phase(cost_fn(x), params[2 * i]),
                mixer_layer(params[2 * i + 1], x),
            ),
        )

    return main


def initial_qaoa_params(num_layers: int) -> np.ndarray:
    """Midpoint linear schedule: gammas ramp up, betas ramp down, interleaved."""
    gammas = math.pi * np.linspace(
        1 / (2 * num_layers), 1 - 1 / (2 * num_layers), num_layers
    )
    betas = gammas[::-1]
    params = np.empty(2 * num_layers)
    params[0::2] = gammas
    params[1::2] = betas
    return params


@dataclass
class QaoaResult:
    optimal_params: np.ndarray
    objective_values: list[float] = field(repr=False)
    samples: pd.DataFrame = field(repr=False)
    num_shots: int = 0


def run_qaoa(
    instance: RoutingInstance,
    weights: QuboWeights = QuboWeights(),
    config: QaoaConfig = QaoaConfig(),
    current_routing: dict[str, int] | None = None,
    qprog: Any | None = None,
) -> QaoaResult:
    """Synthesize (unless `qprog` is given), optimize QAOA parameters, and sample.

    A passed-in `qprog` must have been synthesized from build_qaoa_main with
    the same instance/weights/current_routing and config.num_layers.
    """
    coeffs = compute_coefficients(instance, weights, current_routing)
    cost_fn = build_cost_function(coeffs)
    if qprog is None:
        qprog = synthesize(
            build_qaoa_main(cost_fn, instance.num_qubits, config.num_layers)
        )

    objective_values: list[float] = []
    es = ExecutionSession(
        qprog, num_shots=config.num_shots, random_seed=config.random_seed
    )
    try:

        def estimate(params: np.ndarray) -> float:
            value = es.estimate_cost(
                cost_func=lambda state: cost_fn(state[OUTPUT_VAR]),
                parameters={"params": params.tolist()},
                quantile=config.quantile,
            )
            objective_values.append(value)
            return value

        optimization = minimize(
            estimate,
            x0=initial_qaoa_params(config.num_layers),
            method="COBYLA",
            options={"maxiter": config.max_iterations},
        )
        raw = es.sample({"params": optimization.x.tolist()})
        # classiq 1.29: sample() returns a DataFrame; older versions need .dataframe
        samples = raw if isinstance(raw, pd.DataFrame) else raw.dataframe
    finally:
        es.close()

    return QaoaResult(
        optimal_params=optimization.x,
        objective_values=objective_values,
        samples=samples,
        num_shots=config.num_shots,
    )
