"""Offline Classiq checks: model construction needs no auth/network."""

import math

import numpy as np

from routing_qaoa import build_cost_function, compute_coefficients
from routing_qaoa.qaoa import build_qaoa_main, initial_qaoa_params


def test_qaoa_main_builds_model(micro2) -> None:
    from classiq import create_model

    cost_fn = build_cost_function(compute_coefficients(micro2))
    main = build_qaoa_main(cost_fn, num_qubits=micro2.num_qubits, num_layers=2)
    assert create_model(main)  # serializes the qmod; fails on any symbolic issue


def test_initial_qaoa_params() -> None:
    params = initial_qaoa_params(3)
    assert params.shape == (6,)
    gammas, betas = params[0::2], params[1::2]
    assert np.all(np.diff(gammas) > 0)
    assert np.all(np.diff(betas) < 0)
    assert np.all((params > 0) & (params < math.pi))
    assert np.allclose(gammas, betas[::-1])
