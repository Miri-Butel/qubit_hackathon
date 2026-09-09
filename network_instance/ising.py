"""
QUBO -> Ising conversion for the routing Hamiltonian.

`routing_qaoa.qubo` builds the cost as a polynomial in binary variables and
hands it straight to Classiq's `phase()`, which is all QAOA needs. The Ising
form is never constructed there. It is written here because it is the form
the physics is usually stated in, and because seeing h and J makes the
problem's structure legible: which routes are pushed on individually, and
which pairs of routes actually interact.

The QUBO, after expanding the squares in `CostCoefficients`, is

    H(x) = c + sum_i L_i x_i + sum_{i<j} Q_ij x_i x_j ,      x_i in {0,1}

Binary variables are idempotent, x_i^2 = x_i, so every squared term folds
into the linear part. Substituting the standard change of variable

    x_i = (1 - z_i) / 2 ,                                    z_i in {-1,+1}

and collecting terms gives

    H(z) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j

with

    h_i   = -L_i/2 - (1/4) sum_{j != i} Q_ij
    J_ij  =  Q_ij / 4
    offset = c + (1/2) sum_i L_i + (1/4) sum_{i<j} Q_ij

The sign convention follows from x = (1-z)/2, so z_i = +1 means the route is
*not* selected and z_i = -1 means it is.
"""

import numpy as np


def qubo_matrix(coeffs, num_qubits: int):
    """Expand `routing_qaoa.CostCoefficients` into (Q, L, c).

    Q is strictly upper triangular (pair couplings), L is the linear vector,
    c the constant. The two squared groups in the cost are expanded here:

    * congestion, `cong_scale * (sum_i a_i x_i)^2` for each link, contributes
      `a_i^2` to L (via x^2 = x) and `2 a_i a_j` to Q.
    * one-hot, `lambda * (sum_{i in g} x_i - 1)^2` for each demand,
      contributes `-lambda` to L, `2 lambda` to Q, and `+lambda` to c.
    """
    Q = np.zeros((num_qubits, num_qubits))
    L = np.array(coeffs.linear, dtype=float)
    c = 0.0

    for terms in coeffs.link_terms:
        for a, (i, ai) in enumerate(terms):
            L[i] += coeffs.cong_scale * ai * ai
            for j, aj in terms[a + 1:]:
                lo, hi = (i, j) if i < j else (j, i)
                Q[lo, hi] += 2.0 * coeffs.cong_scale * ai * aj

    lam = coeffs.lambda_onehot_scaled
    for group in coeffs.onehot_groups:
        c += lam
        for a, i in enumerate(group):
            L[i] -= lam
            for j in group[a + 1:]:
                lo, hi = (i, j) if i < j else (j, i)
                Q[lo, hi] += 2.0 * lam
    return Q, L, c


def to_ising(Q: np.ndarray, L: np.ndarray, c: float = 0.0):
    """(Q, L, c) over x in {0,1}  ->  (h, J, offset) over z in {-1,+1}."""
    n = len(L)
    J = Q / 4.0
    sym = Q + Q.T  # so row i sums every coupling touching i, both orientations
    h = -L / 2.0 - sym.sum(axis=1) / 4.0
    offset = c + L.sum() / 2.0 + Q.sum() / 4.0
    return h, J, offset


def ising_energy(h: np.ndarray, J: np.ndarray, offset: float, z: np.ndarray) -> float:
    return float(offset + h @ z + z @ np.triu(J, 1) @ z)


def qubo_energy(Q: np.ndarray, L: np.ndarray, c: float, x: np.ndarray) -> float:
    return float(c + L @ x + x @ np.triu(Q, 1) @ x)


def verify(coeffs, num_qubits: int, trials: int = 200, seed: int = 0) -> dict:
    """Check the two forms agree, and that both agree with routing_qaoa's own
    cost function, on random assignments. Returns the worst absolute gaps."""
    from routing_qaoa.qubo import build_cost_function

    rng = np.random.default_rng(seed)
    cost = build_cost_function(coeffs)
    Q, L, c = qubo_matrix(coeffs, num_qubits)
    h, J, offset = to_ising(Q, L, c)

    worst_qubo = worst_ising = 0.0
    for _ in range(trials):
        x = rng.integers(0, 2, num_qubits)
        reference = float(cost([int(v) for v in x]))
        worst_qubo = max(worst_qubo, abs(qubo_energy(Q, L, c, x) - reference))
        z = 1 - 2 * x  # x = (1 - z)/2
        worst_ising = max(worst_ising, abs(ising_energy(h, J, offset, z) - reference))
    return {
        "max_error_qubo_vs_routing_qaoa": worst_qubo,
        "max_error_ising_vs_routing_qaoa": worst_ising,
        "num_couplings": int((np.abs(np.triu(J, 1)) > 1e-12).sum()),
        "num_fields": int((np.abs(h) > 1e-12).sum()),
    }
