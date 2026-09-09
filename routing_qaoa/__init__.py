"""QAOA-based path selection for network traffic routing (Classiq hackathon).

Pure-python API (model/qubo/decode) imports eagerly; the Classiq solver
(`run_qaoa` etc.) and the matplotlib plots (`plot_routing` etc.) load lazily
on first access so offline work never pays the SDK/plotting imports.
"""

from importlib import import_module

from .decode import (
    CONGESTION_THRESHOLD_PHI_STAR,
    LinkKpi,
    RoutingSolution,
    best_feasible_solution,
    chosen_paths,
    decode_bitstring,
    feasible_probability,
    hop_distance,
    onehot_feasible,
    phi_uncap,
    top_solutions,
)
from .model import CandidatePath, Demand, Link, RoutingInstance
from .qubo import (
    BruteForceResult,
    CostCoefficients,
    QuboWeights,
    brute_force_feasible,
    build_cost_function,
    compute_coefficients,
    fortz_thorup_link_cost,
    fortz_thorup_quadratic_fit,
)

_LAZY_EXPORTS = {
    **dict.fromkeys(
        ("QaoaConfig", "QaoaResult", "build_qaoa_main", "initial_qaoa_params", "run_qaoa"),
        "qaoa",
    ),
    **dict.fromkeys(
        ("network_layout", "plot_link_utilization", "plot_network", "plot_routing", "to_networkx"),
        "viz",
    ),
}

__all__ = [
    "CONGESTION_THRESHOLD_PHI_STAR",
    "BruteForceResult",
    "CandidatePath",
    "CostCoefficients",
    "Demand",
    "Link",
    "LinkKpi",
    "QaoaConfig",
    "QaoaResult",
    "QuboWeights",
    "RoutingInstance",
    "RoutingSolution",
    "best_feasible_solution",
    "brute_force_feasible",
    "build_cost_function",
    "build_qaoa_main",
    "chosen_paths",
    "compute_coefficients",
    "decode_bitstring",
    "feasible_probability",
    "fortz_thorup_link_cost",
    "fortz_thorup_quadratic_fit",
    "hop_distance",
    "initial_qaoa_params",
    "network_layout",
    "onehot_feasible",
    "phi_uncap",
    "plot_link_utilization",
    "plot_network",
    "plot_routing",
    "run_qaoa",
    "to_networkx",
    "top_solutions",
]


def __getattr__(name: str):
    if module_name := _LAZY_EXPORTS.get(name):
        module = import_module(f".{module_name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
