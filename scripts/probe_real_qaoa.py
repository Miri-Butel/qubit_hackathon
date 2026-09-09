"""Probe whether the real AT&T instances synthesize and execute on Classiq.

The deleted notebook cell claimed "28 qubits, exactly the simulator limit" as a
hardcoded string, never a measurement. This measures it: synthesize at one QAOA
layer, report circuit width/depth/gate counts, then attempt a small sample.

Run before committing the pitch to a 28-qubit result.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "network_instance"))

from classiq import ExecutionSession, synthesize  # noqa: E402

from routing_qaoa import QuboWeights  # noqa: E402
from routing_qaoa.qaoa import build_qaoa_main, initial_qaoa_params  # noqa: E402
from routing_qaoa.qubo import build_cost_function, compute_coefficients  # noqa: E402

import att_real as A  # noqa: E402
import export  # noqa: E402
import yen  # noqa: E402

WEIGHTS = QuboWeights(congestion_profile="fortz-thorup-fit", cost_scale=0.1)


def circuit_stats(qprog) -> dict:
    """Pull width/depth/gate counts defensively across classiq versions."""
    stats = {}
    data = getattr(qprog, "data", None)
    for name in ("width", "depth"):
        value = getattr(data, name, None)
        if value is not None:
            stats[name] = value
    transpiled = getattr(qprog, "transpiled_circuit", None)
    if transpiled is not None:
        for name in ("depth", "count_ops"):
            value = getattr(transpiled, name, None)
            if value is not None:
                stats[f"transpiled_{name}"] = value
    return stats


def probe(region: str, num_layers: int = 1, num_shots: int = 100) -> None:
    inst = A.att_backbone(region=region)
    extra = 2 if region == "east10" else 0
    candidates = yen.budgeted_candidate_set(inst, k=5, base=2, extra_for=extra)
    instance, _ = export.to_routing_instance(inst, candidates)
    print(f"\n=== {region}: {instance.num_qubits} qubits, {num_layers} layer(s) ===")

    coeffs = compute_coefficients(instance, WEIGHTS)
    cost_fn = build_cost_function(coeffs)

    start = time.perf_counter()
    try:
        qprog = synthesize(
            build_qaoa_main(cost_fn, instance.num_qubits, num_layers)
        )
    except Exception as exc:  # noqa: BLE001 - probe reports, never raises
        print(f"SYNTHESIS FAILED after {time.perf_counter() - start:.1f}s")
        print(f"  {type(exc).__name__}: {exc}")
        return
    synth_s = time.perf_counter() - start
    print(f"synthesis OK in {synth_s:.1f}s -> {circuit_stats(qprog)}")

    start = time.perf_counter()
    try:
        with ExecutionSession(qprog, num_shots=num_shots, random_seed=42) as es:
            frame = es.sample({"params": initial_qaoa_params(num_layers).tolist()})
        rows = len(frame) if hasattr(frame, "__len__") else "?"
        print(
            f"execution OK in {time.perf_counter() - start:.1f}s "
            f"({num_shots} shots, {rows} distinct bitstrings)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"EXECUTION FAILED after {time.perf_counter() - start:.1f}s")
        print(f"  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    regions = sys.argv[1:] or ["east10", "east10_dense"]
    for region in regions:
        probe(region)
