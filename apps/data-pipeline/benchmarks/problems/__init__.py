"""Benchmark problem registry.

All RecoveryProblem instances must be registered in ALL_PROBLEMS so they are
automatically covered by the identifiability gate in
tests/test_benchmark_identifiability.py.
"""

from benchmarks.problems.four_latent import FOUR_LATENT
from benchmarks.problems.three_latent_robust import THREE_LATENT_ROBUST

ALL_PROBLEMS: dict = {
    "four_latent": FOUR_LATENT,
    "three_latent_robust": THREE_LATENT_ROBUST,
}
