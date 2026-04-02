import numpy as np
import pytest

from pynw import needleman_wunsch

scipy_optimize = pytest.importorskip("scipy.optimize")
rapidfuzz_distance = pytest.importorskip("rapidfuzz.distance")
pytestmark = pytest.mark.benchmark

MATRIX_SIZES = [5, 10, 25, 50, 100, 250, 500, 1000]
STRING_LENGTHS = [5, 10, 25, 50, 100, 250, 500, 1000]
RNG_SEED = 20260322


def _score_matrix(size: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + size)
    return rng.standard_normal((size, size), dtype=np.float64)


def _cost_matrix(size: int) -> np.ndarray:
    # linear_sum_assignment minimizes cost; negate scores for a comparable input.
    return -_score_matrix(size)


# ---------------------------------------------------------------------------
# Needleman-Wunsch vs linear sum assignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", MATRIX_SIZES, ids=lambda n: f"{n}x{n}")
def test_benchmark_needleman_wunsch(benchmark, size: int) -> None:
    score_matrix = _score_matrix(size)
    benchmark.group = f"matrix {size}x{size}"
    benchmark(needleman_wunsch, score_matrix)


@pytest.mark.parametrize("size", MATRIX_SIZES, ids=lambda n: f"{n}x{n}")
def test_benchmark_linear_sum_assignment(benchmark, size: int) -> None:
    cost_matrix = _cost_matrix(size)
    benchmark.group = f"matrix {size}x{size}"
    benchmark(scipy_optimize.linear_sum_assignment, cost_matrix)
