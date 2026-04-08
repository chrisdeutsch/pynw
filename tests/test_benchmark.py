import numpy as np
import pytest

from pynw import alignment_indices, needleman_wunsch, needleman_wunsch_score
from pynw._ops import EditOp

scipy_optimize = pytest.importorskip("scipy.optimize")
rapidfuzz_distance = pytest.importorskip("rapidfuzz.distance")
pytestmark = pytest.mark.benchmark

MATRIX_SIZES = [5, 10, 25, 50, 100, 250, 500, 1000]
STRING_LENGTHS = [5, 10, 25, 50, 100, 250, 500, 1000]
OPS_LENGTHS = [10, 100, 1000, 10_000, 100_000]
RNG_SEED = 20260322


def _ops_array(length: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + length)
    op_values = [EditOp.Align, EditOp.Insert, EditOp.Delete]
    return rng.choice(op_values, size=length).astype(np.uint8)


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
    benchmark(needleman_wunsch, score_matrix, gap_penalty=-1.0)


@pytest.mark.parametrize("size", MATRIX_SIZES, ids=lambda n: f"{n}x{n}")
def test_benchmark_needleman_wunsch_score(benchmark, size: int) -> None:
    score_matrix = _score_matrix(size)
    benchmark.group = f"matrix {size}x{size}"
    benchmark(needleman_wunsch_score, score_matrix, gap_penalty=-1.0)


@pytest.mark.parametrize("size", MATRIX_SIZES, ids=lambda n: f"{n}x{n}")
def test_benchmark_linear_sum_assignment(benchmark, size: int) -> None:
    cost_matrix = _cost_matrix(size)
    benchmark.group = f"matrix {size}x{size}"
    benchmark(scipy_optimize.linear_sum_assignment, cost_matrix)


# ---------------------------------------------------------------------------
# alignment_indices: python vs native
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length", OPS_LENGTHS, ids=lambda n: f"n={n}")
def test_benchmark_alignment_indices(benchmark, length: int) -> None:
    ops = _ops_array(length)
    benchmark.group = f"alignment_indices n={length}"
    benchmark(alignment_indices, ops)
