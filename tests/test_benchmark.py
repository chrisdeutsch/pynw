import string

import numpy as np
import pytest

from pynw import needleman_wunsch
from tests.helpers import char_score_matrix

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


def _random_string_pair(length: int) -> tuple[str, str]:
    """Generate a pair of random strings of the given length."""
    rng = np.random.default_rng(RNG_SEED + length)
    alphabet = list(string.ascii_lowercase)
    s1 = "".join(rng.choice(alphabet, size=length))
    s2 = "".join(rng.choice(alphabet, size=length))
    return s1, s2


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


# ---------------------------------------------------------------------------
# Needleman-Wunsch vs rapidfuzz editops (Levenshtein special case)
# ---------------------------------------------------------------------------
# Both produce an alignment trace (not just a scalar distance), so the
# comparison measures end-to-end cost including matrix construction for pynw.


def _nw_levenshtein(s1: str, s2: str) -> None:
    """Run NW configured as Levenshtein: match=0, mismatch=-1, gap=-1."""
    sm = char_score_matrix(s1, s2, match=0, mismatch=-1)
    needleman_wunsch(sm, gap_penalty=-1)


@pytest.mark.parametrize("length", STRING_LENGTHS, ids=lambda n: f"len{n}")
def test_benchmark_nw_levenshtein(benchmark, length: int) -> None:
    s1, s2 = _random_string_pair(length)
    benchmark.group = f"levenshtein len={length}"
    benchmark(_nw_levenshtein, s1, s2)


@pytest.mark.parametrize("length", STRING_LENGTHS, ids=lambda n: f"len{n}")
def test_benchmark_rapidfuzz_editops(benchmark, length: int) -> None:
    s1, s2 = _random_string_pair(length)
    benchmark.group = f"levenshtein len={length}"
    benchmark(rapidfuzz_distance.Levenshtein.editops, s1, s2)


# ---------------------------------------------------------------------------
# C-order vs Fortran-order similarity matrices
# ---------------------------------------------------------------------------


def _score_matrix_c(size: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + size)
    return np.ascontiguousarray(rng.standard_normal((size, size), dtype=np.float64))


def _score_matrix_f(size: int) -> np.ndarray:
    rng = np.random.default_rng(RNG_SEED + size)
    return np.asfortranarray(rng.standard_normal((size, size), dtype=np.float64))


@pytest.mark.parametrize("size", MATRIX_SIZES, ids=lambda n: f"{n}x{n}")
def test_benchmark_needleman_wunsch_c_order(benchmark, size: int) -> None:
    score_matrix = _score_matrix_c(size)
    benchmark.group = f"memory order {size}x{size}"
    benchmark(needleman_wunsch, score_matrix)


@pytest.mark.parametrize("size", MATRIX_SIZES, ids=lambda n: f"{n}x{n}")
def test_benchmark_needleman_wunsch_f_order(benchmark, size: int) -> None:
    score_matrix = _score_matrix_f(size)
    benchmark.group = f"memory order {size}x{size}"
    benchmark(needleman_wunsch, score_matrix)
