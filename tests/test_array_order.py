import numpy as np

from pynw import needleman_wunsch

DEFAULT_RNG = np.random.default_rng(20260329)


def test_c_and_fortran():
    array_c = DEFAULT_RNG.normal(size=(50, 50))
    assert array_c.flags.c_contiguous

    array_f = np.asfortranarray(array_c)
    assert array_f.flags.f_contiguous

    score_c, row_idx_c, col_idx_c = needleman_wunsch(array_c)
    score_f, row_idx_f, col_idx_f = needleman_wunsch(array_f)

    assert score_c == score_f
    np.testing.assert_equal(row_idx_c, row_idx_f, strict=True)
    np.testing.assert_equal(col_idx_c, col_idx_f, strict=True)
