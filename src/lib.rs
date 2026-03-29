//! PyO3 binding layer for the `pynw._native` extension module.

use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, get_array_module,
};
use pyo3::{intern, prelude::*, sync::PyOnceLock, types::PyDict};

mod nw_core;

fn as_pyarray<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    // This is a modified version of the extract method of PyArrayLike to convert into a 2d f64 array

    if let Ok(array) = obj.cast::<PyArray2<f64>>() {
        // TODO: Check that array is C-contiguous?
        return Ok(array.readonly());
    }

    static AS_ARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    let as_array = AS_ARRAY
        .get_or_try_init(py, || {
            get_array_module(py)?.getattr("asarray").map(Into::into)
        })?
        .bind(py);

    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "dtype"), f64::get_dtype(py))?;

    let array = as_array
        .call((obj,), Some(kwargs).as_ref())?
        .extract()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into 2-dimensional float64 array",
            )
        })?;

    Ok(array)
}

type NeedlemanWunschResultType<'py> = (
    f64,
    Bound<'py, PyArray1<isize>>,
    Bound<'py, PyArray1<isize>>,
);

#[pymodule(name = "_native")]
mod pynw_native {
    use super::*;

    /// Align two ordered sequences given a precomputed similarity matrix.
    ///
    /// The total alignment score is the sum of similarity-matrix entries for
    /// matched positions and gap penalties for insertions/deletions.
    ///
    /// Parameters
    /// ----------
    /// similarity_matrix : array_like, shape (n, m)
    ///     ``similarity_matrix[i, j]`` is the similarity score for aligning
    ///     element *i* of the row sequence with element *j* of the column
    ///     sequence.
    /// gap_penalty : float, default -1.0
    ///     Penalty applied when a gap is inserted in either sequence.
    ///     Use ``gap_penalty_row`` or ``gap_penalty_col`` to specify
    ///     different penalties for each sequence.
    /// gap_penalty_row : float, optional
    ///     Penalty added when a gap is inserted in the row sequence
    ///     (the column sequence advances). This can be thought of as the
    ///     cost of a deletion from the row sequence or an insertion into the
    ///     column sequence.
    ///     Defaults to ``gap_penalty`` if not specified.
    /// gap_penalty_col : float, optional
    ///     Penalty added when a gap is inserted in the column sequence
    ///     (the row sequence advances). This can be thought of as the cost
    ///     of a deletion from the column sequence or an insertion into the
    ///     row sequence.
    ///     Defaults to ``gap_penalty`` if not specified.
    /// check_finite : bool, default False
    ///     If ``True``, raise a ``ValueError`` when ``similarity_matrix``
    ///     or the gap penalties contain ``NaN`` or ``Inf``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``similarity_matrix`` is not 2-dimensional, or if
    ///     ``check_finite`` is ``True`` and any value in
    ///     ``similarity_matrix`` or the gap penalties is ``NaN`` or ``Inf``.
    ///
    /// Returns
    /// -------
    /// NeedlemanWunschResult
    ///     A named tuple with fields ``score``, ``row_idx``, and ``col_idx``.
    ///     ``row_idx`` and ``col_idx`` map each alignment position to an
    ///     index in the original sequence, with ``-1`` indicating a gap.
    ///
    /// Examples
    /// --------
    /// Align two DNA sequences using a simple match/mismatch scoring scheme:
    ///
    /// >>> import numpy as np
    /// >>> row_seq = list("GATTACA")
    /// >>> col_seq = list("GCATGCA")
    /// >>> match, mismatch = 1.0, -1.0
    /// >>> sm = np.where(
    /// ...     np.array(row_seq)[:, None] == np.array(col_seq)[None, :],
    /// ...     match, mismatch,
    /// ... )
    /// >>> result = needleman_wunsch(sm, gap_penalty=-1.0)
    /// >>> result.score
    /// 2.0
    /// >>> "".join(row_seq[i] if i >= 0 else "-" for i in result.row_idx)
    /// 'G-ATTACA'
    /// >>> "".join(col_seq[i] if i >= 0 else "-" for i in result.col_idx)
    /// 'GCA-TGCA'
    ///
    /// Notes
    /// -----
    /// When multiple alignments achieve the same optimal score, ties are
    /// broken deterministically: ``Diagonal > Up > Left``.  This prefers
    /// substitutions over gaps, producing compact alignments.  Other tools
    /// may return different co-optimal alignments.
    ///
    /// All values in ``similarity_matrix`` and the gap penalties must be finite.
    /// Passing ``NaN`` or ``Inf`` is undefined behavior — the output will be
    /// silently meaningless.
    #[pyfunction]
    #[pyo3(
        signature = (similarity_matrix, *, gap_penalty=-1.0, gap_penalty_row=None, gap_penalty_col=None, check_finite=false),
        text_signature = "(similarity_matrix, *, gap_penalty=-1.0, gap_penalty_row=None, gap_penalty_col=None, check_finite=False)",
    )]
    fn needleman_wunsch<'py>(
        py: Python<'py>,
        similarity_matrix: Bound<'py, PyAny>,
        gap_penalty: f64,
        gap_penalty_row: Option<f64>,
        gap_penalty_col: Option<f64>,
        check_finite: bool,
    ) -> PyResult<NeedlemanWunschResultType<'py>> {
        let py_array = as_pyarray(py, &similarity_matrix)?;
        let similarity_matrix = py_array.as_array();

        let gap_penalty_row = gap_penalty_row.unwrap_or(gap_penalty);
        let gap_penalty_col = gap_penalty_col.unwrap_or(gap_penalty);

        if check_finite {
            if !gap_penalty_row.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "gap_penalty_row is non-finite",
                ));
            }
            if !gap_penalty_col.is_finite() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "gap_penalty_col is non-finite",
                ));
            }
            if !similarity_matrix.iter().all(|v: &f64| v.is_finite()) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "similarity_matrix contains non-finite values (NaN or Inf)",
                ));
            }
        }

        let (score, row_idx, col_idx) = nw_core::nw_traceback_indices_core(
            &similarity_matrix,
            gap_penalty_row,
            gap_penalty_col,
        );

        Ok((score, row_idx.into_pyarray(py), col_idx.into_pyarray(py)))
    }
}
