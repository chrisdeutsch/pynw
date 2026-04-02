//! PyO3 binding layer for the `pynw._native` extension module.

use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, get_array_module,
};
use pyo3::{intern, prelude::*, sync::PyOnceLock, types::PyDict};

mod nw;
mod nw_merge_split;

#[pymodule(name = "_native")]
mod pynw_native {
    use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("OP_ALIGN", nw::EditOp::Align as u8)?;
        m.add("OP_INSERT", nw::EditOp::Insert as u8)?;
        m.add("OP_DELETE", nw::EditOp::Delete as u8)?;
        Ok(())
    }

    // NOTE: This doc comment provides the runtime `help()` docstring.
    // A copy exists in pynw/_native.pyi for type checkers and IDE support.
    // Keep both copies in sync.
    //
    /// Align two ordered sequences given a precomputed similarity matrix.
    ///
    /// The total alignment score is the sum of similarity-matrix entries for
    /// matched positions and gap penalties for insertions/deletions.
    ///
    /// Parameters
    /// ----------
    /// similarity_matrix : array_like, shape (n, m)
    ///     ``similarity_matrix[i, j]`` is the similarity score for aligning
    ///     element *i* of the source sequence with element *j* of the target
    ///     sequence.
    /// gap_penalty : float, default -1.0
    ///     Penalty applied when a gap is inserted in either sequence.
    ///     Use ``insert_penalty`` or ``delete_penalty`` to set them
    ///     independently.
    /// insert_penalty : float, optional
    ///     Penalty for advancing the target sequence without consuming a source
    ///     element (gap in source).  Defaults to ``gap_penalty``.
    /// delete_penalty : float, optional
    ///     Penalty for advancing the source sequence without consuming a target
    ///     element (gap in target).  Defaults to ``gap_penalty``.
    /// Raises
    /// ------
    /// ValueError
    ///     If ``similarity_matrix`` is not 2-dimensional, or if any value in
    ///     ``similarity_matrix`` or the gap penalties is ``NaN`` or ``Inf``.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     The optimal alignment score.
    /// ops : ndarray of uint8, shape (k,)
    ///     Sequence of edit operations describing the alignment.  Each element
    ///     is of type ``EditOp``.  Use
    ///     ``iter_alignment`` to iterate over aligned element pairs, or
    ///     ``indices_from_ops`` to reconstruct source and target index arrays.
    ///
    /// Examples
    /// --------
    /// Align two DNA sequences using a simple match/mismatch scoring scheme:
    ///
    /// >>> import numpy as np
    /// >>> source_seq = list("GATTACA")
    /// >>> target_seq = list("GCATGCA")
    /// >>> match, mismatch = 1.0, -1.0
    /// >>> sm = np.where(
    /// ...     np.array(source_seq)[:, None] == np.array(target_seq)[None, :],
    /// ...     match, mismatch,
    /// ... )
    /// >>> from pynw import iter_alignment
    /// >>> score, ops = needleman_wunsch(sm, gap_penalty=-1.0)
    /// >>> score
    /// 2.0
    /// >>> cols = [
    /// ...     (s or "-", t or "-")
    /// ...     for _, s, t in iter_alignment(ops, source_seq, target_seq)
    /// ... ]
    /// >>> "".join(s for s, _ in cols)
    /// 'G-ATTACA'
    /// >>> "".join(t for _, t in cols)
    /// 'GCA-TGCA'
    ///
    /// Notes
    /// -----
    /// When multiple alignments achieve the same optimal score, ties are
    /// broken deterministically: ``Diagonal > Up > Left``.  This prefers
    /// substitutions over gaps, producing compact alignments.  Other tools
    /// may return different co-optimal alignments.
    ///
    #[pyfunction]
    #[pyo3(
        signature = (similarity_matrix, *, gap_penalty=-1.0, insert_penalty=None, delete_penalty=None),
        text_signature = "(similarity_matrix, *, gap_penalty=-1.0, insert_penalty=None, delete_penalty=None)",
    )]
    fn needleman_wunsch<'py>(
        py: Python<'py>,
        similarity_matrix: Bound<'py, PyAny>,
        gap_penalty: f64,
        insert_penalty: Option<f64>,
        delete_penalty: Option<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<u8>>)> {
        let py_array = as_pyarray(py, &similarity_matrix)?;
        let similarity_matrix = py_array.as_array();

        let insert_penalty = insert_penalty.unwrap_or(gap_penalty);
        let delete_penalty = delete_penalty.unwrap_or(gap_penalty);

        if !insert_penalty.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "insert_penalty is non-finite",
            ));
        }
        if !delete_penalty.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "delete_penalty is non-finite",
            ));
        }
        if !similarity_matrix.iter().all(|v: &f64| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "similarity_matrix contains non-finite values (NaN or Inf)",
            ));
        }

        let (score, ops) = nw::needleman_wunsch(similarity_matrix, insert_penalty, delete_penalty);

        let ops: Vec<u8> = ops.into_iter().map(|op| op as u8).collect();
        Ok((score, ops.into_pyarray(py)))
    }

    // NOTE: This doc comment provides the runtime `help()` docstring.
    // A copy exists in pynw/_native.pyi for type checkers and IDE support.
    // Keep both copies in sync.
    //
    /// Align two sequences with one-to-one, one-to-two (split), and
    /// two-to-one (merge) matches.
    ///
    /// Extends Needleman-Wunsch with two additional edit operations: a row
    /// element can be *split* to align with two consecutive column elements, or
    /// two consecutive row elements can be *merged* to align with one column
    /// element.  The score for each operation at each position is supplied by
    /// the caller via separate score matrices.
    ///
    /// Parameters
    /// ----------
    /// align_scores : array_like, shape (n, m)
    ///     ``align_scores[i, j]`` is the score for aligning row element *i*
    ///     with column element *j* (one-to-one match).
    /// split_scores : array_like, shape (n, m-1)
    ///     ``split_scores[i, j]`` is the score for splitting row element *i*
    ///     across column elements *j* and *j+1* (one row → two columns).
    /// merge_scores : array_like, shape (n-1, m)
    ///     ``merge_scores[i, j]`` is the score for merging row elements *i*
    ///     and *i+1* into column element *j* (two rows → one column).
    /// gap_penalty : float, default -1.0
    ///     Penalty for an insert or delete step.  Use ``insert_penalty`` or
    ///     ``delete_penalty`` to set them independently.
    /// insert_penalty : float, optional
    ///     Penalty for advancing the column sequence without consuming a row
    ///     element.  Defaults to ``gap_penalty``.
    /// delete_penalty : float, optional
    ///     Penalty for advancing the row sequence without consuming a column
    ///     element.  Defaults to ``gap_penalty``.
    /// Raises
    /// ------
    /// ValueError
    ///     If any score matrix is not 2-dimensional, or if any value is non-finite.
    ///
    /// Returns
    /// -------
    /// score : float
    ///     The optimal alignment score.
    /// ops : ndarray of uint8, shape (k,)
    ///     Sequence of edit operations describing the alignment.  Each element
    ///     is one of the ``OP_*`` constants (or ``Op`` enum values).  Use
    ///     ``indices_from_ops`` to reconstruct the row and column indices.
    ///
    ///     +---------------+-----------------------------------+
    ///     | Op            | Meaning                           |
    ///     +===============+===================================+
    ///     | ``OP_ALIGN``  | row[i] aligned with col[j]        |
    ///     | ``OP_INSERT`` | gap in row; col[j] consumed        |
    ///     | ``OP_DELETE`` | row[i] consumed; gap in col        |
    ///     | ``OP_SPLIT``  | row[i] split into col[j], col[j+1]|
    ///     | ``OP_MERGE``  | row[i], row[i+1] merged into col[j]|
    ///     +---------------+-----------------------------------+
    ///
    /// Notes
    /// -----
    /// Ties are broken deterministically: ``Align > Merge > Split > Delete > Insert``.
    ///
    #[pyfunction]
    #[pyo3(
        signature = (align_scores, split_scores, merge_scores, *, gap_penalty=-1.0, insert_penalty=None, delete_penalty=None),
        text_signature = "(align_scores, split_scores, merge_scores, *, gap_penalty=-1.0, insert_penalty=None, delete_penalty=None)",
    )]
    #[allow(clippy::too_many_arguments)]
    fn needleman_wunsch_merge_split<'py>(
        py: Python<'py>,
        align_scores: Bound<'py, PyAny>,
        split_scores: Bound<'py, PyAny>,
        merge_scores: Bound<'py, PyAny>,
        gap_penalty: f64,
        insert_penalty: Option<f64>,
        delete_penalty: Option<f64>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<u8>>)> {
        let py_array = as_pyarray(py, &align_scores)?;
        let align_scores = py_array.as_array();

        let py_array = as_pyarray(py, &split_scores)?;
        let split_scores = py_array.as_array();

        let py_array = as_pyarray(py, &merge_scores)?;
        let merge_scores = py_array.as_array();

        let insert_penalty = insert_penalty.unwrap_or(gap_penalty);
        let delete_penalty = delete_penalty.unwrap_or(gap_penalty);

        if !insert_penalty.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "insert_penalty is non-finite",
            ));
        }
        if !delete_penalty.is_finite() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "delete_penalty is non-finite",
            ));
        }
        if !align_scores.iter().all(|v: &f64| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "align_scores contains non-finite values (NaN or Inf)",
            ));
        }
        if !split_scores.iter().all(|v: &f64| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "split_scores contains non-finite values (NaN or Inf)",
            ));
        }
        if !merge_scores.iter().all(|v: &f64| v.is_finite()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "merge_scores contains non-finite values (NaN or Inf)",
            ));
        }

        let (score, ops) = nw_merge_split::needleman_wunsch_merge_split(
            align_scores,
            split_scores,
            merge_scores,
            insert_penalty,
            delete_penalty,
        );

        let ops: Vec<u8> = ops.into_iter().map(|op| op as u8).collect();
        Ok((score, ops.into_pyarray(py)))
    }
}

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
