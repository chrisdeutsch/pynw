//! PyO3 binding layer for the `pynw._native` extension module.

use numpy::{
    Element, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    get_array_module,
};
use pyo3::{intern, prelude::*, sync::PyOnceLock, types::PyDict};

mod nw;

#[pymodule(name = "_native")]
mod pynw_native {
    use num_enum::TryFromPrimitiveError;

    use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("OP_ALIGN", u8::from(nw::EditOp::Align))?;
        m.add("OP_INSERT", u8::from(nw::EditOp::Insert))?;
        m.add("OP_DELETE", u8::from(nw::EditOp::Delete))?;
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
    ///     ``alignment_indices`` to reconstruct source and target index arrays.
    ///
    /// Examples
    /// --------
    /// Align two DNA sequences using a simple match/mismatch scoring scheme:
    ///
    /// >>> import numpy as np
    /// >>> from pynw import alignment_indices
    /// >>> seq1 = np.array(list("GATTACA"))
    /// >>> seq2 = np.array(list("GCATGCA"))
    /// >>> sm = np.where(seq1[:, None] == seq2[None, :], 1.0, -1.0)
    /// >>> score, ops = needleman_wunsch(sm, gap_penalty=-1.0)
    /// >>> score
    /// 2.0
    /// >>> src_idx, tgt_idx = alignment_indices(ops)
    /// >>> "".join(np.ma.array(seq1).take(src_idx).filled("-"))
    /// 'G-ATTACA'
    /// >>> "".join(np.ma.array(seq2).take(tgt_idx).filled("-"))
    /// 'GCA-TGCA'
    ///
    /// Notes
    /// -----
    /// When multiple alignments achieve the same optimal score, ties are
    /// broken deterministically: ``Align > Delete > Insert``.  This prefers
    /// substitutions over gaps, producing compact alignments.  Other tools
    /// may return different co-optimal alignments.
    ///
    /// All values in ``similarity_matrix`` and the gap penalties must be finite.
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
        let py_array = to_array2_f64(py, &similarity_matrix)?;
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

        let ops: Vec<u8> = ops.into_iter().map(Into::into).collect();
        Ok((score, ops.into_pyarray(py)))
    }

    type AlignmentIndicesResult<'py> = (
        Bound<'py, PyArray1<isize>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<isize>>,
        Bound<'py, PyArray1<bool>>,
    );

    #[pyfunction]
    #[pyo3(signature = (ops), text_signature = "(ops)")]
    fn alignment_indices<'py>(
        py: Python<'py>,
        ops: Bound<'py, PyAny>,
    ) -> PyResult<AlignmentIndicesResult<'py>> {
        let ops: Vec<nw::EditOp> = to_array1_u8(py, &ops)?
            .as_slice()?
            .iter()
            .map(|&x| {
                nw::EditOp::try_from(x).map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("Cannot convert u8 into EditOp")
                })
            })
            .collect::<PyResult<_>>()?;

        let (source, target) = nw::alignment_indices((&ops).into());

        Ok((
            source.indices.into_pyarray(py),
            source.mask.into_pyarray(py),
            target.indices.into_pyarray(py),
            target.mask.into_pyarray(py),
        ))
    }
}

// Modified version of the PyArrayLike extract method. Calls numpy.asarray(obj, dtype=dtype).
fn numpy_asarray<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    static AS_ARRAY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let as_array = AS_ARRAY
        .get_or_try_init(py, || {
            get_array_module(py)?.getattr("asarray").map(Into::into)
        })?
        .bind(py);
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "dtype"), dtype)?;
    as_array.call((obj,), Some(&kwargs))
}

fn to_array2_f64<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
    if let Ok(array) = obj.cast::<PyArray2<f64>>() {
        // TODO: Check that array is C-contiguous?
        return Ok(array.readonly());
    }
    numpy_asarray(py, obj, f64::get_dtype(py).into_any())?
        .extract()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into 2-dimensional float64 array",
            )
        })
}

fn to_array1_u8<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<PyReadonlyArray1<'py, u8>> {
    if let Ok(array) = obj.cast::<PyArray1<u8>>() {
        return Ok(array.readonly());
    }
    numpy_asarray(py, obj, u8::get_dtype(py).into_any())?
        .extract()
        .map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into 1-dimensional u8 array",
            )
        })
}
