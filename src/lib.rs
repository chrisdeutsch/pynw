use pyo3::pymodule;

mod nw;
mod utils;

#[pymodule(name = "_native")]
mod pynw_native {
    use super::{nw, utils::*};
    use numpy::{IntoPyArray, PyArray1};
    use pyo3::prelude::*;

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
    /// editops : ndarray of uint8, shape (k,)
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
    /// >>> score, editops = needleman_wunsch(sm, gap_penalty=-1.0)
    /// >>> score
    /// 2.0
    /// >>> src_idx, tgt_idx = alignment_indices(editops)
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
        let pyarray = to_pyreadonly(py, similarity_matrix).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into 2-dimensional float64 array",
            )
        })?;
        let similarity_matrix = pyarray.as_array();

        let insert_penalty = insert_penalty.unwrap_or(gap_penalty);
        let delete_penalty = delete_penalty.unwrap_or(gap_penalty);

        validate_inputs(similarity_matrix, insert_penalty, delete_penalty)?;

        let (score, editops) =
            nw::needleman_wunsch(similarity_matrix, insert_penalty, delete_penalty);
        Ok((score, editops.mapv(Into::into).into_pyarray(py)))
    }

    // NOTE: This doc comment provides the runtime `help()` docstring.
    // A copy exists in pynw/_native.pyi for type checkers and IDE support.
    // Keep both copies in sync.
    //
    /// Compute the optimal Needleman-Wunsch alignment score without the traceback.
    ///
    /// Returns the same score as ``needleman_wunsch`` but uses O(m) memory instead
    /// of O(n*m) by retaining only two rows of the DP table at a time. The runtime
    /// difference between the two is minor. Use this function when you need the
    /// score but not the alignment itself.
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
    ///
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
    ///
    /// Examples
    /// --------
    /// >>> import numpy as np
    /// >>> seq1 = np.array(list("GATTACA"))
    /// >>> seq2 = np.array(list("GCATGCA"))
    /// >>> sm = np.where(seq1[:, None] == seq2[None, :], 1.0, -1.0)
    /// >>> needleman_wunsch_score(sm, gap_penalty=-1.0)
    /// 2.0
    ///
    /// Notes
    /// -----
    /// All values in ``similarity_matrix`` and the gap penalties must be finite.
    ///
    #[pyfunction]
    #[pyo3(
        signature = (similarity_matrix, *, gap_penalty=-1.0, insert_penalty=None, delete_penalty=None),
        text_signature = "(similarity_matrix, *, gap_penalty=-1.0, insert_penalty=None, delete_penalty=None)",
    )]
    fn needleman_wunsch_score<'py>(
        py: Python<'py>,
        similarity_matrix: Bound<'py, PyAny>,
        gap_penalty: f64,
        insert_penalty: Option<f64>,
        delete_penalty: Option<f64>,
    ) -> PyResult<f64> {
        let pyarray = to_pyreadonly(py, similarity_matrix).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into 2-dimensional float64 array",
            )
        })?;
        let similarity_matrix = pyarray.as_array();

        let insert_penalty = insert_penalty.unwrap_or(gap_penalty);
        let delete_penalty = delete_penalty.unwrap_or(gap_penalty);

        validate_inputs(similarity_matrix, insert_penalty, delete_penalty)?;

        Ok(nw::needleman_wunsch_score(
            similarity_matrix,
            insert_penalty,
            delete_penalty,
        ))
    }

    type AlignmentIndicesResult<'py> = (
        Bound<'py, PyArray1<isize>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<isize>>,
        Bound<'py, PyArray1<bool>>,
    );

    #[pyfunction]
    #[pyo3(signature = (editops), text_signature = "(editops)")]
    fn alignment_indices<'py>(
        py: Python<'py>,
        editops: Bound<'py, PyAny>,
    ) -> PyResult<AlignmentIndicesResult<'py>> {
        let pyarray = to_pyreadonly(py, editops).map_err(|_| {
            pyo3::exceptions::PyValueError::new_err(
                "Cannot convert array-like into 1-dimensional u8 array",
            )
        })?;
        let editops = nw::parse_editops(pyarray.as_array())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;

        let (source, target) = nw::alignment_indices(editops.view());

        Ok((
            source.indices.into_pyarray(py),
            source.mask.into_pyarray(py),
            target.indices.into_pyarray(py),
            target.mask.into_pyarray(py),
        ))
    }
}
