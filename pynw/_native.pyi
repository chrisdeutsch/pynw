# NOTE: This stub provides the docstring for type checkers and IDE support.
# A copy exists as a Rust doc comment in src/lib.rs for runtime `help()`.
# Keep both copies in sync.

import numpy as np
import numpy.typing as npt

OP_ALIGN: int
OP_INSERT: int
OP_DELETE: int

def needleman_wunsch(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    gap_penalty_source: float | None = None,
    gap_penalty_target: float | None = None,
) -> tuple[float, npt.NDArray[np.uint8]]:
    """Align two ordered sequences given a precomputed similarity matrix.

    The total alignment score is the sum of similarity-matrix entries for
    matched positions and gap penalties for insertions/deletions.

    Parameters
    ----------
    similarity_matrix : array_like, shape (n, m)
        ``similarity_matrix[i, j]`` is the similarity score for aligning
        element *i* of the source sequence with element *j* of the target
        sequence.
    gap_penalty : float, default -1.0
        Penalty applied when a gap is inserted in either sequence.
        Use ``gap_penalty_source`` or ``gap_penalty_target`` to specify
        different penalties for each sequence.
    gap_penalty_source : float, optional
        Penalty added when a gap is inserted in the source sequence
        (the target sequence advances). This can be thought of as the
        cost of a deletion from the source sequence or an insertion into the
        target sequence.
        Defaults to ``gap_penalty`` if not specified.
    gap_penalty_target : float, optional
        Penalty added when a gap is inserted in the target sequence
        (the source sequence advances). This can be thought of as the cost
        of a deletion from the target sequence or an insertion into the
        source sequence.
        Defaults to ``gap_penalty`` if not specified.

    Raises
    ------
    ValueError
        If ``similarity_matrix`` is not 2-dimensional, or if any value in
        ``similarity_matrix`` or the gap penalties is ``NaN`` or ``Inf``.

    Returns
    -------
    score : float
        The optimal alignment score.
    ops : ndarray of uint8, shape (k,)
        Sequence of edit operations describing the alignment.  Each element
        is one of ``OP_ALIGN``, ``OP_INSERT``, or ``OP_DELETE``.  Use
        ``iter_alignment`` to iterate over aligned element pairs, or
        ``indices_from_ops`` to reconstruct source and target index arrays.

    Examples
    --------
    Align two DNA sequences using a simple match/mismatch scoring scheme:

    >>> import numpy as np
    >>> source_seq = list("GATTACA")
    >>> target_seq = list("GCATGCA")
    >>> match, mismatch = 1.0, -1.0
    >>> sm = np.where(
    ...     np.array(source_seq)[:, None] == np.array(target_seq)[None, :],
    ...     match, mismatch,
    ... )
    >>> from pynw import iter_alignment
    >>> score, ops = needleman_wunsch(sm, gap_penalty=-1.0)
    >>> score
    2.0
    >>> cols = [
    ...     (s or "-", t or "-")
    ...     for _, s, t in iter_alignment(ops, source_seq, target_seq)
    ... ]
    >>> "".join(s for s, _ in cols)
    'G-ATTACA'
    >>> "".join(t for _, t in cols)
    'GCA-TGCA'

    Notes
    -----
    When multiple alignments achieve the same optimal score, ties are
    broken deterministically: ``Diagonal > Up > Left``.  This prefers
    substitutions over gaps, producing compact alignments.  Other tools
    may return different co-optimal alignments.

    All values in ``similarity_matrix`` and the gap penalties must be finite.
    """
    ...

def needleman_wunsch_merge_split(
    align_scores: npt.ArrayLike,
    split_scores: npt.ArrayLike,
    merge_scores: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    insert_penalty: float | None = None,
    delete_penalty: float | None = None,
) -> tuple[float, npt.NDArray[np.uint8]]:
    """Align two sequences with one-to-one, split (1→2), and merge (2→1) operations.

    Extends Needleman-Wunsch with two additional edit operations: a source
    element can be *split* to align with two consecutive target elements, or
    two consecutive source elements can be *merged* to align with one target
    element.  The score for each operation at each position is supplied by
    the caller via separate score matrices.

    Parameters
    ----------
    align_scores : array_like, shape (n, m)
        ``align_scores[i, j]`` is the score for aligning source element *i*
        with target element *j* (one-to-one match).
    split_scores : array_like, shape (n, m-1)
        ``split_scores[i, j]`` is the score for splitting source element *i*
        across target elements *j* and *j+1* (one source → two targets).
    merge_scores : array_like, shape (n-1, m)
        ``merge_scores[i, j]`` is the score for merging source elements *i*
        and *i+1* into target element *j* (two sources → one target).
    gap_penalty : float, default -1.0
        Penalty for an insert or delete step.  Use ``insert_penalty`` or
        ``delete_penalty`` to set them independently.
    insert_penalty : float, optional
        Penalty for advancing the target sequence without consuming a source
        element.  Defaults to ``gap_penalty``.
    delete_penalty : float, optional
        Penalty for advancing the source sequence without consuming a target
        element.  Defaults to ``gap_penalty``.

    Raises
    ------
    ValueError
        If any score matrix is not 2-dimensional, or if any value is non-finite.

    Returns
    -------
    score : float
        The optimal alignment score.
    ops : ndarray of uint8, shape (k,)
        Sequence of edit operations describing the alignment.  Each element
        is one of the ``OP_*`` constants (or ``Op`` enum values).  Use
        ``indices_from_ops`` to reconstruct the source and target indices.

        +---------------+----------------------------------------------+
        | Op            | Meaning                                      |
        +===============+==============================================+
        | ``OP_ALIGN``  | source[i] aligned with target[j]             |
        | ``OP_INSERT`` | gap in source; target[j] consumed            |
        | ``OP_DELETE`` | source[i] consumed; gap in target            |
        | ``OP_SPLIT``  | source[i] split into target[j], target[j+1] |
        | ``OP_MERGE``  | source[i], source[i+1] merged into target[j]|
        +---------------+----------------------------------------------+

    Notes
    -----
    Ties are broken deterministically: ``Align > Merge > Split > Delete > Insert``.

    All score values and penalties must be finite.
    """
    ...
