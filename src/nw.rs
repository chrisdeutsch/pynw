//! Needleman-Wunsch global sequence alignment.
//!
//! # Algorithm
//!
//! Given an `(n, m)` similarity matrix `S` and two gap penalties, we build an
//! `(n+1, m+1)` DP table where `dp[i][j]` is the best score for aligning
//! the first `i` source-elements with the first `j` target-elements.  The
//! recurrence is:
//!
//! ```text
//! dp[i][j] = max(
//!     dp[i-1][j-1] + S[i-1][j-1],   // diagonal - match/mismatch
//!     dp[i-1][j]   + delete_penalty, // up       - gap in target sequence
//!     dp[i][j-1]   + insert_penalty, // left     - gap in source sequence
//! )
//! ```
//!
//! Tie-breaking is deterministic: **Align > Delete > Insert**. This produces
//! compact alignments that prefer substitutions over gaps.
//!
//! # Safety contract
//!
//! All values in the similarity matrix and gap penalties must be finite.
//! Non-finite values (`NaN`, `Inf`) cause undefined output.

use ndarray::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Clone, Copy, Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u8)]
pub(crate) enum EditOp {
    Align = 0,
    Insert = 1,
    Delete = 2,
}

pub fn parse_editops<D: Dimension>(
    array: ArrayView<u8, D>,
) -> Result<Array<EditOp, D>, &'static str> {
    let dim = array.dim();
    let ops: Vec<EditOp> = array
        .iter()
        .map(|&x| EditOp::try_from(x).map_err(|_| "Cannot convert u8 into EditOp"))
        .collect::<Result<_, _>>()?;
    Array::from_shape_vec(dim, ops).map_err(|_| "Shape error")
}

pub(crate) struct MaskedIndexArray {
    pub indices: Array1<isize>,
    pub mask: Array1<bool>,
}

pub(crate) fn needleman_wunsch(
    align_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> (f64, Array1<EditOp>) {
    let (n, m) = (align_scores.nrows(), align_scores.ncols());
    let (dp, tb) = fill_matrices(align_scores, insert_penalty, delete_penalty);
    let ops = traceback_ops(tb.view());
    let score = dp[[n, m]];

    // TODO: Return array of scores?
    (score, ops)
}

pub(crate) fn alignment_indices(ops: ArrayView1<EditOp>) -> (MaskedIndexArray, MaskedIndexArray) {
    let k = ops.len();

    let mut source = MaskedIndexArray {
        indices: Array1::zeros(k),
        mask: Array1::from_elem(k, false),
    };
    let mut target = MaskedIndexArray {
        indices: Array1::zeros(k),
        mask: Array1::from_elem(k, false),
    };

    let mut source_index: isize = 0;
    let mut target_index: isize = 0;

    for (i, &op) in ops.iter().enumerate() {
        source.indices[i] = source_index;
        target.indices[i] = target_index;

        match op {
            EditOp::Align => {
                // both advance
                source_index += 1;
                target_index += 1;
            }
            EditOp::Insert => {
                // gap in source; target advances
                source.mask[i] = true;
                target_index += 1;
            }
            EditOp::Delete => {
                // gap in target; source advances
                target.mask[i] = true;
                source_index += 1;
            }
        }
    }

    (source, target)
}

/// Build `(n+1, m+1)` DP score and traceback direction matrices.
///
/// Both matrices are `O(nm)`.  A rolling 2-row band could eliminate `dp`,
/// but the `tb` matrix is still `O(nm)`.  True linear-space traceback
/// requires Hirschberg's divide-and-conquer algorithm.
fn fill_matrices(
    align_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> (Array2<f64>, Array2<EditOp>) {
    let (n, m) = (align_scores.nrows(), align_scores.ncols());

    let mut dp_table = Array2::zeros((n + 1, m + 1));
    // Each cell stores the editop that produced its score, i.e. a pointer back
    // to the predecessor cell.
    let mut pointers = Array2::from_elem((n + 1, m + 1), EditOp::Align);

    // First column: aligning i source-elements against nothing -> i target gaps.
    for i in 1..=n {
        dp_table[[i, 0]] = i as f64 * delete_penalty;
        pointers[[i, 0]] = EditOp::Delete;
    }
    // First row: aligning j target-elements against nothing -> j source gaps.
    for j in 1..=m {
        dp_table[[0, j]] = j as f64 * insert_penalty;
        pointers[[0, j]] = EditOp::Insert;
    }

    for i in 1..=n {
        for j in 1..=m {
            let align = dp_table[[i - 1, j - 1]] + align_scores[[i - 1, j - 1]];
            let delete = dp_table[[i - 1, j]] + delete_penalty;
            let insert = dp_table[[i, j - 1]] + insert_penalty;

            // Tie-breaking: align > delete > insert
            let (score, op) = if insert > delete && insert > align {
                (insert, EditOp::Insert)
            } else if delete > align {
                (delete, EditOp::Delete)
            } else {
                (align, EditOp::Align)
            };

            dp_table[[i, j]] = score;
            pointers[[i, j]] = op;
        }
    }

    (dp_table, pointers)
}

/// Traces back through the DP matrix, building masked index arrays directly.
///
/// Equivalent to calling [`traceback_ops`] then [`alignment_indices`], but in
/// a single pass.  At position `(i, j)` in the traceback matrix, `i` and `j`
/// represent the number of source/target elements consumed so far, so indices
/// can be recorded without a second traversal.
pub(crate) fn traceback_indices(
    pointers: ArrayView2<EditOp>,
) -> (MaskedIndexArray, MaskedIndexArray) {
    let (n, m) = (pointers.nrows() - 1, pointers.ncols() - 1);

    let mut source_indices: Vec<isize> = Vec::with_capacity(n + m);
    let mut source_mask: Vec<bool> = Vec::with_capacity(n + m);
    let mut target_indices: Vec<isize> = Vec::with_capacity(n + m);
    let mut target_mask: Vec<bool> = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        let op = pointers[[i, j]];

        match op {
            EditOp::Align => {
                debug_assert!(i > 0 && j > 0, "Diagonal at boundary would underflow");
                i -= 1;
                j -= 1;
                source_indices.push(i as isize);
                source_mask.push(false);
                target_indices.push(j as isize);
                target_mask.push(false);
            }
            EditOp::Delete => {
                debug_assert!(i > 0, "Up at row 0 would underflow");
                i -= 1;
                source_indices.push(i as isize);
                source_mask.push(false);
                target_indices.push(j as isize);
                target_mask.push(true);
            }
            EditOp::Insert => {
                debug_assert!(j > 0, "Left at col 0 would underflow");
                j -= 1;
                source_indices.push(i as isize);
                source_mask.push(true);
                target_indices.push(j as isize);
                target_mask.push(false);
            }
        }
    }

    source_indices.reverse();
    source_mask.reverse();
    target_indices.reverse();
    target_mask.reverse();

    (
        MaskedIndexArray {
            indices: Array1::from_vec(source_indices),
            mask: Array1::from_vec(source_mask),
        },
        MaskedIndexArray {
            indices: Array1::from_vec(target_indices),
            mask: Array1::from_vec(target_mask),
        },
    )
}

/// Returns equal-length `(source_idx, target_idx)` (at most `n + m`).
///
/// Debug-asserts that the traceback never underflows.  This invariant is
/// guaranteed by the boundary initialisation in [`fill_matrices`]:
/// column 0 is always `Up` and row 0 is always `Left`.
fn traceback_ops(pointers: ArrayView2<EditOp>) -> Array1<EditOp> {
    let (n, m) = (pointers.nrows() - 1, pointers.ncols() - 1);

    let mut ops = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        let op = pointers[[i, j]];
        ops.push(op);

        match op {
            EditOp::Align => {
                debug_assert!(i > 0 && j > 0, "Diagonal at boundary would underflow");
                i -= 1;
                j -= 1;
            }
            EditOp::Delete => {
                debug_assert!(i > 0, "Up at row 0 would underflow");
                i -= 1;
            }
            EditOp::Insert => {
                debug_assert!(j > 0, "Left at col 0 would underflow");
                j -= 1;
            }
        }
    }

    ops.reverse();
    Array1::from(ops)
}
