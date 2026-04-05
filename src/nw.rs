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
    let (score, traceback) = fill_traceback(align_scores, insert_penalty, delete_penalty);
    let ops = traceback_ops(traceback.view());

    (score, ops)
}

pub(crate) fn needleman_wunsch_score(
    align_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> f64 {
    let (n, m) = (align_scores.nrows(), align_scores.ncols());

    let mut prev_row = Array1::<f64>::zeros(m + 1);
    let mut curr_row = Array1::<f64>::zeros(m + 1);
    for col in 1..=m {
        prev_row[col] = col as f64 * insert_penalty;
    }

    for row in 1..=n {
        curr_row[0] = prev_row[0] + delete_penalty;

        for col in 1..=m {
            let align = prev_row[col - 1] + align_scores[[row - 1, col - 1]];
            let delete = prev_row[col] + delete_penalty;
            let insert = curr_row[col - 1] + insert_penalty;

            // Two sequential comparisons compile to two `maxsd` instructions,
            // faster than a three-way if/else (branchless bitwise-select, ~13
            // SSE instructions). Tie-breaking is preserved: `maxsd dst, src`
            // returns `src` on equality, so align beats delete beats insert.
            let mut score = align;
            if delete > score {
                score = delete;
            }
            if insert > score {
                score = insert;
            }
            curr_row[col] = score;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[m]
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

/// Build the traceback direction matrix and return the final score.
///
/// The DP score table is reduced to a rolling 2-row buffer (`O(m)` space). The
/// traceback matrix must remain `O(nm)` because traceback reads it back-to-front
/// after the fill. True linear-space traceback would require Hirschberg's
/// divide-and-conquer algorithm.
fn fill_traceback(
    align_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> (f64, Array2<EditOp>) {
    let (n, m) = (align_scores.nrows(), align_scores.ncols());

    let mut prev_row = Array1::<f64>::zeros(m + 1);
    let mut curr_row = Array1::<f64>::zeros(m + 1);

    // Each cell stores the editop that produced its score, i.e. a pointer back
    // to the predecessor cell.
    let mut traceback = Array2::from_elem((n + 1, m + 1), EditOp::Align);

    // First row: aligning j target-elements against nothing -> j source gaps.
    for j in 1..=m {
        prev_row[j] = j as f64 * insert_penalty;
        traceback[[0, j]] = EditOp::Insert;
    }

    for i in 1..=n {
        curr_row[0] = prev_row[0] + delete_penalty;
        traceback[[i, 0]] = EditOp::Delete;

        for j in 1..=m {
            let align = prev_row[j - 1] + align_scores[[i - 1, j - 1]];
            let delete = prev_row[j] + delete_penalty;
            let insert = curr_row[j - 1] + insert_penalty;

            // Tie-breaking: align > delete > insert
            let (score, op) = if insert > delete && insert > align {
                (insert, EditOp::Insert)
            } else if delete > align {
                (delete, EditOp::Delete)
            } else {
                (align, EditOp::Align)
            };

            curr_row[j] = score;
            traceback[[i, j]] = op;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    (prev_row[m], traceback)
}

/// Returns equal-length `(source_idx, target_idx)` (at most `n + m`).
///
/// Debug-asserts that the traceback never underflows.  This invariant is
/// guaranteed by the boundary initialisation in [`fill_traceback`]:
/// column 0 is always `Up` and row 0 is always `Left`.
fn traceback_ops(traceback: ArrayView2<EditOp>) -> Array1<EditOp> {
    let (n, m) = (traceback.nrows() - 1, traceback.ncols() - 1);

    let mut ops = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        let op = traceback[[i, j]];
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
