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
pub enum EditOp {
    Align = 0,
    Insert = 1,
    Delete = 2,
}

pub fn parse_editops<D: Dimension>(
    array: ArrayView<u8, D>,
) -> Result<Array<EditOp, D>, &'static str> {
    let dim = array.dim();
    let editops: Vec<EditOp> = array
        .iter()
        .map(|&x| EditOp::try_from(x).map_err(|_| "Cannot convert u8 into EditOp"))
        .collect::<Result<_, _>>()?;
    Array::from_shape_vec(dim, editops).map_err(|_| "Shape error")
}

pub struct MaskedIndexArray {
    pub indices: Array1<isize>,
    pub mask: Array1<bool>,
}

pub fn needleman_wunsch(
    align_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> (f64, Array1<EditOp>) {
    let (score, traceback) = fill_traceback(align_scores, insert_penalty, delete_penalty);
    let editops = traceback_editops(traceback.view());
    (score, editops)
}

/// Score-only NW fill.
///
/// Uses a rolling 2-row DP buffer (`O(m)` space) and carries the `diag`/`left`
/// cells in registers across the inner loop, so each iteration performs one
/// `prev_row` load and one `align_scores` load instead of four indexed reads.
pub fn needleman_wunsch_score(
    align_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> f64 {
    let (n, m) = (align_scores.nrows(), align_scores.ncols());

    let mut prev_row = Array1::<f64>::zeros(m + 1);
    let mut curr_row = Array1::<f64>::zeros(m + 1);

    // Row 0: reaching column j requires j inserts.
    for col in 1..=m {
        prev_row[col] = col as f64 * insert_penalty;
    }

    for row in 1..=n {
        let score_row = align_scores.row(row - 1);

        // Column 0 of each row is reached by a delete from the cell above.
        let cell_score = prev_row[0] + delete_penalty;
        curr_row[0] = cell_score;

        // Seed the rolling window for col = 1:
        //   diag = prev_row[0]  (the cell up-and-left of col 1)
        //   left = curr_row[0]  (the cell immediately left of col 1)
        let mut diag = prev_row[0];
        let mut left = cell_score;

        for col in 1..=m {
            let up = prev_row[col];

            let align = diag + score_row[col - 1];
            let delete = up + delete_penalty;
            let insert = left + insert_penalty;

            // Two sequential comparisons compile to two `maxsd` instructions,
            // faster than a three-way if/else (branchless bitwise-select, ~13
            // SSE instructions). Tie-breaking is preserved: `maxsd dst, src`
            // returns `src` on equality, so align beats delete beats insert.
            let mut cell_score = align;
            if delete > cell_score {
                cell_score = delete;
            }
            if insert > cell_score {
                cell_score = insert;
            }
            curr_row[col] = cell_score;

            // Shift the window one step right: this iteration's `up` is next
            // iteration's `diag`; this iteration's result is next iteration's
            // `left`.
            diag = up;
            left = cell_score;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[m]
}

pub fn alignment_indices(editops: ArrayView1<EditOp>) -> (MaskedIndexArray, MaskedIndexArray) {
    let k = editops.len();

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

    for (i, &op) in editops.iter().enumerate() {
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
/// The DP score table is reduced to a rolling 2-row buffer (`O(m)` space); the
/// inner loop carries `diag`/`left` in registers so each step does one
/// `prev_row` load and one `align_scores` load. The traceback matrix must
/// remain `O(nm)` because traceback reads it back-to-front after the fill.
/// True linear-space traceback would require Hirschberg's divide-and-conquer
/// algorithm.
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

    // Row 0: reaching column j requires j inserts.
    for col in 1..=m {
        prev_row[col] = col as f64 * insert_penalty;
        traceback[[0, col]] = EditOp::Insert;
    }

    for row in 1..=n {
        let score_row = align_scores.row(row - 1);
        let mut traceback_row = traceback.row_mut(row);

        // Column 0 of each row is reached by a delete from the cell above.
        let cell_score = prev_row[0] + delete_penalty;
        curr_row[0] = cell_score;
        traceback_row[0] = EditOp::Delete;

        // Seed the rolling window for col = 1:
        //   diag = prev_row[0]  (the cell up-and-left of col 1)
        //   left = curr_row[0]  (the cell immediately left of col 1)
        let mut diag = prev_row[0];
        let mut left = cell_score;

        for col in 1..=m {
            let up = prev_row[col];

            let align = diag + score_row[col - 1];
            let delete = up + delete_penalty;
            let insert = left + insert_penalty;

            // Tie-breaking: align > delete > insert
            let (cell_score, op) = if insert > delete && insert > align {
                (insert, EditOp::Insert)
            } else if delete > align {
                (delete, EditOp::Delete)
            } else {
                (align, EditOp::Align)
            };

            curr_row[col] = cell_score;
            traceback_row[col] = op;

            // Shift the window one step right: this iteration's `up` is next
            // iteration's `diag`; this iteration's result is next iteration's
            // `left`.
            diag = up;
            left = cell_score;
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
fn traceback_editops(traceback: ArrayView2<EditOp>) -> Array1<EditOp> {
    let (n, m) = (traceback.nrows() - 1, traceback.ncols() - 1);

    let mut editops = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        let op = traceback[[i, j]];
        editops.push(op);

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

    editops.reverse();
    Array1::from(editops)
}
