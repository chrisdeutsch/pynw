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
//!     dp[i-1][j-1] + S[i-1][j-1],        // diagonal - match/mismatch
//!     dp[i-1][j]   + gap_penalty_target,  // up       - gap in target sequence
//!     dp[i][j-1]   + gap_penalty_source,  // left     - gap in source sequence
//! )
//! ```
//!
//! Tie-breaking is deterministic: **diagonal > up > left**. This produces
//! compact alignments that prefer substitutions over gaps.
//!
//! # Safety contract
//!
//! All values in the similarity matrix and gap penalties must be finite.
//! Non-finite values (`NaN`, `Inf`) cause undefined output.

use numpy::ndarray::{Array2, ArrayView2};

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub(crate) enum EditOp {
    Align = 0,
    Insert = 1,
    Delete = 2,
}

pub(crate) fn needleman_wunsch(
    similarity_matrix: ArrayView2<f64>,
    gap_penalty_source: f64,
    gap_penalty_target: f64,
) -> (f64, Vec<EditOp>) {
    let (n, m) = (similarity_matrix.nrows(), similarity_matrix.ncols());
    let (dp, tb) = fill_matrices(similarity_matrix, gap_penalty_source, gap_penalty_target);
    let ops = traceback_indices(tb.view());
    let score = dp[[n, m]];

    // TODO: Return array of scores?
    (score, ops)
}

/// Build `(n+1, m+1)` DP score and traceback direction matrices.
///
/// Both matrices are `O(nm)`.  A rolling 2-row band could eliminate `dp`,
/// but the `tb` matrix is still `O(nm)`.  True linear-space traceback
/// requires Hirschberg's divide-and-conquer algorithm.
fn fill_matrices(
    similarity_matrix: ArrayView2<f64>,
    gap_penalty_source: f64,
    gap_penalty_target: f64,
) -> (Array2<f64>, Array2<EditOp>) {
    let (n, m) = (similarity_matrix.nrows(), similarity_matrix.ncols());
    let mut dp = Array2::<f64>::zeros((n + 1, m + 1));
    let mut tb = Array2::from_elem((n + 1, m + 1), EditOp::Align);

    // First column: aligning i source-elements against nothing -> i target gaps.
    for i in 1..=n {
        dp[[i, 0]] = i as f64 * gap_penalty_target;
        tb[[i, 0]] = EditOp::Delete;
    }
    // First row: aligning j target-elements against nothing -> j source gaps.
    for j in 1..=m {
        dp[[0, j]] = j as f64 * gap_penalty_source;
        tb[[0, j]] = EditOp::Insert;
    }

    for i in 1..=n {
        for j in 1..=m {
            let align = dp[[i - 1, j - 1]] + similarity_matrix[[i - 1, j - 1]];
            let delete = dp[[i - 1, j]] + gap_penalty_target;
            let insert = dp[[i, j - 1]] + gap_penalty_source;

            // Tie-breaking: diagonal > up > left. The cascade uses strict > so
            // that equal scores fall through to the higher-priority move.
            let (score, best_direction) = if insert > delete && insert > align {
                (insert, EditOp::Insert)
            } else if delete > align {
                (delete, EditOp::Delete)
            } else {
                (align, EditOp::Align)
            };

            dp[[i, j]] = score;
            tb[[i, j]] = best_direction;
        }
    }

    (dp, tb)
}

/// Returns equal-length `(source_idx, target_idx)` (at most `n + m`).
///
/// Debug-asserts that the traceback never underflows.  This invariant is
/// guaranteed by the boundary initialisation in [`fill_matrices`]:
/// column 0 is always `Up` and row 0 is always `Left`.
fn traceback_indices(traceback_matrix: ArrayView2<EditOp>) -> Vec<EditOp> {
    let (n, m) = (traceback_matrix.nrows() - 1, traceback_matrix.ncols() - 1);

    let mut ops = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        let op = traceback_matrix[[i, j]];
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
    ops
}
