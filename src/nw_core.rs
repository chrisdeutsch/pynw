//! Needleman-Wunsch global sequence alignment.
//!
//! # Algorithm
//!
//! Given an `(n, m)` similarity matrix `S` and two gap penalties, we build an
//! `(n+1, m+1)` DP table where `dp[i][j]` is the best score for aligning
//! the first `i` row-elements with the first `j` column-elements.  The
//! recurrence is:
//!
//! ```text
//! dp[i][j] = max(
//!     dp[i-1][j-1] + S[i-1][j-1],     // diagonal - match/mismatch
//!     dp[i-1][j]   + gap_penalty_col, // up       - gap in column sequence
//!     dp[i][j-1]   + gap_penalty_row, // left     - gap in row sequence
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

const GAP: isize = -1;

#[derive(Clone, Copy, Debug)]
enum Direction {
    /// Match/mismatch: both sequences advance.
    Diagonal,
    /// Gap in the column sequence: only the row sequence advances.
    Up,
    /// Gap in the row sequence: only the column sequence advances.
    Left,
}

/// Build `(n+1, m+1)` DP score and traceback direction matrices.
///
/// Both matrices are `O(nm)`.  A rolling 2-row band could eliminate `dp`,
/// but the `tb` matrix is still `O(nm)`.  True linear-space traceback
/// requires Hirschberg's divide-and-conquer algorithm.
fn fill_matrices(
    similarity_matrix: &ArrayView2<f64>,
    gap_penalty_row: f64,
    gap_penalty_col: f64,
) -> (Array2<f64>, Array2<Direction>) {
    let (n, m) = (similarity_matrix.nrows(), similarity_matrix.ncols());
    let mut dp = Array2::<f64>::zeros((n + 1, m + 1));
    let mut tb = Array2::from_elem((n + 1, m + 1), Direction::Diagonal);

    // First column: aligning i row-elements against nothing -> i column gaps.
    for i in 1..=n {
        dp[[i, 0]] = i as f64 * gap_penalty_col;
        tb[[i, 0]] = Direction::Up;
    }
    // First row: aligning j column-elements against nothing -> j row gaps.
    for j in 1..=m {
        dp[[0, j]] = j as f64 * gap_penalty_row;
        tb[[0, j]] = Direction::Left;
    }

    for i in 1..=n {
        for j in 1..=m {
            let diagonal = dp[[i - 1, j - 1]] + similarity_matrix[[i - 1, j - 1]];
            let up = dp[[i - 1, j]] + gap_penalty_col;
            let left = dp[[i, j - 1]] + gap_penalty_row;

            // Tie-breaking: diagonal > up > left. The cascade uses strict > so
            // that equal scores fall through to the higher-priority move.
            let (score, best_direction) = if left > up && left > diagonal {
                (left, Direction::Left)
            } else if up > diagonal {
                (up, Direction::Up)
            } else {
                (diagonal, Direction::Diagonal)
            };

            dp[[i, j]] = score;
            tb[[i, j]] = best_direction;
        }
    }

    (dp, tb)
}

/// Returns equal-length `(row_idx, col_idx)` (at most `n + m`).
///
/// Debug-asserts that the traceback never underflows.  This invariant is
/// guaranteed by the boundary initialisation in [`fill_matrices`]:
/// column 0 is always `Up` and row 0 is always `Left`.
fn traceback_indices(traceback_matrix: &ArrayView2<Direction>) -> (Vec<isize>, Vec<isize>) {
    let (n, m) = (traceback_matrix.nrows() - 1, traceback_matrix.ncols() - 1);

    let mut row_idx = Vec::with_capacity(n + m);
    let mut col_idx = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        match traceback_matrix[[i, j]] {
            Direction::Diagonal => {
                debug_assert!(i > 0 && j > 0, "Diagonal at boundary would underflow");
                i -= 1;
                j -= 1;
                row_idx.push(i as isize);
                col_idx.push(j as isize);
            }
            Direction::Up => {
                debug_assert!(i > 0, "Up at row 0 would underflow");
                i -= 1;
                row_idx.push(i as isize);
                col_idx.push(GAP);
            }
            Direction::Left => {
                debug_assert!(j > 0, "Left at col 0 would underflow");
                j -= 1;
                row_idx.push(GAP);
                col_idx.push(j as isize);
            }
        }
    }

    row_idx.reverse();
    col_idx.reverse();

    (row_idx, col_idx)
}

pub(crate) fn nw_traceback_indices_core(
    similarity_matrix: &ArrayView2<f64>,
    gap_penalty_row: f64,
    gap_penalty_col: f64,
) -> (f64, Vec<isize>, Vec<isize>) {
    let (n, m) = (similarity_matrix.nrows(), similarity_matrix.ncols());
    let (dp, tb) = fill_matrices(similarity_matrix, gap_penalty_row, gap_penalty_col);
    let (row_idx, col_idx) = traceback_indices(&tb.view());
    let score = dp[[n, m]];

    (score, row_idx, col_idx)
}
