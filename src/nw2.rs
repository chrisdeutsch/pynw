use numpy::ndarray::{Array2, ArrayView2};

const GAP: isize = -1;

#[derive(Clone, Copy, Debug)]
enum Op {
    Insert,
    Delete,
    Split,
    Merge,
    Keep,
}

pub(crate) fn needleman_wunsch_2(
    similarity_matrix: &ArrayView2<f64>,
    similarity_matrix_split: &ArrayView2<f64>,
    similarity_matrix_merge: &ArrayView2<f64>,
    gap_penalty_row: f64,
    gap_penalty_col: f64,
) -> (f64, Vec<isize>, Vec<isize>) {
    let (n, m) = (similarity_matrix.nrows(), similarity_matrix.ncols());
    let (dp, tb) = fill_matrices(
        similarity_matrix,
        similarity_matrix_split,
        similarity_matrix_merge,
        gap_penalty_row,
        gap_penalty_col,
    );
    let (row_idx, col_idx) = traceback_indices(&tb.view());
    let score = dp[[n, m]];

    (score, row_idx, col_idx)
}

fn fill_matrices(
    similarity_matrix: &ArrayView2<f64>,
    similarity_matrix_split: &ArrayView2<f64>,
    similarity_matrix_merge: &ArrayView2<f64>,
    gap_penalty_row: f64,
    gap_penalty_col: f64,
) -> (Array2<f64>, Array2<Op>) {
    let (n, m) = (similarity_matrix.nrows(), similarity_matrix.ncols());
    let mut dp = Array2::<f64>::zeros((n + 1, m + 1));
    let mut tb = Array2::from_elem((n + 1, m + 1), Op::Keep);

    for i in 1..=n {
        dp[[i, 0]] = i as f64 * gap_penalty_col;
        tb[[i, 0]] = Op::Delete;
    }

    for j in 1..=m {
        dp[[0, j]] = j as f64 * gap_penalty_row;
        tb[[0, j]] = Op::Insert;
    }

    for i in 1..=n {
        for j in 1..=m {
            let keep = dp[[i - 1, j - 1]] + similarity_matrix[[i - 1, j - 1]];
            let delete = dp[[i - 1, j]] + gap_penalty_col;
            let insert = dp[[i, j - 1]] + gap_penalty_row;

            let merge = if i >= 2 {
                dp[[i - 2, j - 1]] + similarity_matrix_merge[[i - 2, j - 1]]
            } else {
                f64::NEG_INFINITY
            };
            let split = if j >= 2 {
                dp[[i - 1, j - 2]] + similarity_matrix_split[[i - 1, j - 2]]
            } else {
                f64::NEG_INFINITY
            };

            let candidate_ops = [
                (Op::Insert, insert),
                (Op::Delete, delete),
                (Op::Split, split),
                (Op::Merge, merge),
                (Op::Keep, keep),
            ];

            let (op, score) = candidate_ops
                .into_iter()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();

            dp[[i, j]] = score;
            tb[[i, j]] = op;
        }
    }

    (dp, tb)
}

fn traceback_indices(traceback_matrix: &ArrayView2<Op>) -> (Vec<isize>, Vec<isize>) {
    let (n, m) = (traceback_matrix.nrows() - 1, traceback_matrix.ncols() - 1);

    let mut row_idx = Vec::with_capacity(n + m);
    let mut col_idx = Vec::with_capacity(n + m);

    let mut i = n;
    let mut j = m;

    while i > 0 || j > 0 {
        match traceback_matrix[[i, j]] {
            Op::Keep => {
                debug_assert!(i > 0 && j > 0, "Keep at boundary would underflow");
                i -= 1;
                j -= 1;
                row_idx.push(i as isize);
                col_idx.push(j as isize);
            }
            Op::Merge => {
                debug_assert!(i > 1 && j > 0, "Merge at boundary would underflow");
                i -= 2;
                j -= 1;
                row_idx.extend([i as isize, (i + 1) as isize]);
                col_idx.extend([j as isize, j as isize]);
            }
            Op::Split => {
                debug_assert!(i > 0 && j > 1, "Split at boundary would underflow");
                i -= 1;
                j -= 2;
                row_idx.extend([i as isize, i as isize]);
                col_idx.extend([j as isize, (j + 1) as isize])
            }
            Op::Delete => {
                debug_assert!(i > 0, "Delete at row 0 would underflow");
                i -= 1;
                row_idx.push(i as isize);
                col_idx.push(GAP);
            }
            Op::Insert => {
                debug_assert!(j > 0, "Insert at col 0 would underflow");
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
