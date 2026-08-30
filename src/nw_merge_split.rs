use numpy::ndarray::{Array2, ArrayView2};

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub(crate) enum EditOp {
    Align = 0,
    Insert = 1,
    Delete = 2,
    Split = 3,
    Merge = 4,
}

pub(crate) fn needleman_wunsch_merge_split(
    align_scores: ArrayView2<f64>,
    split_scores: ArrayView2<f64>,
    merge_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> (f64, Vec<EditOp>) {
    let (score, tb) = fill_matrices(
        align_scores,
        split_scores,
        merge_scores,
        insert_penalty,
        delete_penalty,
    );
    let ops = traceback_ops(tb.view());

    // TODO: Return array of scores?
    (score, ops)
}

fn fill_matrices(
    align_scores: ArrayView2<f64>,
    split_scores: ArrayView2<f64>,
    merge_scores: ArrayView2<f64>,
    insert_penalty: f64,
    delete_penalty: f64,
) -> (f64, Array2<EditOp>) {
    let (n, m) = (align_scores.nrows(), align_scores.ncols());
    let mut dp = Array2::<f64>::zeros((n + 1, m + 1));
    let mut tb = Array2::from_elem((n + 1, m + 1), EditOp::Align);

    for i in 1..=n {
        dp[[i, 0]] = i as f64 * delete_penalty;
        tb[[i, 0]] = EditOp::Delete;
    }

    for j in 1..=m {
        dp[[0, j]] = j as f64 * insert_penalty;
        tb[[0, j]] = EditOp::Insert;
    }

    for i in 1..=n {
        for j in 1..=m {
            let align = dp[[i - 1, j - 1]] + align_scores[[i - 1, j - 1]];
            let delete = dp[[i - 1, j]] + delete_penalty;
            let insert = dp[[i, j - 1]] + insert_penalty;

            let merge = if i >= 2 {
                dp[[i - 2, j - 1]] + merge_scores[[i - 2, j - 1]]
            } else {
                f64::NEG_INFINITY
            };
            let split = if j >= 2 {
                dp[[i - 1, j - 2]] + split_scores[[i - 1, j - 2]]
            } else {
                f64::NEG_INFINITY
            };

            // Ties broken by last-wins max_by: Align > Merge > Split > Delete > Insert.
            let candidate_ops = [
                (EditOp::Insert, insert),
                (EditOp::Delete, delete),
                (EditOp::Split, split),
                (EditOp::Merge, merge),
                (EditOp::Align, align),
            ];

            let (op, score) = candidate_ops
                .into_iter()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .unwrap();

            dp[[i, j]] = score;
            tb[[i, j]] = op;
        }
    }

    (dp[[n, m]], tb)
}

fn traceback_ops(traceback: ArrayView2<EditOp>) -> Vec<EditOp> {
    let (n, m) = (traceback.nrows() - 1, traceback.ncols() - 1);
    let mut ops = Vec::with_capacity(n + m);

    let (mut i, mut j) = (n, m);
    while i > 0 || j > 0 {
        let op = traceback[[i, j]];
        ops.push(op);

        match op {
            EditOp::Align => {
                debug_assert!(i > 0 && j > 0, "Keep at boundary would underflow");
                i -= 1;
                j -= 1;
            }
            EditOp::Merge => {
                debug_assert!(i > 1 && j > 0, "Merge at boundary would underflow");
                i -= 2;
                j -= 1;
            }
            EditOp::Split => {
                debug_assert!(i > 0 && j > 1, "Split at boundary would underflow");
                i -= 1;
                j -= 2;
            }
            EditOp::Delete => {
                debug_assert!(i > 0, "Delete at row 0 would underflow");
                i -= 1;
            }
            EditOp::Insert => {
                debug_assert!(j > 0, "Insert at col 0 would underflow");
                j -= 1;
            }
        }
    }

    ops.reverse();
    ops
}
