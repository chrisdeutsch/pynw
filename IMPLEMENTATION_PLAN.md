# Implementation Plan: Unified Needleman-Wunsch API

## Motivation

pynw currently exposes two functions with incompatible interfaces:

|                    | `needleman_wunsch`                          | `needleman_wunsch_merge_split`      |
| ------------------ | ------------------------------------------- | ----------------------------------- |
| **Input**          | 1 score matrix + 2 gap penalties            | 3 score matrices + 2 gap penalties  |
| **Returns**        | `(score, source_idx, target_idx)`           | `(score, ops)`                      |
| **Gap names**      | `gap_penalty_source` / `gap_penalty_target` | `insert_penalty` / `delete_penalty` |
| **Gap convention** | `-1` sentinel in index arrays               | gap info lives in ops array         |

This creates two problems: users face different mental models depending on
which function they call, and adding new operations (e.g. 2-to-2 swap)
requires yet another function or a growing parameter list.

## Key Insight

Every edit operation in Needleman-Wunsch is fully described by three values:

1. **Source stride** -- how many source elements it consumes (0, 1, 2, ...)
2. **Target stride** -- how many target elements it consumes (0, 1, 2, ...)
3. **Score source** -- either a scalar (constant penalty) or a position-dependent
   matrix of shape `(n - source_stride + 1, m - target_stride + 1)`

All current and future operations fit this model:

| Operation         | Row stride | Col stride | Score shape  |
| ----------------- | ---------- | ---------- | ------------ |
| Insert            | 0          | 1          | scalar       |
| Delete            | 1          | 0          | scalar       |
| Align             | 1          | 1          | (n, m)       |
| Merge             | 2          | 1          | (n-1, m)     |
| Split             | 1          | 2          | (n, m-1)     |
| _Swap (future)_   | _2_        | _2_        | _(n-1, m-1)_ |
| _3-to-1 (future)_ | _3_        | _1_        | _(n-2, m)_   |

## New Public API

### `needleman_wunsch`

One function. Caller-defined operations. Internal dispatch for performance.

```python
def needleman_wunsch(
    *operations: tuple[int, int, npt.ArrayLike | float],
    check_finite: bool = False,
) -> tuple[float, npt.NDArray[np.uint8]]:
    """Align two ordered sequences using caller-defined edit operations.

    Each positional argument defines one edit operation as a
    ``(source_stride, target_stride, scores)`` tuple:

    - **source_stride** / **target_stride**: number of source / target elements
      consumed by this operation (non-negative integers).
    - **scores**: either a scalar (constant cost, e.g. a gap penalty) or
      a 2-D array of shape
      ``(n - source_stride + 1, m - target_stride + 1)``
      giving the position-dependent score. Sequence lengths *n* and *m*
      are inferred from the first matrix-valued operation.

    The position of each operation in the argument list becomes its op
    code in the returned ``ops`` array.  Use ``indices_from_ops`` to
    convert the op sequence into row/column index arrays.

    Parameters
    ----------
    *operations : tuple[int, int, ArrayLike | float]
        Variable number of ``(source_stride, target_stride, scores)`` tuples.
        At least one operation must have both strides > 0 so that *n*
        and *m* can be inferred.
    check_finite : bool, default False
        If ``True``, raise ``ValueError`` when any score matrix or
        scalar contains ``NaN`` or ``Inf``.

    Returns
    -------
    score : float
        The optimal alignment score.
    ops : ndarray of uint8, shape (k,)
        Sequence of op codes.  ``ops[i]`` is the index of the operation
        (in the ``*operations`` argument list) applied at alignment
        step *i*.

    Raises
    ------
    ValueError
        If no matrix-valued operation is provided (cannot infer n, m),
        if a score matrix has the wrong shape, if the input is not
        2-dimensional, or if ``check_finite=True`` and any value is
        non-finite.

    Notes
    -----
    When multiple operations achieve the same score at a DP cell, ties
    are broken by **lowest op index wins** (i.e. the operation listed
    first in ``*operations``).

    All score values must be finite. Passing ``NaN`` or ``Inf`` without
    ``check_finite=True`` is undefined behavior.

    Examples
    --------
    Standard alignment (match/mismatch + gaps):

    >>> import numpy as np
    >>> from pynw import needleman_wunsch, indices_from_ops
    >>> seq1, seq2 = list("GATTACA"), list("GCATGCA")
    >>> sm = np.where(
    ...     np.array(seq1)[:, None] == np.array(seq2)[None, :], 1.0, -1.0
    ... )
    >>> score, ops = needleman_wunsch(
    ...     (1, 1, sm),    # op 0: align
    ...     (0, 1, -1.0),  # op 1: insert (gap in source)
    ...     (1, 0, -1.0),  # op 2: delete (gap in target)
    ... )
    >>> source_idx, target_idx = indices_from_ops(ops, [1, 0, 1], [1, 1, 0])
    """
```

### `indices_from_ops`

Takes caller-provided stride tables instead of a hardcoded enum:

```python
def indices_from_ops(
    ops: npt.NDArray[np.uint8],
    source_strides: Sequence[int],
    target_strides: Sequence[int],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Reconstruct row/column start indices from an ops array.

    Parameters
    ----------
    ops : ndarray of uint8
        Operation codes returned by ``needleman_wunsch``.
    source_strides : sequence of int
        Source stride for each op code (same order as operations passed
        to ``needleman_wunsch``).
    target_strides : sequence of int
        Target stride for each op code.

    Returns
    -------
    source_idx : ndarray of intp
        Starting source index for each operation.
    target_idx : ndarray of intp
        Starting target index for each operation.

    Notes
    -----
    For multi-element operations (stride > 1), the returned index is the
    *first* element consumed.  The subsequent elements are implicit:
    ``source_idx[k], source_idx[k]+1, ..., source_idx[k]+source_stride-1``.

    For zero-stride operations (gaps), the index is the position where
    the gap occurs (i.e. the index of the *next* element that will be
    consumed on that axis).
    """
    rs = np.array(source_strides, dtype=np.intp)
    cs = np.array(target_strides, dtype=np.intp)
    source_idx = np.cumsum(rs[ops]) - rs[ops]
    target_idx = np.cumsum(cs[ops]) - cs[ops]
    return source_idx, target_idx
```

### `iter_alignment`

The current `iter_alignment` signature uses `T | None` to signal gaps:

```python
Iterator[tuple[Op, T | None, S | None]]
```

This cannot represent multi-element ops: SPLIT consumes 1 source element and 2
target elements; MERGE consumes 2 source elements and 1 target element; a future
SWAP consumes 2 of each. The `None` sentinel must be replaced by tuples on
both sides:

```python
Iterator[tuple[int, tuple[T, ...], tuple[S, ...]]]
```

| Op     | `row_items`          | `col_items`          |
| ------ | -------------------- | -------------------- |
| align  | `(row[i],)`          | `(col[j],)`          |
| insert | `()`                 | `(col[j],)`          |
| delete | `(row[i],)`          | `()`                 |
| split  | `(row[i],)`          | `(col[j], col[j+1])` |
| merge  | `(row[i], row[i+1])` | `(col[j],)`          |
| swap   | `(row[i], row[i+1])` | `(col[j+1], col[j])` |

Empty tuple replaces `None`; the arity of each tuple equals the stride of that
side. The op code is now a plain `int` (caller-defined index) rather than the
`Op` enum which is being removed.

`zip(row_items, col_items)` on a SWAP step yields the _crossed_ pairs
`(row[i], col[j+1])` and `(row[i+1], col[j])` — the pairs that actually
contributed to the score — because `col_items` is stored in reversed order.

Python's `match`/`case` keeps consumption ergonomic despite the extra
unpacking:

```python
for op, rs, cs in iter_alignment(ops, source_strides, target_strides, seq1, seq2):
    match op:
        case 0: aligned_row.append(rs[0]); aligned_col.append(cs[0])  # align
        case 1: aligned_row.append("-");   aligned_col.append(cs[0])  # insert
        case 2: aligned_row.append(rs[0]); aligned_col.append("-")    # delete
        case 3: aligned_row.extend([rs[0], rs[0]]); aligned_col.extend(cs)  # split
        case 4: aligned_row.extend(rs); aligned_col.extend([cs[0], cs[0]])  # merge
```

The signature must accept the stride tables so it can reconstruct positions for
arbitrary caller-defined operations — mirroring `indices_from_ops`:

```python
def iter_alignment(
    ops: npt.NDArray[np.uint8],
    source_strides: Sequence[int],
    target_strides: Sequence[int],
    source_seq: Sequence[T],
    target_seq: Sequence[S],
) -> Iterator[tuple[int, tuple[T, ...], tuple[S, ...]]]:
```

This is a **breaking change** from the current implementation (which uses
`T | None` and takes no stride tables). The change should land as part of
Phase 3 alongside the rest of the Python-layer rewrite.

### Convenience constructors (Python-side, optional)

To preserve ergonomics for the common cases, provide thin wrappers in
`pynw/__init__.py` that build the operations tuples:

```python
def align_setup(
    similarity_matrix: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    insert_penalty: float | None = None,
    delete_penalty: float | None = None,
) -> tuple[tuple[int, int, npt.ArrayLike], ...]:
    """Build standard NW operations (align + insert + delete).

    Returns a tuple of operations suitable for unpacking into
    ``needleman_wunsch(*align_setup(sm))``.
    """
    ins = insert_penalty if insert_penalty is not None else gap_penalty
    dlt = delete_penalty if delete_penalty is not None else gap_penalty
    return (
        (1, 1, similarity_matrix),
        (0, 1, ins),
        (1, 0, dlt),
    )


def align_merge_split_setup(
    align_scores: npt.ArrayLike,
    split_scores: npt.ArrayLike,
    merge_scores: npt.ArrayLike,
    *,
    gap_penalty: float = -1.0,
    insert_penalty: float | None = None,
    delete_penalty: float | None = None,
) -> tuple[tuple[int, int, npt.ArrayLike], ...]:
    """Build merge/split NW operations.

    Returns a tuple of operations suitable for unpacking into
    ``needleman_wunsch(*align_merge_split_setup(...))``.
    """
    ins = insert_penalty if insert_penalty is not None else gap_penalty
    dlt = delete_penalty if delete_penalty is not None else gap_penalty
    return (
        (1, 1, align_scores),
        (0, 1, ins),
        (1, 0, dlt),
        (1, 2, split_scores),
        (2, 1, merge_scores),
    )
```

Usage then looks like:

```python
# Concise standard alignment
score, ops = needleman_wunsch(*align_setup(sm, gap_penalty=-1.0))

# Concise merge/split
score, ops = needleman_wunsch(*align_merge_split_setup(
    align_scores, split_scores, merge_scores, gap_penalty=-0.5
))

# Explicit (no helper needed)
score, ops = needleman_wunsch(
    (1, 1, sm),
    (0, 1, -1.0),
    (1, 0, -1.0),
)

# Future 2-to-2 swap -- no API change required
score, ops = needleman_wunsch(
    (1, 1, align_scores),
    (0, 1, -1.0),
    (1, 0, -1.0),
    (2, 2, swap_scores),
)
```

### Removed from public API

- `needleman_wunsch_merge_split` -- subsumed by `needleman_wunsch`
- `Op` enum -- op codes are now caller-defined indices
- `OP_ALIGN`, `OP_INSERT`, `OP_DELETE`, `OP_SPLIT`, `OP_MERGE` constants --
  no longer meaningful as fixed constants

## Rust Implementation

### Data model

```rust
/// A single edit operation definition.
struct Operation {
    source_stride: usize,
    target_stride: usize,
    /// None = scalar (stored in `scalar_score`), Some = matrix view.
    scores: Option<ArrayView2<'_, f64>>,
    scalar_score: f64,
}
```

### General DP kernel (`src/nw_general.rs`)

The general kernel handles arbitrary operations:

```rust
fn fill_general(
    ops: &[Operation],
    n: usize,
    m: usize,
) -> (f64, Array2<u8>) {
    let mut dp = Array2::from_elem((n + 1, m + 1), f64::NEG_INFINITY);
    let mut tb = Array2::<u8>::zeros((n + 1, m + 1));
    dp[[0, 0]] = 0.0;

    // Boundary initialization: only ops with one zero stride can fill edges.
    // For each cell on row 0 or col 0, try applicable operations.

    for i in 0..=n {
        for j in 0..=m {
            if i == 0 && j == 0 { continue; }

            let mut best_score = f64::NEG_INFINITY;
            let mut best_op: u8 = 0;

            for (op_idx, op) in ops.iter().enumerate() {
                let rs = op.source_stride;
                let cs = op.target_stride;

                // Can this operation land at (i, j)?
                if i < rs || j < cs { continue; }

                let prev = dp[[i - rs, j - cs]];
                if prev == f64::NEG_INFINITY { continue; }

                let score = prev + op.score_at(i - rs, j - cs);

                // Tie-break: lowest op index wins (first match with
                // strictly-greater score, or first op if equal).
                // Since we iterate in order 0..n_ops, use >= for
                // "lowest index wins" semantics.
                if score > best_score {
                    best_score = score;
                    best_op = op_idx as u8;
                }
            }

            dp[[i, j]] = best_score;
            tb[[i, j]] = best_op;
        }
    }

    (dp[[n, m]], tb)
}
```

Score lookup for an operation:

```rust
impl Operation {
    fn score_at(&self, i: usize, j: usize) -> f64 {
        match &self.scores {
            Some(matrix) => matrix[[i, j]],
            None => self.scalar_score,
        }
    }
}
```

Traceback becomes stride-aware:

```rust
fn traceback_general(
    tb: &ArrayView2<u8>,
    ops: &[Operation],
) -> Vec<u8> {
    let (n, m) = (tb.nrows() - 1, tb.ncols() - 1);
    let mut result = Vec::with_capacity(n + m);
    let (mut i, mut j) = (n, m);

    while i > 0 || j > 0 {
        let op_idx = tb[[i, j]];
        result.push(op_idx);
        let op = &ops[op_idx as usize];
        i -= op.source_stride;
        j -= op.target_stride;
    }

    result.reverse();
    result
}
```

### Fast-path dispatch

The current specialized kernels in `nw.rs` and `nw_merge_split.rs` are
significantly faster for their respective cases because the compiler can
optimize fixed loop bodies. The general kernel pays per-cell overhead for
the dynamic operation list.

**Strategy: dispatch at the PyO3 layer based on the operation signature.**

```rust
// In lib.rs, after parsing operations:
let ops_signature: Vec<(usize, usize, bool)> = operations
    .iter()
    .map(|op| (op.source_stride, op.target_stride, op.scores.is_some()))
    .collect();

match ops_signature.as_slice() {
    // Standard NW: align(1,1,matrix) + insert(0,1,scalar) + delete(1,0,scalar)
    [(1, 1, true), (0, 1, false), (1, 0, false)] => {
        fast_path_standard(...)  // reuses existing nw.rs
    }
    // Merge/split NW
    [(1, 1, true), (0, 1, false), (1, 0, false), (1, 2, true), (2, 1, true)] => {
        fast_path_merge_split(...)  // reuses existing nw_merge_split.rs
    }
    // Everything else
    _ => {
        general_kernel(...)
    }
}
```

The fast paths call the existing `fill_matrices` + `traceback` code, then
convert the result into the unified `(f64, Vec<u8>)` return type. The
conversion happens during traceback (O(n+m)) and adds negligible cost.

#### Adapting `nw.rs` for the fast path

The existing `traceback_indices` returns `(Vec<isize>, Vec<isize>)`. We
need a `traceback_ops` variant that returns `Vec<u8>`:

```rust
fn traceback_ops(tb: &ArrayView2<Direction>) -> Vec<u8> {
    // Same logic as traceback_indices, but emits op codes:
    //   Diagonal -> 0 (align)
    //   Left     -> 1 (insert)
    //   Up       -> 2 (delete)
    ...
}
```

#### Adapting `nw_merge_split.rs` for the fast path

The existing `traceback_ops` returns `Vec<EditOp>`. Convert to `Vec<u8>`
with the mapping: Align=0, Insert=1, Delete=2, Split=3, Merge=4 -- which
matches the operation order in the fast-path signature check.

### Dimension inference

Sequence lengths `n` and `m` are inferred from the first matrix-valued
operation:

```rust
fn infer_dimensions(operations: &[Operation]) -> Result<(usize, usize), String> {
    for op in operations {
        if let Some(ref matrix) = op.scores {
            let n = matrix.nrows() + op.source_stride - 1;
            let m = matrix.ncols() + op.target_stride - 1;
            return Ok((n, m));
        }
    }
    Err("At least one operation must have a matrix score".into())
}
```

All subsequent matrix operations are validated against the inferred `(n, m)`:

```rust
fn validate_shapes(operations: &[Operation], n: usize, m: usize) -> Result<(), String> {
    for (idx, op) in operations.iter().enumerate() {
        if let Some(ref matrix) = op.scores {
            let expected_rows = n + 1 - op.source_stride;
            let expected_cols = m + 1 - op.target_stride;
            if matrix.nrows() != expected_rows || matrix.ncols() != expected_cols {
                return Err(format!(
                    "Operation {}: expected shape ({}, {}), got ({}, {})",
                    idx, expected_rows, expected_cols,
                    matrix.nrows(), matrix.ncols()
                ));
            }
        }
    }
    Ok(())
}
```

### PyO3 binding layer (`src/lib.rs`)

The binding parses `*operations` from Python tuples:

```rust
#[pyfunction]
#[pyo3(signature = (*operations, check_finite=false))]
fn needleman_wunsch<'py>(
    py: Python<'py>,
    operations: Vec<Bound<'py, PyTuple>>,
    check_finite: bool,
) -> PyResult<(f64, Bound<'py, PyArray1<u8>>)> {
    if operations.is_empty() {
        return Err(PyValueError::new_err("At least one operation required"));
    }
    if operations.len() > 255 {
        return Err(PyValueError::new_err("At most 255 operations supported"));
    }

    // Parse each (source_stride, target_stride, scores_or_scalar) tuple
    let mut parsed_ops = Vec::with_capacity(operations.len());
    let mut matrix_views = Vec::new();  // keep PyReadonlyArray2 alive

    for (idx, tup) in operations.iter().enumerate() {
        let source_stride: usize = tup.get_item(0)?.extract()?;
        let target_stride: usize = tup.get_item(1)?.extract()?;
        let scores_obj = tup.get_item(2)?;

        if let Ok(scalar) = scores_obj.extract::<f64>() {
            // Scalar operation
            parsed_ops.push(Operation {
                source_stride,
                target_stride,
                scores: None,
                scalar_score: scalar,
            });
        } else {
            // Matrix operation
            let py_array = as_pyarray(py, &scores_obj)?;
            // ... store view, push Operation with scores: Some(...)
        }
    }

    // Infer n, m from first matrix op
    let (n, m) = infer_dimensions(&parsed_ops)?;
    validate_shapes(&parsed_ops, n, m)?;

    // check_finite validation ...

    // Dispatch to fast path or general kernel
    let (score, ops) = dispatch(&parsed_ops, n, m);

    Ok((score, ops.into_pyarray(py)))
}
```

## File Changes

### Rust (`src/`)

| File                    | Action      | Description                                                                                          |
| ----------------------- | ----------- | ---------------------------------------------------------------------------------------------------- |
| `src/nw_general.rs`     | **New**     | General DP kernel for arbitrary operations                                                           |
| `src/nw.rs`             | **Modify**  | Add `traceback_ops` returning `Vec<u8>` alongside existing code                                      |
| `src/nw_merge_split.rs` | **Modify**  | Adapt traceback to return `Vec<u8>` with correct op indices                                          |
| `src/lib.rs`            | **Rewrite** | New `needleman_wunsch` signature, operation parsing, dispatch, remove `needleman_wunsch_merge_split` |

### Python (`pynw/`)

| File               | Action      | Description                                                                                               |
| ------------------ | ----------- | --------------------------------------------------------------------------------------------------------- |
| `pynw/__init__.py` | **Modify**  | Update `__all__`, add `align_setup` / `align_merge_split_setup`, remove `Op` re-export                    |
| `pynw/_native.pyi` | **Rewrite** | New `needleman_wunsch` stub, remove `needleman_wunsch_merge_split` and `OP_*` constants                   |
| `pynw/_ops.py`     | **Rewrite** | New `indices_from_ops(ops, source_strides, target_strides)`, remove `Op` enum and hardcoded stride tables |

### Tests (`tests/`)

| File                                     | Action        | Description                                                                                                             |
| ---------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `tests/helpers.py`                       | **Modify**    | Update `nw_score`, `recompute_score_from_indices`, `assert_structural_invariants` for ops-based return                  |
| `tests/test_needleman_wunsch_api.py`     | **Modify**    | Migrate all tests to new API; every `score, ri, ci = ...` becomes `score, ops = ...; ri, ci = indices_from_ops(...)`    |
| `tests/test_edit_distance.py`            | **Modify**    | Same migration pattern                                                                                                  |
| `tests/test_edit_distance_hypothesis.py` | **Modify**    | Same migration pattern                                                                                                  |
| `tests/test_examples.py`                 | **No change** | Tests run code from docs; will pass once docs are updated                                                               |
| `tests/test_benchmark.py`                | **Modify**    | Update benchmark calls                                                                                                  |
| `tests/test_general_ops.py`              | **New**       | Tests for the general kernel: arbitrary strides, mixed scalar/matrix, dimension inference, shape validation, edge cases |
| `tests/test_merge_split.py`              | **New**       | Tests for merge/split alignment (currently missing entirely)                                                            |

### Docs

| File            | Action     | Description                                                  |
| --------------- | ---------- | ------------------------------------------------------------ |
| `README.md`     | **Modify** | Update quick start example, edit-distance table, API section |
| `docs/USAGE.md` | **Modify** | Update code examples                                         |
| `CLAUDE.md`     | **Modify** | Update architecture description                              |

## Implementation Order

### Phase 1: Rust general kernel

1. Create `src/nw_general.rs` with `Operation`, `fill_general`,
   `traceback_general`, dimension inference, and shape validation.
2. Add `traceback_ops` to `src/nw.rs` (returns `Vec<u8>` instead of
   index arrays).
3. Adapt `src/nw_merge_split.rs` traceback to return `Vec<u8>`.
4. Unit-test the general kernel from Rust (cargo test).

### Phase 2: PyO3 binding

1. Rewrite `src/lib.rs`: new `needleman_wunsch` signature with operation
   tuple parsing, dispatch logic (fast paths + general fallback).
2. Remove `needleman_wunsch_merge_split` from the module.
3. Remove `OP_*` constants from module init.

### Phase 3: Python layer

1. Rewrite `pynw/_ops.py`: new `indices_from_ops(ops, source_strides, target_strides)`.
2. Rewrite `pynw/_native.pyi`: new stub for `needleman_wunsch`.
3. Update `pynw/__init__.py`: add convenience constructors, update `__all__`.

### Phase 4: Test migration

1. Update `tests/helpers.py` for ops-based returns.
2. Migrate `tests/test_needleman_wunsch_api.py`.
3. Migrate `tests/test_edit_distance.py` and `test_edit_distance_hypothesis.py`.
4. Write `tests/test_general_ops.py` (general kernel coverage).
5. Write `tests/test_merge_split.py` (merge/split coverage -- currently zero).

### Phase 5: Documentation

1. Update `README.md` examples.
2. Update `docs/USAGE.md` examples.
3. Update `CLAUDE.md`.

## Migration Examples

### Before (standard NW)

```python
from pynw import needleman_wunsch

score, source_idx, target_idx = needleman_wunsch(sm, gap_penalty=-1.0)
for i, j in zip(source_idx, target_idx):
    if i >= 0 and j >= 0:
        print(f"align {i} <-> {j}")
    elif i >= 0:
        print(f"delete {i}")
    else:
        print(f"insert {j}")
```

### After (standard NW)

```python
from pynw import needleman_wunsch, indices_from_ops, align_setup

ALIGN, INSERT, DELETE = 0, 1, 2
SOURCE_STRIDES = [1, 0, 1]
TARGET_STRIDES = [1, 1, 0]

score, ops = needleman_wunsch(*align_setup(sm, gap_penalty=-1.0))
source_idx, target_idx = indices_from_ops(ops, SOURCE_STRIDES, TARGET_STRIDES)

for op, i, j in zip(ops, source_idx, target_idx):
    if op == ALIGN:
        print(f"align {i} <-> {j}")
    elif op == DELETE:
        print(f"delete {i}")
    elif op == INSERT:
        print(f"insert {j}")
```

### Before (merge/split)

```python
from pynw import needleman_wunsch_merge_split, Op, indices_from_ops

score, ops = needleman_wunsch_merge_split(
    align_scores, split_scores, merge_scores, gap_penalty=-0.5
)
source_idx, target_idx = indices_from_ops(ops)
```

### After (merge/split)

```python
from pynw import needleman_wunsch, indices_from_ops, align_merge_split_setup

score, ops = needleman_wunsch(*align_merge_split_setup(
    align_scores, split_scores, merge_scores, gap_penalty=-0.5
))
# Op indices: 0=align, 1=insert, 2=delete, 3=split, 4=merge
SOURCE_STRIDES = [1, 0, 1, 1, 2]
TARGET_STRIDES = [1, 1, 0, 2, 1]
source_idx, target_idx = indices_from_ops(ops, SOURCE_STRIDES, TARGET_STRIDES)
```

### Future (2-to-2 swap, no API change)

```python
score, ops = needleman_wunsch(
    (1, 1, align_scores),    # 0: align
    (0, 1, -1.0),            # 1: insert
    (1, 0, -1.0),            # 2: delete
    (2, 2, swap_scores),     # 3: swap
)
SOURCE_STRIDES = [1, 0, 1, 2]
TARGET_STRIDES = [1, 1, 0, 2]
source_idx, target_idx = indices_from_ops(ops, SOURCE_STRIDES, TARGET_STRIDES)
```

## Performance Considerations

### Why this doesn't sacrifice speed

The fill phase is O(nm) and dominates runtime. Traceback is O(n+m). The
return format (ops vs indices) is irrelevant to performance.

For the common cases (standard 3-op NW, 5-op merge/split), the dispatch
layer detects the operation signature and routes to the existing
specialized kernels. These kernels have fixed loop bodies that the
compiler optimizes aggressively (branch prediction, unrolling, SIMD).

The general kernel is only used for non-standard operation sets. It pays
a per-cell cost for iterating over the dynamic operation list, but this
is bounded by the number of operations (typically 3-6) and still runs in
O(nm) time.

### Benchmark validation

After implementation, verify with `pixi run test -m benchmark`:

- Standard NW performance should be identical to current (same Rust code path)
- Merge/split performance should be identical (same Rust code path)
- General kernel with standard ops should be within 2-3x of the fast path
  (acceptable since it's only used for non-standard configurations)

## Tie-Breaking

The new API defines tie-breaking as **lowest op index wins**. This means
the caller controls tie-breaking priority through argument order. For the
standard fast path, this maps to:

- Op 0 (align) > Op 1 (insert) > Op 2 (delete)

Which matches the current behavior (Diagonal > Left > Up... wait, the
current code uses Diagonal > Up > Left). We need to ensure the fast path
preserves the existing tie-breaking when called through the new API.

**Resolution:** The fast path preserves its internal tie-breaking
(Diagonal > Up > Left for standard NW, Align > Merge > Split > Delete >
Insert for merge/split). The mapping from internal priority to op index
priority is handled by the dispatch layer. Since `align_setup` always
puts align first, the dominant tie-break (prefer alignment over gaps)
is preserved. The secondary tie-break between insert and delete is a
minor behavioral detail that most users don't depend on -- but it should
be documented.

## Open Questions

1. **Should `indices_from_ops` move to Rust?** The current NumPy
   implementation is a one-liner (`cumsum` + index), so the overhead is
   negligible. Moving it to Rust adds complexity for no measurable gain.
   Recommendation: keep in Python.

2. **Should the convenience constructors return stride tables too?**
   Currently `align_setup` returns just the operations tuple. The caller
   still needs to know the stride tables for `indices_from_ops`. We could
   return a richer object, but that adds API surface. Recommendation: for
   the common cases, document the stride tables. Advanced users building
   custom operations already know their strides.

3. **Maximum number of operations?** Op codes are `u8`, so the hard limit
   is 256. In practice, more than ~10 operations would make the general
   kernel slow. Validate `len(operations) <= 255` at the binding layer.

4. **Should we support `(0, 0, ...)` operations?** A (0, 0) operation
   consumes nothing and would cause an infinite loop. Validate that every
   operation has `source_stride + target_stride > 0`.
