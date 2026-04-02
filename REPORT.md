# Code Review: `needleman_wunsch_merge_split`

Reviewed files: `src/nw_merge_split.rs`, `src/lib.rs`, `pynw/_native.pyi`,
`pynw/_ops.py`.

---

## Summary

The algorithm is logically correct and the code is clean, but the feature ships
without any test coverage, without input validation for the new score matrices,
and with undocumented API conventions that callers cannot discover without
reading the source.

---

## Issues

### 1. No test coverage (high)

There are no tests anywhere for `needleman_wunsch_merge_split` or
`indices_from_ops`. The existing suite covers only `needleman_wunsch`. Because
the merge/split DP involves two non-standard recurrences and a new traceback
path, correctness is entirely unverified.

Minimum required coverage:

- A hand-computed example exercising each of the five op types.
- `indices_from_ops` round-trips: given known ops, check reconstructed indices.
- Structural invariants analogous to `assert_structural_invariants` in
  `tests/helpers.py` (every source element accounted for exactly once, every
  target element accounted for exactly once).
- Property test: reported score equals score recomputed by walking the ops.

---

### 2. No dimension validation for `split_scores` and `merge_scores` (high)

`fill_matrices` silently assumes:

| Matrix         | Expected shape |
| -------------- | -------------- |
| `align_scores` | `(n, m)`       |
| `merge_scores` | `(n-1, m)`     |
| `split_scores` | `(n, m-1)`     |

The code derives `n` and `m` only from `align_scores.nrows()` and
`align_scores.ncols()`. If a caller passes `merge_scores` or `split_scores`
with the wrong shape, the out-of-bounds array access in the inner loop panics
with an ndarray message that gives no hint of which matrix is at fault or what
the expected shape was.

The PyO3 wrapper in `lib.rs` validates `check_finite` but never checks
shapes. Validation should be added at the Rust boundary and surfaced as a
Python `ValueError` with a message that states the expected shape.

---

### 3. `indices_from_ops` semantics undocumented (medium)

`pynw/_ops.py` `indices_from_ops` returns one index per op, but MERGE and
SPLIT ops consume two source or target elements respectively:

- **MERGE** at op `k`: `row_idx[k]` is the index of the _first_ source element;
  `row_idx[k] + 1` is the second. The second is implicit and not surfaced.
- **SPLIT** at op `k`: `col_idx[k]` is the index of the _first_ target element;
  `col_idx[k] + 1` is the second.

The function docstring says only "Reconstruct row and column indices from an
ops array", giving no indication of this convention. Callers who do not read
the source will misinterpret MERGE and SPLIT entries.

Additionally, the index semantics here differ from `needleman_wunsch`, which
uses `-1` to signal a gap. `indices_from_ops` uses a stride of 0 for
INSERT/DELETE, meaning the returned index is the _current position_, not a
sentinel. This difference is also undocumented.

---

### 4. `needleman_wunsch_merge_split` has no docstring (medium)

The stub in `pynw/_native.pyi` has `...` in place of a docstring. The Rust
function in `lib.rs` has only `/// Experimental Needleman-Wunsch with merges
and splits`. Callers have no documentation for:

- What each score matrix represents.
- The required shape of each score matrix relative to the others.
- What the returned `uint8` array contains and how to interpret it.
- Tie-breaking behaviour.

The existing `needleman_wunsch` docstring is a good template to follow.

---

### 5. Undocumented tie-breaking in `fill_matrices` (low)

The candidate op ordering in the inner loop:

```rust
let candidate_ops = [
    (EditOp::Insert, insert),
    (EditOp::Delete, delete),
    (EditOp::Split, split),
    (EditOp::Merge, merge),
    (EditOp::Align, align),   // last = wins ties
];
```

`Iterator::max_by` returns the _last_ maximum, so on equal scores the priority
is `Align > Merge > Split > Delete > Insert`. This is deterministic and
probably correct, but it is not documented anywhere, unlike the tie-breaking
note in `needleman_wunsch`.

---

### 6. `debug_assert` in traceback not active in release builds (low)

`traceback_ops` uses `debug_assert!` to guard against impossible transitions
(e.g., a MERGE op when `i < 2`). In a release build these assertions are
removed. A corrupt or inconsistent traceback matrix would cause a `usize`
underflow and then an out-of-bounds panic, with no diagnostic message.

Since the traceback is always internally generated (not caller-supplied), this
is unlikely to fire in practice. Nonetheless, the guards would be more useful
as regular `assert!` calls, or at minimum with a comment explaining why
`debug_assert` is sufficient here.

---

### 7. Unresolved TODO in `as_pyarray` (low)

`src/lib.rs:18` has `// TODO: Check that array is C-contiguous?`. This applies
equally to the existing `needleman_wunsch` and the new function. Non-contiguous
or Fortran-order arrays are silently accepted; the DP will produce wrong results
if the layout assumption is violated. This is pre-existing but worth tracking.
