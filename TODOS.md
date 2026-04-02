# TODOs

## merge-split NW

- [ ] **Add tests** — no tests exist for `needleman_wunsch_merge_split` or merge/split ops in
      `indices_from_ops`. Minimum: hand-traced example exercising all five op types; round-trip
      invariants (`row_strides.sum() == n`, `col_strides.sum() == m`); score recomputed from ops
      equals returned score; pure-gap edge cases (`n=0`, `m=0`).

- [ ] **Fix `indices_from_ops` for merge/split ops** — `_ops.py` stride tables are sized
      `max(EditOp)+1 = 3`; indexing with `Split=3` or `Merge=4` raises `IndexError`. The function
      needs stride entries for `OP_SPLIT` (source×1, target×2) and `OP_MERGE` (source×2, target×1),
      plus updated docstring explaining that merge/split indices point to the _first_ element consumed.

- [ ] **Validate `merge_scores`/`split_scores` shapes** — `lib.rs` checks finiteness but not
      shapes. Wrong-shaped inputs panic in Rust with a cryptic ndarray message. Add a `PyValueError`
      at the PyO3 boundary stating the expected shapes (`merge_scores: (n-1, m)`,
      `split_scores: (n, m-1)`) relative to `align_scores: (n, m)`.
