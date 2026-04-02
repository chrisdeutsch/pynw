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

## User experience

- [ ] **Return a `NamedTuple` instead of a plain tuple** — `needleman_wunsch` returns
      `tuple[float, NDArray[uint8]]`; a `NamedTuple` with `.score` and `.ops` fields is
      backward-compatible (still unpacks as a tuple) but adds discoverability and IDE support.

- [ ] **`score_matrix` helper** — every usage example requires the caller to build an (n, m)
      similarity matrix manually. A Python-side `score_matrix(source, target, scorer)` that takes
      two sequences and a `(a, b) -> float` callable would cover the common case. Power users who
      precompute matrices keep using `needleman_wunsch` directly. Keep the implementation as a
      simple element-wise Python loop — the DP is the expensive part (handled in Rust), and
      domain-specific fast paths (cdist, vectorized NumPy, embedding dot products) all look
      different. Users who care about matrix construction performance build matrices directly.
      Document this tradeoff.

- [ ] **`NamedTuple` for `iter_alignment` steps** — the yielded `(EditOp, T|None, T|None)` triples
      would benefit from `.op`, `.source`, `.target` fields instead of positional-only access.

- [ ] **High-level `align()` function** — convenience layer combining matrix construction,
      `needleman_wunsch`, and result iteration into one call:
      `align(source, target, scorer=..., gap_penalty=...)`. Returns a rich result object that is
      iterable over alignment steps. Depends on the items above.
