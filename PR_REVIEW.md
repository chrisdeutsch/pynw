# Review: `needleman_wunsch_2` (merge/split NW)

## Algorithm Summary

Extends standard Needleman-Wunsch with two new DP transitions:

| Op     | Transition                              | Meaning       |
| ------ | --------------------------------------- | ------------- |
| Match  | `dp[i-1][j-1] + match_scores[i-1][j-1]` | 1 row : 1 col |
| Delete | `dp[i-1][j] + penalty_delete`           | 1 row : gap   |
| Insert | `dp[i][j-1] + penalty_insert`           | gap : 1 col   |
| Merge  | `dp[i-2][j-1] + merge_scores[i-2][j-1]` | 2 row : 1 col |
| Split  | `dp[i-1][j-2] + split_scores[i-1][j-2]` | 1 row : 2 col |

## Correctness

The core DP and traceback are correct:

- Merge/split guards (`i >= 2` / `j >= 2` with `NEG_INFINITY` fallback) prevent impossible transitions.
- Traceback can never underflow — merge only appears in cells where `i >= 2`, split where `j >= 2`.
- Boundary initialization (cumulative gap penalties along first row/column) is correct.
- Empty inputs (`n=0` or `m=0`) produce the right score and ops.
- Gap penalty mapping through PyO3 (`gap_penalty_row` → `penalty_insert`, `gap_penalty_col` → `penalty_delete`) is consistent with the original `nw.rs`.
- `check_finite` validates all three score matrices.
- `indices_from_ops` stride tables and cumsum logic are correct.

## Actionable Items

### 1. Add tests (high priority)

There are zero tests for `needleman_wunsch_2`, `indices_from_ops`, or `Op`. Suggested cases:

- **Hand-traced example**: a small (3×3 or similar) case with a known optimal alignment that exercises merge and/or split.
- **Degeneration to standard NW**: when `merge_scores` and `split_scores` are filled with `-inf`, the result should match the original `needleman_wunsch` (same score and equivalent alignment).
- **Pure-gap edges**: `n=0` (all inserts) and `m=0` (all deletes).
- **Merge-optimal case**: a scoring scheme where merging two rows into one column is strictly better than matching or gapping.
- **Split-optimal case**: same idea for split.
- **`indices_from_ops` round-trip**: verify that reconstructed indices consume every row and column element exactly once (`row_strides.sum() == n`, `col_strides.sum() == m`).
- **Score consistency**: recompute the score by walking the ops array against the score matrices and verify it equals the returned score.

### 2. Validate score matrix dimensions (medium priority)

`fill_matrices` derives `(n, m)` from `match_scores.shape` only (`nw2.rs:39`) and trusts that the other matrices are large enough. Access patterns require:

| Matrix         | Accessed indices | Minimum shape |
| -------------- | ---------------- | ------------- |
| `match_scores` | `[0..n, 0..m]`   | `(n, m)`      |
| `merge_scores` | `[0..n-1, 0..m]` | `(n-1, m)`    |
| `split_scores` | `[0..n, 0..m-1]` | `(n, m-1)`    |

Undersized matrices cause an ndarray index-out-of-bounds panic with no useful context. Options:

- **Check in Rust** at the top of `fill_matrices` and return a descriptive error.
- **Check in the PyO3 layer** (`lib.rs`) before calling into `nw2`, returning a `PyValueError`.

Either way, decide whether the canonical shape is `(n, m)` for all three (simplest for callers, last row/column unused for merge/split) or the minimal shapes above, and document it.

### 3. Document the public API (medium priority)

The Python stub (`_native.pyi:104-113`) has no docstring and the Rust doc comment is a single line. At minimum, document:

- **Score matrix semantics**: `merge_scores[r, c]` is the score for merging row elements `r` and `r+1` into column element `c`; `split_scores[r, c]` is the score for splitting row element `r` into column elements `c` and `c+1`.
- **Expected shapes** (see above).
- **Return value**: the ops array contains `Op` codes (uint8); use `indices_from_ops` to get index arrays.
- **`indices_from_ops` output semantics**: `row_idx[k]` / `col_idx[k]` is the _starting_ row/column element for operation `k`. This differs from the original NW's `-1`-for-gap convention — gap information is encoded in the ops array, not the indices.

### 4. Document tie-breaking order (low priority)

The `candidate_ops` array ordering (`nw2.rs:70-76`) produces this tie-breaking priority:

Match > Merge > Split > Delete > Insert

This is consistent with the original NW's "Diagonal > Up > Left" for the three shared operations, with merge and split slotted in between. Worth a code comment or docstring note so callers know what to expect with co-optimal alignments.

### 5. Align internal naming (low priority)

The Rust function uses `penalty_insert` / `penalty_delete` while the Python API uses `gap_penalty_row` / `gap_penalty_col`. The mapping is correct but the naming gap could confuse maintainers. Consider renaming the Rust parameters to match, or adding a comment at the call site in `lib.rs:247-253` noting the correspondence.

## Related Algorithms

The merge/split extension is an instance of a well-studied family of alignment algorithms that allow many-to-one and one-to-many correspondences. The transition set `{1:1, 1:0, 0:1, 2:1, 1:2}` used here is a subset of the "bead model" from sentence alignment in computational linguistics.

### Direct precedents

- **Gale & Church (1993)**, "A Program for Aligning Sentences in Bilingual Corpora", _Computational Linguistics_ 19(1), 75–102. The most directly analogous algorithm. Uses the same DP structure with bead types `{1:1, 1:0, 0:1, 2:1, 1:2, 2:2}` to align sentences across parallel corpora. The only difference is the additional `2:2` bead type (two source sentences aligned to two target sentences), which `needleman_wunsch_2` omits. Gale–Church scores beads by modeling sentence length ratios rather than accepting arbitrary score matrices.

- **Brown, Lai & Mercer (1991)**, "Aligning Sentences in Parallel Corpora", _Proceedings of ACL_, 169–176. Earlier formulation of the same DP with `{1:1, 2:1, 1:2, 2:2}` bead types for bilingual sentence alignment. Demonstrated that the approach is tractable at corpus scale.

### Base algorithm

- **Needleman & Wunsch (1970)**, "A General Method Applicable to the Search for Similarities in the Amino Acid Sequence of Two Proteins", _Journal of Molecular Biology_ 48(3), 443–453. The original global alignment algorithm with `{1:1, 1:0, 0:1}` transitions. `needleman_wunsch_2` is a direct generalization.

### Related extensions

- **Gotoh (1982)**, "An Improved Algorithm for Matching Biological Sequences", _Journal of Molecular Biology_ 162(3), 705–708. Extends NW in a different direction — affine gap penalties (separate gap-open and gap-extend costs) — using a three-matrix DP. Orthogonal to the merge/split extension and could in principle be combined with it.

- **Sakoe & Chiba (1978)**, "Dynamic Programming Algorithm Optimization for Spoken Word Recognition", _IEEE Transactions on ASSP_ 26(1), 43–49. Dynamic Time Warping (DTW) for time series alignment. DTW allows many-to-one mappings via slope constraints but uses a different cost model (distance minimization without explicit gap penalties). The merge/split transitions serve a similar purpose to DTW's asymmetric step patterns.

### Modern sentence/paragraph aligners

Several practical tools implement the Gale–Church bead model or close variants:

- **Hunalign** (Varga et al., 2005) — combines sentence-length and dictionary-based scoring with the Gale–Church DP.
- **Bleualign** (Sennrich & Volk, 2010) — uses MT output to guide alignment; the underlying DP is a bead-model variant.
- **Vecalign** (Thompson & Koehn, 2019) — uses sentence embeddings as the score function fed into the same DP structure.

These are useful reference implementations if you want to compare behavior or borrow test cases.
