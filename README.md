# pynw

[![Check](https://github.com/chrisdeutsch/pynw/actions/workflows/check.yml/badge.svg)](https://github.com/chrisdeutsch/pynw/actions/workflows/check.yml)
[![Build](https://github.com/chrisdeutsch/pynw/actions/workflows/build.yml/badge.svg)](https://github.com/chrisdeutsch/pynw/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/pynw)](https://pypi.org/project/pynw/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Rust-accelerated [Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
global sequence alignment for caller-supplied similarity matrices. Python
bindings are built with [PyO3](https://pyo3.rs).

Align two ordered sequences given any precomputed pairwise similarity matrix.
Unlike string-distance or bioinformatics libraries, which are designed around
specific alphabets and scoring rules, `pynw` accepts whatever scores you
provide: cosine similarity of embeddings, model outputs, distance metrics, or
exact match. Semantically related tokens (e.g. "cat" and "feline") can align
with low penalty; tokens with no suitable counterpart are assigned a gap.

## Features

- **Fast:** Alignment runs in $O(nm)$ time; a `1000×1000` matrix takes <10 ms on modern CPUs.
- **NumPy-first:** Accepts NumPy arrays directly — no intermediate Python objects.
- **Domain-agnostic:** Operates on a precomputed similarity matrix, so any scoring function (distance metrics, semantic similarity, etc.) works out of the box.
- **Asymmetric gaps:** Optionally set separate insert and delete penalties.

## Installation

Requires Python 3.10+ and NumPy 1.21+. Prebuilt wheels for Linux, macOS, and
Windows are published on PyPI.

```sh
pip install pynw
```

On platforms without a prebuilt wheel, pip will build from the source
distribution. This requires a [Rust toolchain](https://rustup.rs/) (1.85+).

## Quick start

Align two token sequences using precomputed cosine similarity from GloVe
embeddings. Semantically similar words align even without an exact match, and
words with no good counterpart are gapped out:

```python
import numpy as np
from pynw import needleman_wunsch, alignment_indices

hypothesis = np.array(
    ["clever", "sneaky", "fox", "leaped"]
)
reference = np.array(
    ["sly", "fox", "jumped", "across"]
)

# Cosine similarity from GloVe (glove-wiki-gigaword-50)
similarity_matrix = np.array([
    # sly     fox     jumped  across
    [ 0.65,   0.25,   0.06,   0.20],  # clever
    [ 0.57,   0.06,  -0.14,  -0.05],  # sneaky
    [ 0.26,   1.00,   0.30,   0.41],  # fox
    [-0.00,   0.07,   0.77,   0.35],  # leaped
])

# gap_penalty=-0.5 means: prefer a gap over aligning tokens with cosine similarity < 0.5
score, ops = needleman_wunsch(similarity_matrix, gap_penalty=-0.5)
src_idx, tgt_idx = alignment_indices(ops)

# Reconstruct aligned sequences; masked positions are gaps
aligned_hyp = np.ma.array(hypothesis).take(src_idx).filled("-")
aligned_ref = np.ma.array(reference).take(tgt_idx).filled("-")

print(f"Score: {round(score, 2)}")
for h, r in zip(aligned_hyp, aligned_ref):
    print(f"  {h:10s}  {r}")
# Score: 1.42
#   clever     sly       <- semantic match
#   sneaky     -         <- deleted
#   fox        fox       <- exact match
#   leaped     jumped    <- semantic match
#   -          across    <- inserted
```

## Details

### Precomputed similarity matrix

`pynw` takes a precomputed `(n, m)` similarity matrix rather than a scoring
function. This means the entire alignment runs in compiled Rust code with no
Python callbacks, and you can build scores with vectorized NumPy operations
rather than element-wise Python loops.

The trade-off is that you must allocate the full matrix up front, which uses
$O(nm)$ memory even when the scoring rule could be expressed more compactly.

### Scoring

The total alignment score is the sum of similarity-matrix entries for matched
positions and gap penalties for insertions/deletions. Gap penalties are
typically negative. By default a single `gap_penalty` applies to both
directions; set `insert_penalty` and/or `delete_penalty` to penalise them
independently.

When multiple alignments achieve the same optimal score, `pynw` breaks ties
deterministically: `Align > Delete > Insert`.

### Usage guide

Use `needleman_wunsch_score` when you only need the alignment score, for
example when ranking or filtering many sequence pairs. It skips the traceback
entirely, using O(m) memory instead of O(nm):

```python
from pynw import needleman_wunsch_score

score = needleman_wunsch_score(similarity_matrix, gap_penalty=-0.5)
```

Use `needleman_wunsch` when you need the edit operations. Ops alone are
sufficient for aggregate counts:

```python
from pynw import EditOp, needleman_wunsch

score, ops = needleman_wunsch(similarity_matrix, gap_penalty=-0.5)

n_aligned = np.sum(ops == EditOp.Align)
n_inserted = np.sum(ops == EditOp.Insert)
n_deleted = np.sum(ops == EditOp.Delete)
```

Use `alignment_indices` when you need to compare the aligned elements
themselves, for example to count substitutions among aligned positions:

```python
from pynw import alignment_indices

src_idx, tgt_idx = alignment_indices(ops)
aligned = ops == EditOp.Align
substitutions = np.sum(seq1[src_idx.data[aligned]] != seq2[tgt_idx.data[aligned]])
```

### Edit-distance parameterizations

Needleman-Wunsch can reproduce common metrics with the right similarity-matrix
values and gap penalty:

| Metric               | `S[i,j]` match | `S[i,j]` mismatch | `gap_penalty` | NW score equals |
| -------------------- | -------------- | ----------------- | ------------- | --------------- |
| Levenshtein distance | 0              | -1                | -1            | `-distance`     |
| Indel distance       | 0              | -2                | -1            | `-distance`     |
| LCS length           | 1              | 0                 | 0             | `lcs_length`    |
| Hamming distance     | 0              | -1                | `-(n+1)`      | `-distance`     |

For Hamming distance, strings must have equal length.

## API

Full API documentation is available at
[chrisdeutsch.github.io/pynw](https://chrisdeutsch.github.io/pynw/pynw.html).
To browse locally:

```sh
pixi run docs          # generate static HTML in site/
pixi run docs-serve    # serve live docs in the browser
```

## Development

This repository uses [pixi](https://pixi.sh) for development:

```sh
pixi install
pixi run build            # build the Rust extension
pixi run test             # run deterministic tests
pixi run lint             # run all pre-commit checks (ruff, cargo fmt, prettier, markdownlint, taplo, actionlint)
pixi run check            # run all pre-push checks (cargo clippy, mypy)
pixi run docs             # generate API docs in site/
pixi run docs-serve       # serve API docs in the browser
```

Linting and formatting are managed by [lefthook](https://github.com/evilmartians/lefthook)
and run automatically as git hooks.

## Related projects

- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz):
  Highly optimized string distances (Levenshtein, Jaro-Winkler, etc.) with
  scoring, edit operations, and alignment. The better choice when you only
  need standard string metrics.
- [sequence-align](https://github.com/kensho-technologies/sequence_align):
  Rust-accelerated Needleman-Wunsch and Hirschberg for token sequences with
  built-in match/mismatch scoring.
- [`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html):
  Solves unconstrained bipartite matching ($O(N^3)$) where order does not
  matter.
- [BioPython `Bio.Align.PairwiseAligner`](https://biopython.org/docs/dev/Tutorial/chapter_align.html):
  Needleman-Wunsch/Smith-Waterman for biological sequences with
  alphabet-based substitution matrices (built-in and custom).

## License

[MIT](LICENSE)
