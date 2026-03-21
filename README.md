# pynw

[![Check](https://github.com/chrisdeutsch/pynw/actions/workflows/check.yml/badge.svg)](https://github.com/chrisdeutsch/pynw/actions/workflows/check.yml)
[![Release](https://github.com/chrisdeutsch/pynw/actions/workflows/release.yml/badge.svg)](https://github.com/chrisdeutsch/pynw/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Rust-accelerated [Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
global sequence alignment for precomputed similarity matrices. Python
bindings are built with [PyO3](https://pyo3.rs).

Align two ordered sequences, allowing for gaps (insertions/deletions), given any
precomputed pairwise similarity matrix. Useful for strings, time series, token
sequences, or any domain where element order matters.

## Features

- **Fast:** The alignment runs in $O(NM)$ time; a `1000x1000` matrix takes < 10ms on modern CPUs.
- **NumPy-First:** The interface accepts NumPy arrays directly and requires no intermediate Python objects.
- **Domain-Agnostic:** Operates on a precomputed similarity matrix, so any scoring function (e.g. distance metrics, semantic similarity) works out of the box.

## Installation

Requires Python 3.10+ and NumPy 1.16+. Prebuilt wheels for Linux, macOS, and
Windows are published on PyPI.

```sh
pip install pynw
```

On platforms without a prebuilt wheel, pip will build from the source
distribution. This requires a [Rust toolchain](https://rustup.rs/) (1.85+).

## Quick start

Align two DNA sequences using a simple match/mismatch scoring scheme:

```python
import numpy as np
from pynw import needleman_wunsch

seq1 = list("GATTACA")
seq2 = list("GCATGCA")

match, mismatch = 1.0, -1.0
similarity_matrix = np.where(
    np.array(seq1)[:, None] == np.array(seq2)[None, :], match, mismatch
)

result = needleman_wunsch(similarity_matrix, gap_penalty=-1.0)

aligned1 = "".join(seq1[i] if i >= 0 else "-" for i in result.row_idx)
aligned2 = "".join(seq2[i] if i >= 0 else "-" for i in result.col_idx)
print(f"Score: {result.score}\n{aligned1}\n{aligned2}")
# Score: 2.0
# G-ATTACA
# GCA-TGCA
```

`row_idx` and `col_idx` map each alignment position to an index in the
original sequence, with `-1` indicating a gap.

## Details

### Precomputed similarity matrix

`pynw` takes a precomputed `(n, m)` similarity matrix rather than a scoring
function. This means the entire alignment runs in compiled Rust code with no
Python callbacks, and you can build scores with vectorized NumPy operations
rather than element-wise Python loops.

The trade-off is that you must allocate the full matrix up front, which uses
$O(nm)$ memory even when the scoring rule could be expressed more compactly.

### Scoring

The total alignment score is the sum of similarity matrix entries for matched
positions and gap penalties for insertions/deletions. Gap penalties are
typically negative.

When multiple alignments achieve the same optimal score, `pynw` breaks ties
deterministically: `Diagonal > Up > Left`.

### Edit-distance parameterizations

Needleman-Wunsch can reproduce common metrics with the right scoring
parameters:

| Metric               | match | mismatch | gap      | NW score equals |
|----------------------|-------|----------|----------|-----------------|
| Levenshtein distance | 0     | -1       | -1       | `-distance`     |
| Indel distance       | 0     | -2       | -1       | `-distance`     |
| LCS length           | 1     | 0        | 0        | `lcs_length`    |
| Hamming distance     | 0     | -1       | `-(n+1)` | `-distance`     |

For Hamming, strings must have equal length.

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
pixi run test-hypothesis  # run property-based tests
pixi run lint             # run all pre-commit checks (ruff, cargo fmt, markdownlint, actionlint)
pixi run check            # run all pre-push checks (cargo clippy, mypy)
pixi run docs             # generate API docs in site/
pixi run docs-serve       # serve API docs in the browser
```

Linting and formatting are managed by [lefthook](https://github.com/evilmartians/lefthook)
and run automatically as git hooks.

## Related projects

- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz):
  Fast string distance functions (Levenshtein, Jaro-Winkler, etc.) with
  built-in scoring and edit operations. Likely the better choice for
  standard string distances.
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
