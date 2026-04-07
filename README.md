# pynw

[![Check](https://github.com/chrisdeutsch/pynw/actions/workflows/check.yml/badge.svg)](https://github.com/chrisdeutsch/pynw/actions/workflows/check.yml)
[![Build](https://github.com/chrisdeutsch/pynw/actions/workflows/build.yml/badge.svg)](https://github.com/chrisdeutsch/pynw/actions/workflows/build.yml)
[![PyPI](https://img.shields.io/pypi/v/pynw)](https://pypi.org/project/pynw/)
[![Python](https://img.shields.io/pypi/pyversions/pynw)](https://pypi.org/project/pynw/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`pynw` aligns two ordered sequences element-by-element using any pairwise
similarity matrix you supply. Think of it as a generalised diff for arbitrary
ordered collections: words, embeddings, tokens, or any objects where you define
how well each pair of elements matches.

Unlike string-distance or bioinformatics libraries that assume a fixed alphabet
and built-in scoring scheme, `pynw` delegates scoring entirely to the user, so
any pairwise metric works: cosine similarity of embeddings, model outputs,
learned distances, or domain-specific rules. The
[Needleman-Wunsch](https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm)
global sequence alignment algorithm provides the underpinning for `pynw`. It is
implemented in Rust with Python bindings built using [PyO3](https://pyo3.rs).

```python
import numpy as np
from pynw import needleman_wunsch

# Pairwise similarity matrix between your two sequences (source × target)
S = np.array([
    [1.0, 0.1],
    [0.1, 0.1],
    [0.1, 1.0],
])

score, editops = needleman_wunsch(S, gap_penalty=-0.5)
# score: 1.5
# editops: [EditOp.Align, EditOp.Delete, EditOp.Align]
```

## Features

- **Fast:** Alignment runs in $\mathcal{O}(nm)$ time; a $1000 \times 1000$ matrix takes <10 ms on modern CPUs.
- **NumPy-first:** Pass NumPy arrays directly, no conversion needed.
- **Domain-agnostic:** Operates on a user-supplied similarity matrix.
- **Asymmetric gaps:** Penalize inserts and deletes independently.

## When to Use pynw

Reach for `pynw` when you need global alignment of two ordered sequences and
the notion of similarity is specific to your domain: aligning sentences from
two translations using embedding similarities, matching token streams emitted
by different tokenizers, comparing event logs with custom match rules, or any
case where precomputing scores in NumPy is natural.

If your problem fits a standard string metric (Levenshtein, Jaro-Winkler, and
friends), [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) is faster and more
featureful. If you are aligning biological sequences, use a bioinformatics
library such as
[BioPython](https://biopython.org/docs/dev/Tutorial/chapter_align.html). If
ordering does not matter and you want optimal one-to-one assignment, see
[`scipy.optimize.linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
See [Related Projects](#related-projects) for more.

## Installation

Prebuilt wheels for Linux, macOS, and Windows are published on PyPI:

```sh
pip install pynw
```

`pynw` requires Python 3.10+ and NumPy 1.22+. On platforms without a prebuilt
wheel, pip will build from the source distribution; this requires a
[Rust toolchain](https://rustup.rs/) (1.85+).

## Quick Start

Using `pynw` involves three steps:

1. **Build a similarity matrix.** Compute pairwise scores between every element
   of your two sequences using any scoring function.
2. **Run the alignment.** Pass the matrix and a gap penalty to
   `needleman_wunsch`. The gap penalty controls when leaving an element unmatched
   is preferable to a low-scoring match.
3. **Interpret the results.** The returned edit operations tell you which
   elements were aligned, inserted, or deleted.

The example below aligns two word sequences. The similarity matrix is built from
cosine similarities of [GloVe](https://nlp.stanford.edu/projects/glove/) word
embeddings, letting semantically related words align even without an exact
match:

```python
import numpy as np
from pynw import EditOp, needleman_wunsch, alignment_indices

source = np.array(
    ["clever", "sneaky", "fox", "leaped"]
)
target = np.array(
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

# Each gap deducts 0.5 from the total score; increase the penalty to force
# more alignments, decrease it to allow more gaps
score, editops = needleman_wunsch(similarity_matrix, gap_penalty=-0.5)
source_indices, target_indices = alignment_indices(editops)

# Reconstruct aligned sequences; masked positions are gaps
aligned_source = np.ma.array(source).take(source_indices).filled("-")
aligned_target = np.ma.array(target).take(target_indices).filled("-")

LABELS = {EditOp.Align: "match", EditOp.Delete: "delete", EditOp.Insert: "insert"}

print(f"Score: {round(score, 2)}")
for op, s, t in zip(editops, aligned_source, aligned_target):
    print(f"  {s:10s}  {t:10s}  ({LABELS[op]})")
```

Output:

```text
Score: 1.42
  clever      sly         (match)
  sneaky      -           (delete)
  fox         fox         (match)
  leaped      jumped      (match)
  -           across      (insert)
```

## User Guide

`pynw` exposes two alignment functions and a helper for interpreting results.
Use `needleman_wunsch` when you need the actual alignment, and
`needleman_wunsch_score` when you only need the score.

`pynw` takes a precomputed $n \times m$ similarity matrix rather than a scoring
callback. This allows the alignment to run entirely in native code and lets you
build scores using vectorized NumPy operations, at the cost of $\mathcal{O}(nm)$
memory for the matrix.

### Score and alignment: `needleman_wunsch`

`needleman_wunsch` returns the optimal score along with an array of edit
operations (`editops`). Each element in the editops array is one of three
`EditOp` values:

- `EditOp.Align`: a source element is matched with a target element.
- `EditOp.Delete`: a source element is consumed with no matching target element
  (gap in target).
- `EditOp.Insert`: a target element is consumed with no matching source element
  (gap in source).

The editops array alone is enough for aggregate statistics:

```python
from pynw import EditOp, needleman_wunsch

score, editops = needleman_wunsch(similarity_matrix)

n_aligned = np.sum(editops == EditOp.Align)
n_inserted = np.sum(editops == EditOp.Insert)
n_deleted = np.sum(editops == EditOp.Delete)
```

When multiple alignments achieve the same optimal score, `pynw` breaks ties
deterministically: `Align > Delete > Insert`.

By default, `gap_penalty` applies equally to insertions and deletions. Pass
`insert_penalty` and/or `delete_penalty` to penalize them independently, which
is useful when the cost of missing a source element differs from the cost of
introducing a spurious target element:

```python
score, editops = needleman_wunsch(
    similarity_matrix,
    insert_penalty=-0.3,
    delete_penalty=-0.7,
)
```

### Reconstructing the alignment: `alignment_indices`

`alignment_indices` converts an editops array into two masked index arrays
(one per sequence) with one entry per alignment position. Gap positions are
masked, so `take(...).filled("-")` reconstructs aligned sequences with gap
markers:

```python
from pynw import alignment_indices

source_indices, target_indices = alignment_indices(editops)

source = np.ma.array(["the", "quick", "fox"])
target = np.ma.array(["the", "slow", "red", "fox"])

aligned_source = source.take(source_indices).filled("-")
aligned_target = target.take(target_indices).filled("-")
# aligned_source: ['the', 'quick', '-',   'fox']
# aligned_target: ['the', 'slow',  'red', 'fox']
```

Iterating over a masked array yields
[`np.ma.masked`](https://numpy.org/doc/stable/reference/maskedarray.baseclass.html#numpy.ma.masked)
at gap positions, so you can branch on the editop without explicit mask checks:

```python
for op, src, tgt in zip(editops, source_indices, target_indices):
    if op == EditOp.Align:
        print(f"  {source[src]}")
    elif op == EditOp.Delete:
        print(f"- {source[src]}")
    elif op == EditOp.Insert:
        print(f"+ {target[tgt]}")
```

### Score only: `needleman_wunsch_score`

Use `needleman_wunsch_score` when you only need the alignment score, for
example when ranking or filtering many sequence pairs. It skips the traceback
entirely, using $\mathcal{O}(m)$ memory instead of $\mathcal{O}(nm)$:

```python
from pynw import needleman_wunsch_score

score = needleman_wunsch_score(similarity_matrix)
```

## Reproducing Classical Edit Distances

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

## Related Projects

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

## Contributing & Support

Open a [GitHub issue](https://github.com/chrisdeutsch/pynw/issues) for bug
reports, questions, or feature requests. See
[CONTRIBUTING](CONTRIBUTING.md) for guidelines on submitting changes.

## License

[MIT](LICENSE)
