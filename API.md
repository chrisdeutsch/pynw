# API Design Notes

Design considerations for `pynw`'s public API, informed by analysing the
target user base, performance characteristics, and the relative cost of each
output representation.

## Target users

The library's differentiating feature is the caller-supplied similarity
matrix. Existing string-distance libraries (e.g. `rapidfuzz`) hardcode
match/mismatch scoring; `pynw` enables arbitrary scoring including neural
embeddings, where "cat" and "feline" can align with low penalty. This
positions the library squarely at **ML/NLP practitioners** who need
order-preserving alignment on token sequences with semantic similarity.

Typical tasks:

- **ASR/MT evaluation** — aligning hypothesis tokens to a reference to compute
  WER or similar metrics, using exact-match or embedding-based similarity.
- **Structured output diffing** — comparing two lists of predicted spans,
  entities, or records where similarity is semantic rather than lexical.
- **Embedding-based alignment** — using cosine similarity of token embeddings
  as the similarity matrix.
- **Classic edit-distance metrics** — Levenshtein, LCS, indel distance, all
  recoverable via appropriate matrix values (see README).

Bioinformatics is a valid secondary use case but `BioPython` already serves it
well; the primary audience is data scientists and ML engineers who build their
own scoring.

### Further use cases

Beyond the core NLP/ML evaluation loop, the precomputed-matrix design enables
several other alignment tasks where order matters and similarity is non-binary:

- **Semantic code diffing** — aligning lines (or AST nodes) of two program
  versions using embedding or edit-distance similarity per line. Standard
  `diff` only sees exact line matches; a similarity matrix lets "moved and
  slightly modified" lines align instead of showing as delete + insert.
- **Time-series event alignment** — aligning ordered event sequences (user
  clickstreams, log traces, process steps) where events are typed or
  featurized and similarity is continuous. Useful for comparing expected vs
  observed workflows, or aligning traces across distributed systems.
- **OCR / HTR post-correction** — aligning OCR-output tokens to ground-truth
  tokens using character-level or visual-feature similarity, so substitution
  vs insertion/deletion errors can be separated for targeted correction.
- **Music and audio alignment** — aligning symbolic music sequences (MIDI
  notes, chord labels, melodic intervals) where similarity encodes harmonic
  or perceptual closeness (e.g. a perfect fifth is more similar than a
  tritone).
- **Document revision tracking** — aligning paragraphs or sections of two
  document versions using sentence-embedding similarity. Produces a
  structural diff that surfaces moved, rewritten, and inserted sections
  rather than character-level noise.
- **Multimodal sequence alignment** — aligning sequences from different
  modalities (video frames to transcript tokens, slide images to lecture
  sentences) using cross-modal embedding similarity.
- **Annotation reconciliation** — aligning ordered annotation spans from
  independent annotators (NER tags, dialogue acts, temporal labels) to
  compute inter-annotator agreement that accounts for boundary mismatches
  rather than requiring exact span overlap.

## Output representations

`needleman_wunsch` produces three distinct levels of output, with different
costs:

| Representation | Size    | Cost to produce          |
| -------------- | ------- | ------------------------ |
| Score          | 1 float | included in every run    |
| Ops            | O(n+m)  | requires O(nm) traceback |
| Indices        | O(n+m)  | O(n+m) pass over ops     |

The score is always free. Ops require the O(nm) traceback — there is no way to
recover them without it. Indices are a cheap O(n+m) post-processing step over
ops; their cost is noise relative to the alignment itself.

## Score-only mode

Skipping the traceback entirely lets the DP run with O(m) space instead of
O(nm), and eliminates the traceback pass. This is a genuine algorithmic
difference, not just a smaller output: the two-row rolling-array formulation
cannot reconstruct the alignment path.

This is worth exposing because ranking or filtering alignments by score — with
no interest in the path — is a common batch-evaluation pattern. A separate
`needleman_wunsch_score` function (or a `return_ops=False` keyword) would
serve this without complicating the primary API.

## Ops vs indices

Ops are sufficient whenever the question is about **aggregate structure**, not
about the actual paired elements:

```python
# These require only ops:
n_ins = np.sum(ops == OP_INSERT)
n_del = np.sum(ops == OP_DELETE)
n_aln = np.sum(ops == OP_ALIGN)
wer_denominator = n_aln + n_del  # reference length
```

Indices are required whenever the question involves the **elements
themselves**:

```python
# These require indices:
src_idx, tgt_idx = alignment_indices(ops)
substitutions = np.sum(seq1[src_idx] != seq2[tgt_idx])  # among OP_ALIGN
aligned1 = np.ma.array(seq1).take(src_idx).filled("-")   # reconstruction
```

The critical case is **substitution counting**. A position with `OP_ALIGN`
could be a match or a mismatch; to distinguish them you must compare the
actual elements, which requires indices. This rules out ops-only computation
for the most common NLP metric (WER includes substitutions separately from
insertions and deletions).

In practice: use ops alone only when counting gap operations or total alignment
length. Use indices as soon as element values matter.

## Against a three-level API (score / score+ops / score+indices)

A natural instinct is to offer three variants so callers pay only for what they
need. This is the right analysis for score vs ops — those differ algorithmically
— but wrong for ops vs indices.

**Ops → indices is not a meaningful boundary.** The conversion is O(n+m)
against an O(nm) alignment. Fusing it into the NW call saves nothing
measurable, hides a cheap composable step inside an expensive one, and
forces callers to decide upfront what they'll need. If they choose wrong they
must redo the full O(nm) alignment to recover.

**The existing two-step decomposition is correct:**

1. `needleman_wunsch` — expensive, produces score + ops.
2. `alignment_indices` — cheap, produces indices on demand.

The only gap is score-only, which requires a distinct code path and is worth
adding. Adding a combined score+indices function offers no performance benefit
and increases API surface area without justification.

## Per-position scores

Per-position scores along the traceback path — the contribution of each
alignment position to the total score — are fully derivable from existing
outputs:

```python
src_idx, tgt_idx = alignment_indices(ops)
pos_scores = np.where(
    ops == OP_ALIGN,
    similarity_matrix[src_idx.filled(0), tgt_idx.filled(0)],
    np.where(ops == OP_INSERT, insert_penalty, delete_penalty),
)
assert pos_scores.sum() == score
```

Because they are trivially composable from `ops` + `alignment_indices` +
the caller's own similarity matrix, they do not belong in the core API.

The one non-trivial case is when the caller wants per-position scores but
does **not** want to retain the similarity matrix after alignment — relevant
in batch workflows with large embedding matrices where peak memory is a
concern. Emitting scores during the traceback would allow the matrix to be
freed immediately. This is a legitimate but niche requirement; it does not
justify adding a new return value by default.

## Batch alignment

The largest missing capability for performance-critical users is batched
alignment: running NW over many sequence pairs without returning to the Python
interpreter between calls. A Python loop over `needleman_wunsch` incurs
per-call overhead and cannot exploit parallelism across pairs.

A `needleman_wunsch_many(matrices, ...)` function that accepts a list of
matrices (or a ragged structure) and dispatches to a Rust thread pool would be
the highest-leverage addition for this user base. The score-only variant is
especially attractive here, since batch ranking/filtering is a common pattern.

## Summary of recommendations

| Capability              | Status          | Recommendation                         |
| ----------------------- | --------------- | -------------------------------------- |
| Score + ops             | Implemented     | Primary API, keep as-is                |
| Ops → indices           | Implemented     | Keep as separate step                  |
| Score only              | Not exposed     | Add as `needleman_wunsch_score`        |
| Score + indices (fused) | Not implemented | Do not add; no performance benefit     |
| Per-position scores     | Derivable       | Do not add to core; document the idiom |
| Batch alignment         | Not implemented | High priority for performance users    |
