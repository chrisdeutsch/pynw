# Usage Examples

## Global Alignment of String Sequences

Align full movie titles against shortened versions using fuzzy string
similarity from [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz). Titles
without a close match are gapped, so you can see which entries were dropped.

```python
import numpy as np
from pynw import needleman_wunsch, iter_alignment
from rapidfuzz.process import cdist
from rapidfuzz.fuzz import ratio

source_seq = np.array([
    "Episode I - The Phantom Menace",
    "Episode II - Attack of the Clones",
    "Episode III - Revenge of the Sith",
    "Episode IV - A New Hope",
    "Episode V - The Empire Strikes Back",
    "Episode VI - Return of the Jedi",
])

target_seq = np.array([
    "Attack of the Clones",
    "A New Hope",
    "The Empire Strikes Back",
    "Return of the Jedi",
])

score = cdist(source_seq, target_seq, scorer=ratio) / 100
_, ops = needleman_wunsch(score)

for _, source_item, target_item in iter_alignment(ops, source_seq, target_seq):
    s = source_item if source_item is not None else "GAP"
    t = target_item if target_item is not None else "GAP"
    print(f"{s:40} -> {t}")
```

Expected output:

```text
Episode I - The Phantom Menace           -> GAP
Episode II - Attack of the Clones        -> Attack of the Clones
Episode III - Revenge of the Sith        -> GAP
Episode IV - A New Hope                  -> A New Hope
Episode V - The Empire Strikes Back      -> The Empire Strikes Back
Episode VI - Return of the Jedi          -> Return of the Jedi
```

## Semantic Document Diff using Embeddings

A powerful use-case for sequence alignment is comparing drafts of documents where sentences have been heavily paraphrased, reordered, or removed. A standard textual diff (like `git diff`) struggles with this, but we can compute a "Semantic Diff" using dense embeddings.

### The Key Intuition

Standard sequence alignment rewards exact matches and penalizes gaps (insertions/deletions). To adapt this for semantic comparisons, we use two tricks:

1. **Shift the Scoring Matrix:** Cosine similarity yields values between `-1` and `1`. By defining a matching threshold (e.g., `0.65`) and subtracting it from the similarities, good matches become positive scores, and poor matches become negative scores.
2. **Zero Gap Penalties:** By setting gap penalties to `0.0`, there is no inherent punishment for skipping a sentence. The algorithm will strictly prefer to align sentences _only_ if their similarity is above the threshold (since that's the only way to increase the total score), elegantly handling completely new or deleted paragraphs.

Here we use the lightweight, CPU-friendly [`fastembed`](https://qdrant.github.io/fastembed/) library to compute cosine similarities to construct our score matrix.

```bash
# Requires additional dependencies
uv pip install fastembed numpy
```

```python
import numpy as np
from pynw import needleman_wunsch, iter_alignment
from fastembed import TextEmbedding

# Initialize the embedding model
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# Our original draft vs a heavily paraphrased and edited final version
draft = [
    "The company was founded in 2012 by two friends.",
    "They started working in a small garage.",
    "Initially, they struggled to find investors.",
    "However, their first product was a big hit.",
    "Today, they employ over 500 people."
]

final_rev = [
    "Two university buddies established the corp in 2012.",
    "Their origins trace back to a tiny residential garage.",
    "After several months, they launched a successful app.",
    "They recently expanded to the European market.",
    "Currently, the workforce consists of 500+ employees."
]

# 1. Generate embeddings
# `model.embed` returns a generator, so we convert it to a list and then a NumPy array
emb_1 = np.array(list(model.embed(draft)))
emb_2 = np.array(list(model.embed(final_rev)))

# 2. Compute cosine similarity matrix
# Normalize embeddings to compute cosine similarity efficiently via dot product
emb_1_norm = emb_1 / np.linalg.norm(emb_1, axis=1, keepdims=True)
emb_2_norm = emb_2 / np.linalg.norm(emb_2, axis=1, keepdims=True)
cosine_sim = np.dot(emb_1_norm, emb_2_norm.T)

# 3. Create a scoring matrix for the "Semantic Diff"
# We only want to align sentences if they are semantically similar.
# By subtracting a threshold, matches get positive scores and mismatches get negative ones.
threshold = 0.65
similarity_matrix = cosine_sim - threshold

# 4. Run alignment
# We set gap penalties to 0.0. The algorithm will naturally align sentences
# that score > 0 (similarity > threshold), and insert gaps otherwise.
_, ops = needleman_wunsch(similarity_matrix, gap_penalty=0.0)

# 5. Print the Semantic Diff
print("--- Semantic Document Diff ---\n")
for _, i1, i2 in iter_alignment(ops, range(len(draft)), range(len(final_rev))):
    if i1 is not None and i2 is not None:
        print(f"[ MATCH ] (sim: {cosine_sim[i1, i2]:.2f})")
        print(f"  - {draft[i1]}")
        print(f"  + {final_rev[i2]}\n")
    elif i1 is not None:
        print(f"[ DELETED ]")
        print(f"  - {draft[i1]}\n")
    elif i2 is not None:
        print(f"[ INSERTED ]")
        print(f"  + {final_rev[i2]}\n")
```

Expected output:

```text
--- Semantic Document Diff ---

[ MATCH ]
  - The company was founded in 2012 by two friends.
  + Two university buddies established the corp in 2012.

[ MATCH ]
  - They started working in a small garage.
  + Their origins trace back to a tiny residential garage.

[ DELETED ]
  - Initially, they struggled to find investors.

[ MATCH ]
  - However, their first product was a big hit.
  + After several months, they launched a successful app.

[ INSERTED ]
  + They recently expanded to the European market.

[ MATCH ]
  - Today, they employ over 500 people.
  + Currently, the workforce consists of 500+ employees.
```
