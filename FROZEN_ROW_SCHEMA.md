# Frozen Row Exchange Schema

The cleanest evidence for a tie-handling audit is a compact payload created **after scoring and candidate sampling but before sorting**. It lets reviewers isolate the evaluator without redistributing raw user histories or trained models.

## Required NPZ arrays

| Name | Shape | Recommended dtype | Meaning |
|---|---:|---|---|
| `candidate_ids` | `(n_rows, n_candidates)` | `int64` | Item identity in each frozen candidate row |
| `primary_scores` | `(n_rows, n_candidates)` | original float dtype | Score supplied to the evaluator |
| `positive_item_ids` | `(n_rows,)` | `int64` | The single relevant item in each row |
| `user_ids` | `(n_rows,)` | `int64` | Stable anonymized user key used by keyed policies |

Optional arrays may include `row_ids`, `split_ids`, or a non-sensitive score-family label. Do not include names, emails, raw review text, or reversible external user identifiers.

## Required invariants

- Every row has the same declared candidate count.
- Candidate IDs are unique within a row.
- The positive item occurs exactly once in its row.
- Scores are finite except for explicitly documented padding, which must never enter a real candidate position.
- Item and user IDs are stable across row permutations.
- The payload is captured before any tie-breaking or sorting.

## Recommended sidecar metadata

```json
{
  "dataset": "neutral dataset label",
  "score_family": "raw_count",
  "candidate_sampling_seed": 42,
  "tie_seed": 20260316,
  "num_negatives": 30,
  "k": 10,
  "row_construction": "positive first, then sampled negatives",
  "negative_exclusions": ["positive item", "pre-test history"],
  "score_dtype": "float32",
  "candidate_ids_sha256": "...",
  "primary_scores_sha256": "..."
}
```

## Why hashes alone are not enough

Hashes prove that two parties possess the same arrays, but they do not let a reviewer verify tie blocks, ranks, or metrics. When licensing prevents row redistribution, provide a deterministic row-reconstruction command and hashes for each reconstructed array. A tiny public toy payload should still accompany the implementation so the evaluator can be exercised without restricted data.
