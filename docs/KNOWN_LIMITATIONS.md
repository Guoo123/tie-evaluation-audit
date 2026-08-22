# Known limitations and evidence gaps

## Original frozen rows were not present in the supplied development ZIP

The submission archive did not contain historical candidate rows or candidate-score
matrices for either dataset. For Amazon, preserved historical inputs later allowed a
deterministic reconstruction of 30,000 rows and six score matrices; all 84 checked
aggregate values match the original run exactly. These arrays are labeled as
reconstructed, not as files frozen during the original run.

For MovieLens, the canonical paired source file and row-level arrays were not
recovered. The camera-ready paper therefore reports the accepted aggregate values
unchanged and makes no claim of exact row-level MovieLens replay. See
`docs/CAMERA_READY_PROVENANCE.md`.

## Amazon category files lack an adjacent published checksum

The artifact records the hash observed during download. A later upstream replacement
could lead to different processed counts. Exact count checks will detect such a
change. A camera-ready release should retain the observed manifest and, if permitted,
a durable snapshot reference.

## BPC preprocessing is computationally demanding

The memory-safe JSONL-to-Parquet and DuckDB stages reduce peak memory, but the
cross-fitted user/group control can still require substantial RAM because it estimates
many sparse user-group coefficients. The pipeline is resumable, and reviewers can run
RawCount/centered checks without the low-tie control using
`--skip-low-tie-control`.

## Archived float32 keys are not collision-proof

Float32 conversion can map distinct 64-bit mixer outputs to equal keys. In that rare
case, stable sorting can again consult input order. The policy is retained only to
match the submitted table. `hardened_uint64` supplies the strict invariant reference.

## One keyed seed is one admissible realization

A deterministic key does not marginalize ties. The artifact therefore reports an
analytic expectation and repeated randomized estimates in addition to deterministic
keyed values.

## MovieLens uses the archived global split

The submitted MovieLens script used a global 80/10/10 timestamp split, not a per-user
last-item split. The artifact reproduces that choice for result compatibility and
documents it explicitly. It should not be generalized as the only appropriate split.

## The Amazon low-tie diagnostic must be identified precisely

The reported `0.1689 -> 0.1685` row combines item-side residuals with cross-fitted
user/group outcome residualization. Item-side residualization alone is a different
score. The camera-ready manuscript and artifact now distinguish them explicitly.

## The retained MovieLens summary contains a nearby but different hash-mode run

The dedicated paired tie-audit CSV is the canonical source for the paper table.
`experiment_summary.json` contains slightly different hash-mode values, indicating a
separate candidate-sampling/evaluation execution. For RawCount, the summary reports
0.2336924146 while the paired tie audit reports 0.2332344277. The artifact exposes
both files and documents the choice in `docs/ARCHIVED_EVIDENCE_AUDIT.md` rather than
silently merging them.

## BPC `RawCount` is rating weighted in the operational code

The BPC implementation multiplies each historical item attribute vector by
`clip((rating - 1) / 4, 0, 1)` before aggregation. The historical identifier
`raw_count` is retained, but an exact manuscript description should call it a
rating-weighted historical attribute total.

## Public-data regeneration remains prospective in this source repository

The source repository is validated with unit tests, toy end-to-end data,
archived-evidence checks, camera-ready table reconstruction, package-hygiene checks,
and package hashes. The separate Amazon camera-ready release payload also passed an
evaluator-only replay on the reconstructed 30,000-row arrays. A fresh public-data
MovieLens or Amazon run remains prospective and is not conflated with the accepted
aggregate evidence. Full verification fails unless at least one full-scale regenerated
dataset is present.
