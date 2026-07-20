# Known limitations and evidence gaps

## Historical rows were not present in the supplied development ZIP

The archive contained aggregate BPC tie-audit numbers and exact MovieLens aggregate
CSV files, but not the historical candidate rows or candidate-score matrices. The new
repository can regenerate rows from public data; it cannot prove byte-for-byte
identity with objects that were never preserved.

This limitation is stated in `results/archived/PROVENANCE.json` and is not hidden by
the verification tooling.

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

## Paper wording for the BPC low-tie control needs precision

The submitted numeric row corresponds to a score that combines item-side residuals
with cross-fitted user/group outcome residualization. Item-side residualization alone
produced a different archived value. The code and documentation distinguish them;
the manuscript should do the same.

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

## Full public-data execution was not performed while constructing this ZIP

The package was validated with unit tests, toy end-to-end data, archived-evidence
checks, paper-output reconstruction, anonymity checks, and package hashes. External
dataset downloads were unavailable in the construction environment, so this ZIP does
not claim that a fresh full MovieLens or BPC run was observed here. Full verification
is intentionally executable by the reviewer and fails unless at least one full-scale
regenerated dataset is present.
