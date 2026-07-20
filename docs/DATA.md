# Data acquisition and preprocessing

## MovieLens 25M + Tag Genome

The artifact downloads both `ml-25m.zip` and the adjacent official `.md5` file from
the GroupLens host. It parses the published checksum, requires it to equal the pinned
paper value `6b51fb2759a8657d3bfcbfc42b592ada`, and verifies the archive before extraction. Required files are
`ratings.csv`, `genome-tags.csv`, and `genome-scores.csv`.

The archived experiment performs the following steps:

1. select the Tag Genome concepts `action`, `classic`, `visually appealing`,
   `based on a book`, `violence`, and `sci-fi`;
2. threshold relevance at 0.7 to construct six binary item attributes;
3. retain ratings of at least 4.0 for movies represented in the Tag Genome subset;
4. assign user and item indices by first appearance in the retained positive rows;
5. sort all retained positives globally by timestamp;
6. split the rows globally into 80% train, 10% validation, and 10% test;
7. choose the first 10,000 unique users encountered in the test portion and the first
   test row for each selected user;
8. exclude training-history items from each user's negative pool; and
9. sample 30 unique negatives with seed 42, placing the positive first.

This global split is reproduced because it generated the submitted MovieLens values.
It is not described as a per-user chronological split.

## Amazon Reviews 2023 — Beauty and Personal Care

The artifact downloads two category files directly from the official public host:

- review records for `Beauty_and_Personal_Care`; and
- metadata records for `meta_Beauty_and_Personal_Care`.

The source host does not publish an adjacent category-file checksum. The downloader
therefore records the observed SHA-256 and byte size in
`data/raw/bpc/download_manifest.json`. A camera-ready archive should retain this
manifest and, where data terms permit, a durable upstream snapshot identifier.

### Canonicalization

Review records retain `user_id`, `parent_asin`, `rating`, and `timestamp`. Rows with a
missing or nonnumeric required value are removed. `parent_asin` becomes the item ID.
For duplicate `(user, item, timestamp)` records, the last source row is retained.
Source-row indices are preserved during streaming so this policy remains explicit.

Metadata records retain item identity, store, main category, the category list,
price, and the text fields used to construct `item_text`. For duplicate item IDs, the
last source row is retained.

### Filtering and indexing

Reviews are inner-joined to items with metadata. The artifact then alternates:

1. remove users with fewer than three interactions;
2. remove items with fewer than three interactions;

until the row count converges or 25 iterations have completed. User and item IDs are
then mapped to zero-based integer indices in lexicographic order.

### Split

Within each user, interactions are ordered by `(timestamp, item_id)`. The final row is
test, the preceding row is validation, and all earlier rows are training. Users need
at least one training row, one validation row, and one test row; the converged 3-core
satisfies this condition.

The submitted dataset summary is:

| Quantity | Count |
|---|---:|
| users | 2,073,945 |
| items | 389,478 |
| training rows | 7,528,500 |
| validation rows | 2,073,945 |
| test rows | 2,073,945 |

The regeneration verifier requires exact equality for these counts before comparing
full-run metrics.

### Candidate sampling

Test history is training plus validation. The paper-compatible BPC sampler repeatedly
draws integer item indices from a uniform generator and rejects the positive or any
history item. It preserves a historical quirk: duplicate negative IDs are not
explicitly rejected. The artifact keeps this policy under the name
`paper_compatible_rejection` and also implements a portable unique-negative sampler.

Both modes store the positive in column zero before ranking.

## Data integrity outputs

Each downloader writes a manifest beside the raw data. The MovieLens manifest records
the published archive MD5 and SHA-256 hashes of the extracted files. Amazon does not
publish an adjacent hash for the two category files, so the run records the observed
SHA-256 and byte count and relies on exact processed-count checks to detect upstream
changes.

A completed experiment's `run_integrity_manifest.json` hashes the raw/processed
manifests and all retained row-level result objects. This separates two questions:

- whether a reviewer acquired the same upstream bytes; and
- whether the evaluator can reproduce the recorded metrics from the run's frozen
  rows and scores.
