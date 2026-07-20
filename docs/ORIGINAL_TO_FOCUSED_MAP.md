# Development-ZIP to focused-artifact map

The supplied ZIP was a broad deconfounding/recommender research repository. This
artifact keeps only the implementation needed to support the submitted tie-handling
paper. The table records where each retained idea came from and where it now lives.
It is a provenance map, not a claim that omitted historical row objects were recovered.

| Paper-relevant object in supplied ZIP | Focused implementation | Treatment |
|---|---|---|
| `scripts/tie_artifact_experiment.py` | `src/tie_eval/ranking.py`, `metrics.py`, `diagnostics.py` | split into tested evaluator modules; stable and archived hash semantics preserved |
| `src/amazon_po/eval/ranking.py` | `src/tie_eval/hashing.py`, `ranking.py` | 64-bit mixer and float32 conversion preserved; hardened uint64 policy added |
| `src/amazon_po/data/splits.py::sample_eval_candidates_flat` | `src/tie_eval/sampling.py::sample_bpc_paper_compatible` | seed, positive-first layout, history exclusion, draw/rejection behavior retained |
| `src/amazon_po/data/amazon23.py`, `filtering.py`, `interactions.py`, `splits.py` | `src/tie_eval/bpc_preprocess.py` | translated to streaming Parquet + DuckDB with explicit source-row and sort semantics |
| `src/amazon_po/labels/*` | `src/tie_eval/labels.py`, `text.py` | three BPC lexicons and metadata text construction retained |
| `src/amazon_po/propensity/model.py` | `src/tie_eval/bpc_models.py::fit_propensities` | three-fold masked-text TF-IDF logistic workflow retained |
| `src/amazon_po/features/groups.py` | `src/tie_eval/bpc_models.py::build_item_groups` | store/category/price-bucket groups retained |
| `src/amazon_po/control_variate/user_group.py`, `crossfit.py` | `src/tie_eval/bpc_models.py::fit_group_control_oof` | cross-fitted user/group control retained |
| `src/amazon_po/po/scoring.py` | `src/tie_eval/bpc_models.py::build_score_matrices` | paper-relevant raw, centered, item-residual, and group-control scores isolated |
| `scripts/run_movielens_tag_genome.py` | `src/tie_eval/movielens.py` | selected tags, mappings, split, sampling, scoring, and paired tie audit retained |
| BPC result/config files under `outputs/` and paper writer pack | `results/archived/bpc/` | sanitized aggregate evidence retained |
| MovieLens output directory under final discovery sprint | `results/archived/movielens/` | paired tie-mode CSV and summary evidence retained |

## Deliberately excluded

The focused package omits unrelated recommender baselines, LightGCN/SASRec training,
MIND and semi-synthetic studies, logs, plots unrelated to the submitted manuscript,
TensorBoard files, local machine paths, development planning notes, and source-control
metadata. Those objects do not help a reviewer test the tie-handling claim and would
make anonymous inspection harder.

## Changes that intentionally improve, rather than reproduce, the old packaging

- every new candidate row and candidate-score matrix is retained;
- the evaluator works in bounded-memory batches;
- random tie handling uses an unchanged primary score plus an independent secondary
  key instead of tiny score jitter;
- strict uint64-plus-item-ID tie breaking is tested under row permutation;
- effective configuration, runtime environment, raw/processed manifests, and result
  hashes are written automatically; and
- every quantitative paper claim has an explicit evidence pointer.
