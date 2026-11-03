# How to install and reproduce

## Environment

Python 3.13 with the versions pinned in `code/requirements.txt` (scikit-learn 1.7.2, pandas 2.3.3,
numpy 2.2.6, scipy 1.18.1, matplotlib 3.11.2, seaborn 0.13.2, networkx 3.4.2, shap 0.54.1). The
revision analyses were run with exactly these versions on a 2-core Linux machine. `torch` is
needed only for the 1-D CNN branch of `code/comparators.py`.

```
pip install -r code/requirements.txt
```

## Data

Put the UCI Condition Monitoring of Hydraulic Systems files (`PS1.txt` ... `SE.txt`,
`profile.txt`) in `data/uci_hydraulic_raw/`, or let `code/paper3_diagnostics_experiment.py`
download them. Check them against the digests in `deposit/MANIFEST.csv` (rows with
`deposited=no`). The first call of `build_feature_table()` writes `data/processed/`.

## Original analyses (submission version)

Run order matters, because two scripts write `manuscript/generated_ceiling_bound_table.tex`
(see `code/EMITTER_ORDER.md`):

```
python3 code/paper3_diagnostics_experiment.py
python3 code/leakage_robustness.py
python3 code/comparators.py
python3 code/ceiling_analysis.py
python3 code/degradation_walk.py
python3 code/make_degradation_table.py
python3 code/run_clustered_bootstrap.py      # last
```

These scripts fit models and rewrite tables. In the revised manuscript several table captions
were edited after generation (terminology, the undefined cooler F1, the removed "Mean score"
column); the committed `.tex` tables are the reported ones. `code/verify_manuscript_numbers.py`
and `code/PROVENANCE_REPORT.txt` belong to the submission version of the manuscript.

## Revision analyses

```
python3 code/revision_analyses.py --stage fit       # fits every model; caches results/revision/cache/*.npz
python3 code/revision_analyses.py --stage analyse   # reads only the cache
python3 code/revision_figures.py                    # data figures with revised axis text
python3 code/revision_schematics.py                 # Figs. 1-3
```

`--stage fit` first reproduces the published blocked-split errors (probability-only 0.0224,
priority score 0.0513, bound 0.0229, seed 2026) before any new protocol is used. It then fits:
the ten-block split for ten seeds; the run-aligned purged split and the condition-profile
holdout for three seeds; HistGradientBoosting on the ten-block split; the chronological split;
the nested weight selection; and grouped SHAP per blocked fold.

| reviewer item | output |
|---|---|
| label mapping, stable flag | `results/revision/label_mapping.csv`, `stable_flag_subsets.csv`, Table `tab:labelmap` |
| purged blocks, profile holdout | `split_classification.csv`, `split_diagnostics.csv`, `split_B_purge.json`, Table `tab:splitcompare` |
| undefined F1 | `split_classification.csv` (n/a when no positives) |
| order preservation, ties, paired error minus bound | `ranking_by_split.csv`, `paired_and_bound_differences.csv`, `order_preservation_fold0_10seeds.csv` |
| channel-to-family map, evidence graph | `family_occlusion_blocked.csv`, `family_permutation_blocked.csv` |
| attribution stability | `explanation_stability.csv`, `occlusion_top_family_agreement.csv` |
| out-of-fold calibration | `calibration_oof.csv`, `figures/rev_reliability_blocked.pdf` |
| weight sensitivity on the blocked split | `priority_sensitivity_blocked.csv`, `weight_grid_blocked.csv`, `nested_weight_choices.csv` |
| label-free example | `revision_summary.json` (key `example`), `figures/rev_priority_example.pdf` |
| Table VIII vs. XIII | `r13_seed_reconciliation.csv`, Table `tab:reconcile` |
| HistGradientBoosting ranking and attributions | `hgb_ranking_blocked.csv`, `revision_summary.json` |
| single-fault subset | `single_fault_subset.csv` |

## Building the manuscript

```
cd manuscript
pdflatex main && pdflatex main
```

`main.bbl` is included, so BibTeX is not required.

## Deposit checks

`python3 deposit/build_manifest.py` writes `deposit/MANIFEST.csv`; `--check` exits 1 on drift.
`python3 deposit/check_deposit_ready.py` exits non-zero while the DOI, licence or version is unset.
