# Explainable component-fault diagnostics and maintenance-priority screening (UCI hydraulic rig)

Code, result files and manuscript sources for the paper *Explainable Component-Fault
Diagnostics and Maintenance-Priority Screening for Hydraulic-System Condition Monitoring*
(Imran, KFUPM), submitted to *IEEE Transactions on Instrumentation and Measurement* (revised
version, November 2026).

Archived release: https://doi.org/DEPOSITDOI

## What is here

| folder | contents |
|---|---|
| `code/` | the original pipeline (`paper3_diagnostics_experiment.py`, `leakage_robustness.py`, `ceiling_analysis.py`, `run_clustered_bootstrap.py`, `degradation_walk.py`, `make_degradation_table.py`, `comparators.py`) and the revision analyses (`revision_analyses.py`, `revision_analyse.py`, `revision_figures.py`, `revision_schematics.py`) |
| `results/` | every result CSV behind a reported number; `results/revision/` holds the analyses added in revision and `results/revision/cache/` the cached out-of-fold predictions they were computed from |
| `data/processed/` | the 187-feature cycle table, the targets and the channel-to-feature map, derived from the UCI raw streams |
| `figures/` | every figure in the manuscript |
| `manuscript/` | LaTeX source of the revised manuscript, its formatted bibliography and every generated table |
| `deposit/` | `MANIFEST.csv` (path, size, SHA-256 of every file, including digests of the raw UCI files that are not redistributed), the manifest builder and the release check |

## Data

The UCI Condition Monitoring of Hydraulic Systems data set (Helwig, Pignanelli and Schütze,
I2MTC 2015, doi:10.1109/I2MTC.2015.7151267) is not redistributed. Place its 17 sensor files and
`profile.txt` in `data/uci_hydraulic_raw/`, or let `code/paper3_diagnostics_experiment.py`
download it from the UCI repository. `deposit/MANIFEST.csv` records the SHA-256 digest of each
raw file, so you can confirm that your copy is the one used here.

## Reproduce

See `HOW_TO_INSTALL_AND_REPRODUCE.md`. In short:

```
pip install -r code/requirements.txt
python3 code/revision_analyses.py --stage fit      # ~25 min on 2 cores; caches OOF predictions
python3 code/revision_analyses.py --stage analyse  # writes results/revision/*, tables, figures
python3 code/revision_figures.py
python3 code/revision_schematics.py
```

## Licence

Code: MIT (see `LICENSE`). Result files, figures and text: CC BY 4.0.

## Citation

See `CITATION.cff`.
