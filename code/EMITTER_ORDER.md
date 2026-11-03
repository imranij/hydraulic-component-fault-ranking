# Order the emitters must run in — read before regenerating anything

Written 2026-09-17 after a near-miss. Two facts about this paper's build are not
obvious from the scripts, and getting either wrong silently damages a result.

## 1. `run_clustered_bootstrap.py` MUST run last

**Two scripts write `manuscript/generated_ceiling_bound_table.tex`:**

| script | what it writes into that table |
|---|---|
| `ceiling_analysis.py` | intervals from resampling **individual cycles** — **superseded** |
| `run_clustered_bootstrap.py` | intervals from resampling **condition-runs** — **current** |

The 2026-09-12 correction (`results/SUPERSEDED_2026-09-12.md`) established that the
cycle is not the experimental unit: the 2205 cycles are 194 contiguous condition-runs, so
the cycle-resampled intervals were all too narrow. Point estimates were unchanged; every
interval was replaced.

**So running the emitters in alphabetical, arbitrary, or "all of them" order overwrites the
corrected table with the superseded one, and nothing fails.** The numbers still look
plausible, the verifier still passes against whichever CSV it reads, and the correction is
gone. `run_clustered_bootstrap.py` runs last, always.

Correct order:

```
python3 code/paper3_diagnostics_experiment.py     # features, probabilities, 4 tables
python3 code/leakage_robustness.py                # 2 tables
python3 code/comparators.py                       # 1 table
python3 code/ceiling_analysis.py                  # 5 tables, incl. the SUPERSEDED ceiling table
python3 code/degradation_walk.py                  # results + figure  (see note below)
python3 code/make_degradation_table.py            # 1 table
python3 code/run_clustered_bootstrap.py           # LAST -- rewrites the ceiling table correctly
python3 code/verify_manuscript_numbers.py         # must exit 0
```

`run_clustered_bootstrap.py` also redraws `figures/ceiling_bound_vs_observed.png` and
`figures/ceiling_chronological_curve.png` and rewrites
`results/chronological_curve_runclustered.csv`. Its own control re-derives the published
point estimates first and prints the residual — on 2026-09-17 that was
`max |delta| = 1.110e-16`, i.e. exact.

## 2. Only two of the six emitters are table rebuilds; the rest re-run the experiment

This matters because "just regenerate the tables and diff" is not a thing this paper can
do for eleven of its thirteen tables.

| emitter | fits models? | tables written |
|---|---|---|
| `paper3_diagnostics_experiment.py` | **yes** | calibration, component_metrics, priority_sensitivity, ranking_baselines |
| `leakage_robustness.py` | **yes** | confusion, leakage_robustness |
| `comparators.py` | **yes** | comparator |
| `ceiling_analysis.py` | **yes** | ceiling_bound (superseded), chronological_curve, constants, cross_auc, explanation_agreement |
| `make_degradation_table.py` | no — reads a CSV | degradation_walk |
| `run_clustered_bootstrap.py` | no — reads CSVs | ceiling_bound (current) |

Re-running the four model-fitting scripts is a **re-execution of the study**, not a table
rebuild, and it carries real drift risk. Precedent from the same day: re-running
`degradation_walk.py` under a recorded environment moved `macro_f1` on 126 of 1400 rows
(max 6.7e-3) and changed one printed manuscript value, 0.793 to 0.791
(`results/SUPERSEDED_2026-09-17_degradation_walk.md`). The committed outputs of 2026-09-10
had been produced under library versions nobody recorded.

**Verified 2026-09-17:** the two thin emitters were run in the order above and both tables
came back **byte-identical** to what shipped. The other eleven were deliberately not
regenerated, because doing so re-runs the study.

## 3. `run_all.bat` is stale — do not treat it as the recipe

The repository's own `run_all.bat` runs **two** of the six emitters
(`paper3_diagnostics_experiment.py`, `leakage_robustness.py`), then LaTeX. It never runs
`comparators.py`, `ceiling_analysis.py`, `make_degradation_table.py` or
`run_clustered_bootstrap.py`, and it never runs the verifier.

Anyone following it rebuilds four of the thirteen generated tables and leaves the other
nine at whatever is on disk — including the run-clustered ceiling table, which would simply
never be produced. **Not corrected here:** changing the build recipe changes what the paper
asserts is reproducible, and the ordering above must be confirmed by the author before it
becomes the committed recipe. Recorded as an author item.

## 4. Environment

The 2026-09-17 runs used scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6, networkx 3.4.2.
`code/requirements.txt` carries `>=` bounds only; pinning it is recommended and is an
author decision, since a pin is part of what the deposit claims.
