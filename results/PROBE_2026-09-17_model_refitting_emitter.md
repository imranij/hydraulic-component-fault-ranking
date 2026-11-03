# Probe — does a model-refitting emitter reproduce its committed outputs?

2026-09-17. One emitter, then stop, whatever the result.

**Nothing was deleted and nothing was left changed.** The pre-run state is in
`data/processed/_preleakage_20260917/` and `manuscript/_preleakage_20260917/`; the re-run's
own outputs are in `results/_probe_rerun_20260917/`; and the committed files were restored
exactly (all five `cmp`-identical to their backups). Verifier after the probe:
**368 / 368 / 0 / 0, exit 0**, unchanged.

## Why not `comparators.py`, which was the emitter chosen

`comparators.py` **cannot run in this paper's documented environment**, and that is a
finding in its own right:

- It imports `torch` at module level (lines 37, 42–43) for a small MLP comparator.
- **`torch` is not in `code/requirements.txt`.** Anyone installing from the paper's own pin
  cannot execute this emitter at all.
- On this machine `torch` is absent. `download.pytorch.org` (the CPU wheel index) is
  **403 at the egress proxy**; the default PyPI wheel is the CUDA build — a 554.6 MB wheel
  plus the whole `cuda-toolkit` / `nvidia-*` dependency set, to run a CPU MLP.

Installing a GPU stack on this VM to probe one table is disproportionate, so the probe moved
to the next cheapest **torch-free** model-refitting emitter.

## The emitter probed: `code/leakage_robustness.py`

Two generated tables, three derived CSVs, `RandomForestClassifier` retrained from scratch,
`SEED = 2026` fixed throughout. Ran in **36 s**.

Writes: `data/processed/paper3_split_robustness.csv`,
`paper3_confusion_matrices_temporal.csv`, `paper3_noise_stress.csv`,
`manuscript/generated_leakage_robustness_table.tex`,
`manuscript/generated_confusion_table.tex`.

## Result: every number reproduces exactly. The presentation does not, and re-running reverts it.

**All five outputs differed byte-for-byte. Not one numeric value moved.**

| file | numeric verdict |
|---|---|
| `paper3_split_robustness.csv` | all columns identical — 0.9975398253884368, 0.66739242327332, 0.9654607248619044 |
| `paper3_noise_stress.csv` | all columns identical |
| `paper3_confusion_matrices_temporal.csv` | all columns identical |

The CSV byte differences are **line endings only**: the committed files are CRLF (written on
Windows), the re-run writes LF on this Linux VM. Nothing else.

**The table differences are presentational, and they are a regression.** The committed
`generated_leakage_robustness_table.tex` is the *better* one and the script produces the
*worse* one:

| | committed (shipped) | what re-running produces |
|---|---|---|
| float | `table*` — two-column, correct for this layout | `table` — one column |
| rules | `booktabs`: `\toprule` / `\midrule` / `\bottomrule` | `\hline` throughout |
| grouping | `\multicolumn{2}{l}{\emph{Split protocol}}` and `\emph{Sensor-noise stress…}` section headers | removed |
| headline | `\textbf{0.965}` on the leakage-safe row | unbolded |
| labels | "Blocked group 5-fold split (leakage-safe headline)" | "Blocked group 5-fold split" |
| row order | random, blocked, chronological | random, chronological, blocked |

So the committed tables were improved after the script last generated them, and **the script
is older than the tables it is supposed to produce** — the mirror image of the
degradation-walk case, where the script was newer than its outputs.

## What this means for the open question

The worry was value drift: re-running `degradation_walk.py` under unrecorded library
versions moved `macro_f1` on 126 of 1400 rows and changed a printed manuscript value from
0.793 to 0.791. **That does not happen here.** A fixed-seed random forest on this dataset
reproduced to the last decimal place under scikit-learn 1.7.2.

What re-running *does* cost is silent: it reverts two tables to an earlier, plainer,
differently-ordered form, loses the booktabs rules, the grouped headers and the bolded
headline, and **nothing fails**. The verifier would still pass — it checks values, and the
values are right.

**One emitter probed, stopped as instructed.** The other three model-refitting emitters
(`ceiling_analysis.py`, `paper3_diagnostics_experiment.py`, and `comparators.py` if torch is
ever available) remain untested, and this result does not license assuming they behave the
same — `ceiling_analysis.py` in particular contains a bootstrap and writes the superseded
ceiling table.

## Why the committed files were restored rather than retired

The re-run's numbers are identical, so there is nothing to retire on numeric grounds, and
its tables are strictly worse. Leaving them in place would have degraded the shipped
manuscript for no gain. The re-run outputs are kept in `results/_probe_rerun_20260917/` so
the claim above is checkable. **Flagged for reversal** if the intent was to adopt whatever
the script produces.

## Two author items this raises

1. **Add `torch` to `code/requirements.txt`**, or drop the MLP comparator. The pin currently
   omits a dependency one emitter requires.
2. **Regenerate the two tables' formatting from the script, or update the script to emit the
   committed formatting.** Right now the script and the shipped tables disagree about
   presentation, and whichever runs last wins.
