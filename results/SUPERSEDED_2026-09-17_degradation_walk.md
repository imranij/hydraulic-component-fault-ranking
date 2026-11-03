# Superseded degradation-walk outputs — 2026-09-17

**Nothing has been deleted.** The 2026-09-10 outputs are retained verbatim in
`results/_preexisting_20260917/` and `figures/_preexisting_20260917/`.

## Why the walk was re-run

`AUTHOR_ACTION_REQUIRED.md` ("Added 2026-09-12 — portfolio missing-artifact sweep")
recorded that `code/degradation_walk.py` **was restored but not re-run**: its output
schema matched the two committed CSVs, but the numbers had not been recomputed, and the
note said to re-run it once before the deposit is minted. Timestamps confirmed it — the
script was dated 2026-09-12 17:02 while every artefact it produces was dated 2026-09-10
21:29. A deposit shipping a script that does not reproduce its own outputs is not a
reproducible deposit.

Re-run 2026-09-17 on the device VM: 4 axes x 5 levels x 10 fixed seeds = **200 forest
fits, 1400 rows**, matching the plan's budget exactly. The plan's own reproduction check
passed first: undegraded blocked-split macro-F1 (seed 2026) = **0.9645** against the
expected ~0.965.

The walk exceeds this shell's per-call time limit, so it was run one axis at a time
(seed-chunked where an axis still did not fit) through a harness that imports
`degradation_walk.py` **unmodified** and writes the outputs through the same code path
`main()` uses. The harness lives outside the repository and is not part of the deposit.
Row order in `degradation_walk.csv` is restored to what a single uninterrupted run
produces, and was verified identical.

## What changed

Row-for-row, all 1400 rows present in both, same order, same keys.

| quantity | rows differing > 1e-12 | max abs difference |
|---|---|---|
| `macro_f1` | 126 / 1400 | 6.70e-03 |
| `top1_accuracy` | 10 / 1400 | 8.99e-03 |
| `mean_cross_auc` | 7 / 1400 | 8.74e-04 |
| `bound_top1_error` | 7 / 1400 | 2.25e-03 |
| `n_train`, `n_features`, `n_flipped` | 0 | 0 |

At the summary level the drift is smaller still, and **at printed precision exactly one
reported value moves**:

| | 2026-09-10 | 2026-09-17 |
|---|---|---|
| macro-F1, priority policy, cycle window retained = 0.05 | 0.792664 (**0.793**) | 0.791462 (**0.791**) |

`generated_degradation_walk_table.tex` differs from the committed version in **one cell**
and is otherwise byte-identical. Every other value the manuscript quotes from this
experiment — 0.961, 0.978, 0.975, bound 0.041, 0.942, bound 0.027 and 0.080, 0.976,
0.910, 0.977, 0.932, std 0.083 — is unchanged to three decimal places, checked
individually.

## What the manuscript said, and what it says now

`manuscript/main.tex` Sec. Results (degradation sweep) read:

> "Retaining only 5\% of each cycle before feature extraction collapses macro-F1 from
> 0.961 to **0.793**"

It now reads **0.791**. That is the only prose change. The sentence's claim is unaffected:
the point is that macro-F1 collapses while top-1 accuracy barely moves (0.978 to 0.975,
both unchanged), and the collapse is if anything marginally larger.

## Why the numbers moved at all

Seeds, the blocked-group split, and every hyperparameter are fixed in the script, so the
walk is deterministic within one environment. The drift is concentrated in the
`cycle_window_retained` axis, which is the only axis that rebuilds the feature table from
truncated raw windows and therefore the only one whose *inputs* are recomputed rather
than subsampled — its moment features run through SciPy. The 2026-09-10 outputs were
produced under whatever library versions were installed then; that environment was not
recorded. Today's are recorded here:

```
scikit-learn 1.7.2   pandas 2.3.3   numpy 2.2.6   networkx 3.4.2
```

`networkx` was not installed on this VM and had to be added before the script would
import at all, which is further evidence that the committed outputs came from a different
environment.

**Recommendation for the deposit:** pin these versions in `code/requirements.txt`, which
currently uses `>=` bounds only. Not done here — changing an environment pin changes what
the deposit claims, and the pins should be the ones the author intends to publish.
