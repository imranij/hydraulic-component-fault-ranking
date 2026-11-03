# Superseded intervals — 2026-09-12

**Nothing in this directory has been deleted.** This note records which committed
values the manuscript still quotes and which it no longer does.

## What changed

`code/ceiling_analysis.py:112-124` builds every 95% bootstrap interval in this paper
by resampling **individual hydraulic cycles** i.i.d.:

```python
idx = BOOT_RNG.integers(0, n, size=n)
```

The portfolio resampling-unit audit of 2026-09-12
(`/home/claude/verify/RESAMPLING_AUDIT_2026-09-12.md`, verdict table row for paper 09)
classified this SOUND on the grounds that "the cycle is the experimental unit of that
dataset". **That is wrong**, and this note is the correction.

`data/uci_hydraulic_raw/profile.txt` holds one row per cycle. Identical rows occur in
**contiguous runs**, because the rig was set to a fault configuration and then left
there for a block of consecutive 60-second load cycles. Grouping consecutive identical
rows gives:

| | |
|---|---|
| cycles | 2205 |
| condition-runs | **194** |
| run-length histogram | 36 × 1, 154 × 10, 1 × 19, 2 × 200, 1 × 210 |

The condition-run, not the cycle, is the unit that was independently set. The paper
already relies on this fact elsewhere — it calls the random split "optimistic ...
because the cycles are sequential" and builds a blocked-group split precisely to stop
adjacent cycles leaking across the train/test boundary — but the bootstrap did not.

## Superseded (kept on disk, no longer quoted)

* `results/ceiling_bound_check.csv` — the `observed_ci_lo`, `observed_ci_hi`,
  `bound_ci_lo`, `bound_ci_hi` columns. Cycle-level, too narrow.
* `results/chronological_curve.csv` — the `top1_accuracy_ci_lo`,
  `top1_accuracy_ci_hi`, `bound_ci_lo`, `bound_ci_hi` columns. Same defect.

**The point estimates in both files are unaffected and are still the reported ones.**
`code/run_clustered_bootstrap.py` reproduces every one of them exactly before changing
anything (control: max |delta| = 1.1e-16 on the bound check and on the nine-point
chronological curve, against the committed CSVs).

## Reported instead

* `results/ceiling_bound_check_runclustered.csv` — both units side by side
  (`iid_lo`/`iid_hi` = the superseded cycle-level interval, `clu_lo`/`clu_hi` = the
  reported condition-run interval), plus the paired priority-vs-probability row.
* `results/chronological_curve_runclustered.csv` — same, for the nine cut points.
* `manuscript/generated_ceiling_bound_table.tex` — regenerated with the run-clustered
  intervals and a caption that names the unit.
* `figures/ceiling_bound_vs_observed.png`, `figures/ceiling_chronological_curve.png` —
  redrawn from the run-clustered intervals with the committed plotting code unchanged.

## Effect (2,000 resamples, seed 20260912, all three splits)

| Split | Quantity | Cycle-level 95% CI | Run-clustered 95% CI | Width × |
|---|---|---|---|---|
| Blocked group 5-fold | bound 0.023 | [0.017, 0.030] | [0.009, 0.040] | 2.45 |
| Blocked group 5-fold | probability-only error 0.022 | [0.016, 0.029] | [0.009, 0.040] | 2.45 |
| Blocked group 5-fold | priority-score error 0.051 | [0.042, 0.061] | [0.027, 0.084] | 3.02 |
| Blocked group 5-fold | local occlusion 0.028 | [0.022, 0.036] | [0.012, 0.051] | 2.70 |
| Blocked group 5-fold | explanation conf. only 0.087 | [0.076, 0.099] | [0.044, 0.138] | 4.02 |
| Blocked group 5-fold | severity-only 0.330 | [0.310, 0.349] | [0.194, 0.492] | 7.67 |
| Blocked group 5-fold | low-effort / severity-effort 0.265 | [0.247, 0.284] | [0.134, 0.433] | 8.04 |
| Random stratified | severity-only 0.303 | [0.269, 0.340] | [0.187, 0.448] | 3.66 |
| Random stratified | explanation conf. only 0.014 | [0.006, 0.023] | [0.002, 0.028] | 1.56 |
| Chronological 0.70 | bound / every prob. policy 0.175 | [0.145, 0.206] | [0.081, 0.301] | 3.64 |

Zero-width intervals on the random split (probability-only, local occlusion, priority
score, all at error 0.000) are unchanged, as is the chronological severity-only 1.000.

## Does any conclusion change? No.

1. **"The bound is essentially tight on the leakage-safe split."** Observed 0.022 and
   bound 0.023 now carry the same interval, [0.009, 0.040]. Survives.
2. **"The full priority score ranks worse than probability alone."** This is the paper's
   headline negative result, and two overlapping marginal intervals were never the right
   test for it — the two policies are scored on the same cycles. A **run-clustered
   paired** bootstrap of the difference gives **+0.029, 95% CI [0.012, 0.052]**, which
   excludes zero. Survives, and is now tested correctly. (Cycle-level paired, for
   reference: [0.022, 0.036].)
3. **"Static policies sit far above the bound."** On the blocked split they are
   0.265–0.330 with run-clustered intervals no lower than 0.134, against a bound whose
   interval tops out at 0.040. Survives with room to spare.
4. **"The chronological split's bound is 0.175 and every probability policy lands on
   it."** Point estimates unchanged; all intervals widen together. Survives.

The correction is real and the intervals in the manuscript changed, but no claim was
withdrawn. The audit's SOUND verdict was right about the conclusions and wrong about
the resampling unit.

Reproduce with: `cd code && python3 run_clustered_bootstrap.py`
