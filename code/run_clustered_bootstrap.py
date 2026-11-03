# ORDERING WARNING (2026-09-17): this script must run LAST of the emitters. It
# rewrites manuscript/generated_ceiling_bound_table.tex with the run-clustered
# intervals that superseded ceiling_analysis.py's cycle-resampled ones.
# See code/EMITTER_ORDER.md.
"""Run-clustered re-analysis of the separability-ceiling bootstrap (2026-09-12).

Why this script exists
----------------------
`code/ceiling_analysis.py:112-124` resamples **individual hydraulic cycles**
i.i.d. when it builds the 95% bootstrap intervals for the union bound and for
each policy's observed top-1 error. The portfolio resampling-unit audit of
2026-09-12 classified that as SOUND on the grounds that "the cycle is the
experimental unit of that dataset". Re-reading `data/uci_hydraulic_raw/
profile.txt` shows that it is not. The 2205 cycles are not 2205 independent
settings of the rig: identical `profile.txt` rows occur in **contiguous runs**,
because the rig was set to a fault configuration and then left there for a
block of consecutive 60-second load cycles. Grouping consecutive identical
rows gives

    2205 cycles  ->  194 condition-runs
    run-length histogram: 36 x 1, 154 x 10, 1 x 19, 2 x 200, 1 x 210

so the condition-run, not the cycle, is the unit that was independently set.
Cycles inside a run share one fault configuration and a slowly drifting thermal
state; the paper already concedes this when it calls the random split
"optimistic ... because the cycles are sequential" and builds a blocked-group
split to remove adjacent-cycle leakage. The same recurrence makes the
cycle-level bootstrap anticonservative.

What this script does
---------------------
1. **Control.** Re-derives every point estimate in
   `results/ceiling_bound_check.csv` (cross-component AUC, union bound, and
   each policy's observed top-1 error, on all three splits) from the same
   seeds, and asserts they match the committed file. Nothing about the
   estimator is changed at this step.
2. **Correction.** Changes *only* the resampling unit: the bootstrap draws 194
   condition-runs (or, on a split whose test set is a subset, the runs present
   in that test set) with replacement and concatenates their member cycles.
3. **The right instrument for the headline negative result.** The paper's
   principal claim -- the full priority score ranks *worse* than probability
   alone on the leakage-safe blocked-group split -- is a comparison of two
   policies scored on the same cycles. Two separate marginal intervals are the
   wrong test for it; this script adds a run-clustered **paired** bootstrap of
   the difference.

Outputs
-------
  results/ceiling_bound_check_runclustered.csv   (both units, side by side)
  manuscript/generated_ceiling_bound_table.tex   (regenerated, run-clustered CIs)

The i.i.d.-cycle intervals in `results/ceiling_bound_check.csv` are **not
deleted**: they remain committed, they are reproduced exactly by the control
above, and `results/SUPERSEDED_2026-09-12.md` records why they are no longer
the quoted ones.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ceiling_analysis as ca  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
COMP = ca.COMPONENTS

N_BOOT = 2000
RNG_SEED = 20260912

POLICY_SHORT = {
    "Probability only": "Probability only",
    "Severity-only policy": "Severity only",
    "Low-effort-only policy": "Low effort only",
    "Severity/effort heuristic": "Severity/effort",
    "Explanation confidence only": "Explanation conf.\\ only",
    "Local occlusion": "Local occlusion",
    "Priority score (prob + evidence + severity/effort)": "Priority score",
}
SPLIT_SHORT = {
    "random_stratified": "Random",
    "blocked_group_5fold": "Blocked 5-fold",
    "temporal_chronological_0.70": "Chronological",
}
PRIORITY = "Priority score (prob + evidence + severity/effort)"


def condition_runs() -> np.ndarray:
    """Cluster id per cycle: index of the contiguous run of identical profile.txt rows."""
    rows = [tuple(l.rstrip("\n").split("\t"))
            for l in open(ROOT / "data/uci_hydraulic_raw/profile.txt")]
    return np.concatenate(
        [[i] * len(list(g)) for i, (_, g) in enumerate(itertools.groupby(rows))]
    )


def build_splits():
    feats, targ, groups = ca.p3.build_feature_table()
    y = targ[[f"{c}_fault" for c in COMP]].to_numpy(int)
    n = len(feats)
    idx = np.arange(n)
    gidx = ca.p3.group_indices(feats, groups)
    runid = condition_runs()
    assert len(runid) == n, (len(runid), n)

    # random stratified, seed 17 -- matches ceiling_analysis.run_bound_check
    b = ca.p3.train_models(feats, targ, seed=17)
    occ_r = ca.occlusion_confidence(b["model"], b["x_test"], b["x_train"], gidx)

    # blocked group 5-fold, seed 2026 -- out-of-fold over every cycle
    blocks = (idx // max(1, n // 10)).clip(max=9)
    prob_blk = np.zeros((n, len(COMP)))
    occ_blk = np.zeros((n, len(COMP)))
    for tr, te in GroupKFold(n_splits=5).split(idx, groups=blocks):
        p, mdl, _sc, xtr, xte = ca.fit_predict(feats, y, tr, te, ca.SEED_BLOCKED)
        prob_blk[te] = p
        occ_blk[te] = ca.occlusion_confidence(mdl, xte, xtr, gidx)

    # chronological 0.70
    cut = int(0.70 * n)
    tr_c, te_c = idx[:cut], idx[cut:]
    p_c, mdl_c, _sc_c, xtr_c, xte_c = ca.fit_predict(feats, y, tr_c, te_c, ca.SEED_BLOCKED)
    occ_c = ca.occlusion_confidence(mdl_c, xte_c, xtr_c, gidx)

    return {
        "random_stratified":
            (y[b["test_idx"]], b["prob"], ca.policy_scores(b["prob"], occ_r), runid[b["test_idx"]]),
        "blocked_group_5fold":
            (y, prob_blk, ca.policy_scores(prob_blk, occ_blk), runid),
        "temporal_chronological_0.70":
            (y[te_c], p_c, ca.policy_scores(p_c, occ_c), runid[te_c]),
    }


def control(splits) -> float:
    """Reproduce the committed point estimates exactly. Returns max |delta|."""
    pub = pd.read_csv(RESULTS / "ceiling_bound_check.csv")
    puba = pd.read_csv(RESULTS / "cross_component_auc.csv")
    dmax = 0.0
    for sp, (yy, pp, ss, _cl) in splits.items():
        aucs = ca.cross_component_auc(yy, pp)
        bnd = ca.mean_union_bound(yy, aucs)
        for comp in COMP:
            r = puba[(puba.split == sp) & (puba.component == comp)].iloc[0]
            dmax = max(dmax, abs(aucs[comp][0] - r.cross_auc), abs(aucs[comp][1] - r.n_cycles))
        for name, s in ss.items():
            r = pub[(pub.split == sp) & (pub.policy == name)].iloc[0]
            dmax = max(dmax, abs(ca.top1_error(yy, s) - r.observed_top1_error),
                       abs(bnd - r.bound))
    return dmax


def bootstrap(yy, pp, ss, cl, clustered: bool, n_boot: int = N_BOOT):
    rng = np.random.default_rng(RNG_SEED)
    members = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    keys = list(ss.keys())
    errs = {k: np.full(n_boot, np.nan) for k in keys}
    bnds = np.full(n_boot, np.nan)
    diff = np.full(n_boot, np.nan)
    for i in range(n_boot):
        if clustered:
            pick = rng.integers(0, len(members), size=len(members))
            ix = np.concatenate([members[j] for j in pick])
        else:
            ix = rng.integers(0, len(yy), size=len(yy))
        yb = yy[ix]
        if (yb.sum(axis=1) > 0).sum() == 0:
            continue
        bnds[i] = ca.mean_union_bound(yb, ca.cross_component_auc(yb, pp[ix]))
        for k in keys:
            errs[k][i] = ca.top1_error(yb, ss[k][ix])
        if PRIORITY in errs:
            diff[i] = errs[PRIORITY][i] - errs["Probability only"][i]
    return errs, bnds, diff


def ci(a):
    a = a[~np.isnan(a)]
    if not len(a):
        return float("nan"), float("nan")
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def write_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r'''\caption{Probability-derived union bound $B_p$ on top-1 component-fault ranking error (Sec.~\ref{sec:ceiling}) against each policy's observed top-1 error, with 2{,}000-resample bootstrap 95\% intervals in brackets. The bootstrap resamples whole condition-runs (194 contiguous blocks of cycles sharing one \texttt{profile.txt} row; Sec.~\ref{sec:resampling}). The last column marks whether the point estimate of the error is at or below the point estimate of $B_p$; it is descriptive only. Whether a policy differs from the bound is tested with the paired intervals of Table~\ref{tab:paireddiff}, and $B_p$ is guaranteed to cover only order-preserving policies (Proposition~\ref{prop:bound}). Blocked 5-fold pools the out-of-fold predictions of all five folds (2205 cycles, seed 2026). ``Priority score'' is Eq.~\eqref{eq:priority}. Code: \texttt{code/run\_clustered\_bootstrap.py}; data: \texttt{results/ceiling\_bound\_check\_runclustered.csv}.}''',
        r"\label{tab:ceilingbound}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{p{1.35cm}p{1.85cm}p{1.75cm}p{1.75cm}p{0.5cm}}",
        r"\hline",
        r"Split & Policy & Obs.\ error & Bound $B_p$ & $E\le B_p$ \\",
        r" & & [95\% CI] & [95\% CI] & (point) \\",
        r"\hline",
    ]
    for sp in SPLIT_SHORT:
        sub = df[(df.split == sp) & (df.quantity.isin(POLICY_SHORT))]
        bnd = df[(df.split == sp) & (df.quantity == "bound")].iloc[0]
        for _, r in sub.iterrows():
            at = "Y" if (r.est <= bnd.est + 1e-9) else "N"
            lines.append(
                f"{SPLIT_SHORT[sp]} & {POLICY_SHORT[r.quantity]} & "
                f"{r.est:.3f} [{r.clu_lo:.3f}, {r.clu_hi:.3f}] & "
                f"{bnd.est:.3f} [{bnd.clu_lo:.3f}, {bnd.clu_hi:.3f}] & {at} \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_ceiling_bound_table.tex").write_text("\n".join(lines))


def chronological_curve(runid: np.ndarray) -> pd.DataFrame:
    """Re-run the nine-point chronological curve with the condition-run as the
    bootstrap unit. Control: the point estimates must match the committed
    results/chronological_curve.csv exactly."""
    from sklearn.metrics import f1_score

    feats, targ, groups = ca.p3.build_feature_table()
    y = targ[[f"{c}_fault" for c in COMP]].to_numpy(int)
    n = len(feats)
    idx = np.arange(n)
    gidx = ca.p3.group_indices(feats, groups)
    pub = pd.read_csv(RESULTS / "chronological_curve.csv")
    dmax = 0.0
    rows = []
    for frac in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        cut = int(frac * n)
        tr, te = idx[:cut], idx[cut:]
        y_test = y[te]
        prob, model, _s, x_train, x_test = ca.fit_predict(feats, y, tr, te, ca.SEED_BLOCKED)
        occ = ca.occlusion_confidence(model, x_test, x_train, gidx)
        pred = (prob >= 0.5).astype(int)
        macro_f1 = float(np.mean([f1_score(y_test[:, j], pred[:, j], zero_division=0)
                                  for j in range(len(COMP))]))
        ss = ca.policy_scores(prob, occ)
        aucs = ca.cross_component_auc(y_test, prob)
        bound = ca.mean_union_bound(y_test, aucs)
        mean_auc = float(np.nanmean([aucs[c][0] for c in COMP]))
        e_c, b_c, _ = bootstrap(y_test, prob, ss, runid[te], clustered=True,
                                n_boot=ca.N_BOOT_CURVE)
        b_lo, b_hi = ci(b_c)
        for name in ss:
            obs = ca.top1_error(y_test, ss[name])
            e_lo, e_hi = ci(e_c[name])
            r = pub[(pub.cut_fraction == frac) & (pub.policy == name)].iloc[0]
            dmax = max(dmax, abs((1.0 - obs) - r.top1_accuracy),
                       abs(macro_f1 - r.macro_f1), abs(bound - r.bound_top1_error),
                       abs(mean_auc - r.mean_cross_auc))
            rows.append({"cut_fraction": frac, "n_train": len(tr), "n_test": len(te),
                         "n_test_runs": int(len(np.unique(runid[te]))),
                         "macro_f1": macro_f1, "mean_cross_auc": mean_auc,
                         "policy": name, "top1_accuracy": 1.0 - obs,
                         "top1_accuracy_ci_lo": 1.0 - e_hi,
                         "top1_accuracy_ci_hi": 1.0 - e_lo,
                         "bound_top1_error": bound,
                         "bound_ci_lo": b_lo, "bound_ci_hi": b_hi})
    print(f"[control] chronological curve point estimates, max |delta| = {dmax:.3e}")
    assert dmax < 1e-9, "CONTROL FAILED -- chronological curve not reproduced"
    return pd.DataFrame(rows)


def main() -> None:
    runid = condition_runs()
    print(f"[run_clustered_bootstrap] {len(runid)} cycles in {runid.max() + 1} condition-runs")
    splits = build_splits()

    d = control(splits)
    print(f"[control] point estimates vs committed CSVs, max |delta| = {d:.3e}")
    assert d < 1e-9, "CONTROL FAILED -- published point estimates not reproduced"

    recs = []
    for sp, (yy, pp, ss, cl) in splits.items():
        e_i, b_i, d_i = bootstrap(yy, pp, ss, cl, clustered=False)
        e_c, b_c, d_c = bootstrap(yy, pp, ss, cl, clustered=True)
        n_runs = len(np.unique(cl))
        print(f"\n=== {sp}: {len(yy)} test cycles in {n_runs} condition-runs ===")
        bnd = ca.mean_union_bound(yy, ca.cross_component_auc(yy, pp))
        lo, hi = ci(b_i)
        clo, chi = ci(b_c)
        recs.append(dict(split=sp, quantity="bound", n_cycles=len(yy), n_runs=n_runs,
                         est=bnd, iid_lo=lo, iid_hi=hi, clu_lo=clo, clu_hi=chi))
        print(f"  bound {bnd:.4f}  iid[{lo:.4f},{hi:.4f}]  run[{clo:.4f},{chi:.4f}]")
        for k in ss:
            e = ca.top1_error(yy, ss[k])
            lo, hi = ci(e_i[k])
            clo, chi = ci(e_c[k])
            recs.append(dict(split=sp, quantity=k, n_cycles=len(yy), n_runs=n_runs,
                             est=e, iid_lo=lo, iid_hi=hi, clu_lo=clo, clu_hi=chi))
            w = (chi - clo) / (hi - lo) if hi > lo else float("nan")
            print(f"  {k[:46]:48s} {e:.4f} iid[{lo:.4f},{hi:.4f}] run[{clo:.4f},{chi:.4f}] x{w:.2f}")
        if PRIORITY in ss:
            de = ca.top1_error(yy, ss[PRIORITY]) - ca.top1_error(yy, ss["Probability only"])
            lo, hi = ci(d_i)
            clo, chi = ci(d_c)
            recs.append(dict(split=sp, quantity="paired_priority_minus_probonly",
                             n_cycles=len(yy), n_runs=n_runs, est=de,
                             iid_lo=lo, iid_hi=hi, clu_lo=clo, clu_hi=chi))
            print(f"  PAIRED priority - prob-only {de:+.4f} iid[{lo:+.4f},{hi:+.4f}] "
                  f"run[{clo:+.4f},{chi:+.4f}] excludes 0: {not (clo <= 0 <= chi)}")

    df = pd.DataFrame(recs)
    iid_w = df.iid_hi - df.iid_lo
    df["width_ratio_clu_over_iid"] = np.where(iid_w > 0, (df.clu_hi - df.clu_lo) / iid_w, np.nan)
    df.to_csv(RESULTS / "ceiling_bound_check_runclustered.csv", index=False)

    # Run-length structure of profile.txt, so the design description in
    # Sec. "Resampling Unit for Every Reported Interval" is traceable.
    lengths = np.array([int((runid == u).sum()) for u in np.unique(runid)])
    lc = pd.Series(lengths).value_counts().sort_index()
    pd.DataFrame({"run_length": lc.index, "n_runs": lc.values,
                  "n_cycles": lc.index * lc.values}).to_csv(
        RESULTS / "condition_run_lengths.csv", index=False)
    print("wrote results/condition_run_lengths.csv")
    write_table(df)
    print("\nwrote results/ceiling_bound_check_runclustered.csv")
    print("wrote manuscript/generated_ceiling_bound_table.tex")

    curve = chronological_curve(runid)
    curve.to_csv(RESULTS / "chronological_curve_runclustered.csv", index=False)
    print("wrote results/chronological_curve_runclustered.csv")

    # Redraw both ceiling figures with the run-clustered intervals, using the
    # committed plotting code unchanged (ceiling_analysis.make_figures reads the
    # CI columns by name, so only their contents change).
    bound_fig = df[df.quantity.isin(POLICY_SHORT)].rename(
        columns={"quantity": "policy", "est": "observed_top1_error",
                 "clu_lo": "observed_ci_lo", "clu_hi": "observed_ci_hi"})
    bmap = df[df.quantity == "bound"].set_index("split")
    bound_fig = bound_fig.assign(
        bound=bound_fig.split.map(bmap.est),
        bound_ci_lo=bound_fig.split.map(bmap.clu_lo),
        bound_ci_hi=bound_fig.split.map(bmap.clu_hi))
    ca.make_figures(bound_fig, curve)
    print("redrew figures/ceiling_bound_vs_observed.png and "
          "figures/ceiling_chronological_curve.png")


if __name__ == "__main__":
    main()
