# ORDERING WARNING (2026-09-17): this script writes
# manuscript/generated_ceiling_bound_table.tex with CYCLE-resampled intervals,
# which the 2026-09-12 correction SUPERSEDED. run_clustered_bootstrap.py must run
# AFTER this script, or the corrected run-clustered table is silently overwritten
# and nothing fails. See code/EMITTER_ORDER.md.
"""
Separability-ceiling derivation, chronological generalization curve, and
explanation-agreement check for Paper 09 (hydraulic diagnostics), Phase 1.

Implements plan Section 5, E2 (the saturation bound and its check) and the
chronological expansion requested for Phase 1, plus E5's Spearman-agreement
check between occlusion, permutation, and grouped SHAP importance, restricted
to what is runnable locally on the existing UCI hydraulic data.

Cross-component AUC (as specified in top10percentplan.md Sec. 4a): for a
component c and the fault-containing cycles where c is a true negative,
AUC_c = P(best true-positive-component score > score_c), estimated empirically
(ties broken at 0.5) over all such cycles. Top-1 error is then bounded above,
cycle by cycle, by the sum over that cycle's negative components of
(1 - AUC_c); the reported bound is the mean of that per-cycle quantity over
all fault-containing cycles (a union bound, so it can be loose).

Outputs (all under results/ and manuscript/):
  results/ceiling_bound_check.csv        -- E2: bound vs. observed error, 3 splits x 8 policies
  results/cross_component_auc.csv        -- per split, per component AUC with bootstrap CI
  results/chronological_curve.csv        -- E2/curve: cut point x policy x metric, with CI
  results/explanation_agreement.csv      -- E5: Spearman agreement, occlusion/permutation/SHAP
  figures/ceiling_chronological_curve.png
  figures/ceiling_bound_vs_observed.png
  manuscript/generated_ceiling_bound_table.tex
  manuscript/generated_chronological_curve_table.tex
  manuscript/generated_explanation_agreement_table.tex
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

import paper3_diagnostics_experiment as p3

ROOT = p3.ROOT
PROC = p3.PROC_DIR
FIG_DIR = p3.FIG_DIR
MANUSCRIPT = ROOT / "manuscript"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

COMPONENTS = p3.COMPONENTS
SEVERITY = p3.SEVERITY
EFFORT = p3.EFFORT
SEED_BLOCKED = 2026  # matches code/leakage_robustness.py
N_BOOT = 2000       # headline bound-check CIs (E2), per integrity guardrails
N_BOOT_CURVE = 500  # chronological curve has 9 cut points x 8 policies; kept smaller for runtime
BOOT_RNG = np.random.default_rng(20260909)


# ---------------------------------------------------------------------------
# Core bound machinery (vectorized)
# ---------------------------------------------------------------------------

def cross_component_auc(y: np.ndarray, score: np.ndarray) -> Dict[str, Tuple[float, int]]:
    """AUC_c per component: P(best true-positive score > score_c) on cycles
    where c is a true negative and at least one other component is positive.
    Ties count as 0.5. Returns {component: (auc, n_cycles_used)}."""
    masked = np.where(y == 1, score, -np.inf)
    pos_best = masked.max(axis=1)
    any_pos = y.sum(axis=1) > 0
    out: Dict[str, Tuple[float, int]] = {}
    for c, name in enumerate(COMPONENTS):
        mask = (y[:, c] == 0) & any_pos
        if not np.any(mask):
            out[name] = (float("nan"), 0)
            continue
        pb = pos_best[mask]
        neg = score[mask, c]
        wins = (pb > neg).astype(float) + 0.5 * (pb == neg).astype(float)
        out[name] = (float(np.mean(wins)), int(np.sum(mask)))
    return out


def top1_error(y: np.ndarray, score: np.ndarray) -> float:
    any_pos = y.sum(axis=1) > 0
    if not np.any(any_pos):
        return float("nan")
    yf = y[any_pos]
    sf = score[any_pos]
    top = np.argmax(sf, axis=1)
    correct = yf[np.arange(len(yf)), top] == 1
    return 1.0 - float(np.mean(correct))


def mean_union_bound(y: np.ndarray, aucs: Dict[str, Tuple[float, int]]) -> float:
    """Mean, over fault-containing cycles, of sum_{c negative in that cycle} (1 - AUC_c)."""
    any_pos = y.sum(axis=1) > 0
    if not np.any(any_pos):
        return float("nan")
    yf = y[any_pos]
    one_minus_auc = np.array(
        [1.0 - aucs[name][0] if not np.isnan(aucs[name][0]) else 0.0 for name in COMPONENTS]
    )
    bound_per_cycle = (1 - yf) @ one_minus_auc
    return float(np.mean(bound_per_cycle))


def bootstrap_bound_and_error(y: np.ndarray, score: np.ndarray, n_boot: int):
    n = y.shape[0]
    errs = np.full(n_boot, np.nan)
    bounds = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = BOOT_RNG.integers(0, n, size=n)
        yb, sb = y[idx], score[idx]
        if yb.sum() == 0 or (yb.sum(axis=1) > 0).sum() == 0:
            continue
        aucs_b = cross_component_auc(yb, sb)
        errs[b] = top1_error(yb, sb)
        bounds[b] = mean_union_bound(yb, aucs_b)
    return errs, bounds


def ci95(arr: np.ndarray) -> Tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------

def fit_predict(features: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                 seed: int) -> Tuple[np.ndarray, MultiOutputClassifier, StandardScaler, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.iloc[train_idx].to_numpy(float))
    x_test = scaler.transform(features.iloc[test_idx].to_numpy(float))
    base = RandomForestClassifier(n_estimators=260, min_samples_leaf=2, random_state=seed,
                                   class_weight="balanced_subsample", n_jobs=-1)
    model = MultiOutputClassifier(base)
    model.fit(x_train, y[train_idx])
    prob = np.vstack([p3.positive_class_proba(e, x_test) for e in model.estimators_]).T
    return prob, model, scaler, x_train, x_test


def policy_scores(prob: np.ndarray, occlusion_conf: np.ndarray | None) -> Dict[str, np.ndarray]:
    n = prob.shape[0]
    out: Dict[str, np.ndarray] = {
        "Probability only": prob,
        "Severity-only policy": np.tile(SEVERITY, (n, 1)),
        "Low-effort-only policy": np.tile(1.0 / EFFORT, (n, 1)),
        "Severity/effort heuristic": np.tile(SEVERITY / EFFORT, (n, 1)),
    }
    if occlusion_conf is not None:
        confidence = np.maximum(occlusion_conf, 1e-6)
        out["Explanation confidence only"] = occlusion_conf
        out["Local occlusion"] = prob * (1.0 + occlusion_conf)
        out["Priority score (prob + evidence + severity/effort)"] = prob * (1.0 + confidence) * (SEVERITY / EFFORT)
    return out


def occlusion_confidence(model: MultiOutputClassifier, x_test: np.ndarray, x_train: np.ndarray,
                          gidx: Dict[str, List[int]]) -> np.ndarray:
    """Per-cycle, per-component occlusion confidence (mean drop over sensor groups),
    shape (n_cycles, n_components) -- matches evaluate_rankings' `local_component`."""
    baseline = np.mean(x_train, axis=0)
    occ = p3.occlusion_scores(model, x_test, baseline, gidx)
    return occ.mean(axis=1)


# ---------------------------------------------------------------------------
# E2: bound check on the three headline splits
# ---------------------------------------------------------------------------

def run_bound_check(features: pd.DataFrame, targets: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    n = len(features)
    idx = np.arange(n)
    gidx = p3.group_indices(features, groups)

    rows = []
    auc_rows = []

    # --- random stratified (seed 17, matches the manuscript's headline table) ---
    bundle = p3.train_models(features, targets, seed=17)
    occ_conf = occlusion_confidence(bundle["model"], bundle["x_test"], bundle["x_train"], gidx)
    scores = policy_scores(bundle["prob"], occ_conf)
    y_test = bundle["y_test"]
    aucs = cross_component_auc(y_test, bundle["prob"])
    bound = mean_union_bound(y_test, aucs)
    _, boot_bounds = bootstrap_bound_and_error(y_test, bundle["prob"], N_BOOT)
    b_lo, b_hi = ci95(boot_bounds)
    for name, score in scores.items():
        obs = top1_error(y_test, score)
        errs_b, _ = bootstrap_bound_and_error(y_test, score, N_BOOT)
        e_lo, e_hi = ci95(errs_b)
        rows.append({"split": "random_stratified", "policy": name, "observed_top1_error": obs,
                      "observed_ci_lo": e_lo, "observed_ci_hi": e_hi,
                      "bound": bound, "bound_ci_lo": b_lo, "bound_ci_hi": b_hi,
                      "bound_holds": bool(obs <= bound + 1e-9)})
    for comp in COMPONENTS:
        a, nn = aucs[comp]
        auc_rows.append({"split": "random_stratified", "component": comp, "cross_auc": a, "n_cycles": nn})

    # --- blocked group 5-fold (seed 2026, matches code/leakage_robustness.py) ---
    blocks = (idx // max(1, n // 10)).clip(max=9)
    gkf = GroupKFold(n_splits=5)
    y_all_blocked = np.zeros_like(y)
    prob_all_blocked = np.zeros((n, len(COMPONENTS)))
    occ_all_blocked = np.zeros((n, len(COMPONENTS)))
    filled = np.zeros(n, dtype=bool)
    for tr_b, te_b in gkf.split(idx, groups=blocks):
        prob_b, model_b, _, x_train_b, x_test_b = fit_predict(features, y, tr_b, te_b, SEED_BLOCKED)
        occ_conf_b = occlusion_confidence(model_b, x_test_b, x_train_b, gidx)
        y_all_blocked[te_b] = y[te_b]
        prob_all_blocked[te_b] = prob_b
        occ_all_blocked[te_b] = occ_conf_b
        filled[te_b] = True
    assert filled.all(), "GroupKFold did not cover every cycle"
    scores_blocked = policy_scores(prob_all_blocked, occ_all_blocked)
    aucs_blk = cross_component_auc(y_all_blocked, prob_all_blocked)
    bound_blk = mean_union_bound(y_all_blocked, aucs_blk)
    _, boot_bounds_blk = bootstrap_bound_and_error(y_all_blocked, prob_all_blocked, N_BOOT)
    b_lo_blk, b_hi_blk = ci95(boot_bounds_blk)
    for name, score in scores_blocked.items():
        obs = top1_error(y_all_blocked, score)
        errs_b, _ = bootstrap_bound_and_error(y_all_blocked, score, N_BOOT)
        e_lo, e_hi = ci95(errs_b)
        rows.append({"split": "blocked_group_5fold", "policy": name, "observed_top1_error": obs,
                      "observed_ci_lo": e_lo, "observed_ci_hi": e_hi,
                      "bound": bound_blk, "bound_ci_lo": b_lo_blk, "bound_ci_hi": b_hi_blk,
                      "bound_holds": bool(obs <= bound_blk + 1e-9)})
    for comp in COMPONENTS:
        a, nn = aucs_blk[comp]
        auc_rows.append({"split": "blocked_group_5fold", "component": comp, "cross_auc": a, "n_cycles": nn})

    # --- chronological, primary cut at 0.70 (matches code/leakage_robustness.py) ---
    cut = int(0.70 * n)
    prob_c, model_c, _, x_train_c, x_test_c = fit_predict(features, y, idx[:cut], idx[cut:], SEED_BLOCKED)
    occ_conf_c = occlusion_confidence(model_c, x_test_c, x_train_c, gidx)
    scores_c = policy_scores(prob_c, occ_conf_c)
    y_test_c = y[idx[cut:]]
    aucs_c = cross_component_auc(y_test_c, prob_c)
    bound_c = mean_union_bound(y_test_c, aucs_c)
    _, boot_bounds_c = bootstrap_bound_and_error(y_test_c, prob_c, N_BOOT)
    b_lo_c, b_hi_c = ci95(boot_bounds_c)
    for name, score in scores_c.items():
        obs = top1_error(y_test_c, score)
        errs_b, _ = bootstrap_bound_and_error(y_test_c, score, N_BOOT)
        e_lo, e_hi = ci95(errs_b)
        rows.append({"split": "temporal_chronological_0.70", "policy": name, "observed_top1_error": obs,
                      "observed_ci_lo": e_lo, "observed_ci_hi": e_hi,
                      "bound": bound_c, "bound_ci_lo": b_lo_c, "bound_ci_hi": b_hi_c,
                      "bound_holds": bool(obs <= bound_c + 1e-9)})
    for comp in COMPONENTS:
        a, nn = aucs_c[comp]
        auc_rows.append({"split": "temporal_chronological_0.70", "component": comp, "cross_auc": a, "n_cycles": nn})

    bound_df = pd.DataFrame(rows)
    auc_df = pd.DataFrame(auc_rows)
    bound_df.to_csv(RESULTS / "ceiling_bound_check.csv", index=False)
    auc_df.to_csv(RESULTS / "cross_component_auc.csv", index=False)
    return bound_df, auc_df


# ---------------------------------------------------------------------------
# Chronological generalization curve (multiple cut points)
# ---------------------------------------------------------------------------

def run_chronological_curve(features: pd.DataFrame, targets: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    n = len(features)
    idx = np.arange(n)
    gidx = p3.group_indices(features, groups)
    cut_fracs = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    rows = []
    for frac in cut_fracs:
        cut = int(frac * n)
        train_idx, test_idx = idx[:cut], idx[cut:]
        y_test = y[test_idx]
        if y_test.sum() == 0 or (y_test.sum(axis=1) > 0).sum() < 2:
            continue
        prob, model, _, x_train, x_test = fit_predict(features, y, train_idx, test_idx, SEED_BLOCKED)
        occ_conf = occlusion_confidence(model, x_test, x_train, gidx)
        pred = (prob >= 0.5).astype(int)
        macro_f1 = float(np.mean([f1_score(y_test[:, j], pred[:, j], zero_division=0) for j in range(len(COMPONENTS))]))
        scores = policy_scores(prob, occ_conf)
        aucs = cross_component_auc(y_test, prob)
        bound = mean_union_bound(y_test, aucs)
        errs_bound, boot_bounds = bootstrap_bound_and_error(y_test, prob, N_BOOT_CURVE)
        b_lo, b_hi = ci95(boot_bounds)
        mean_cross_auc = float(np.nanmean([aucs[c][0] for c in COMPONENTS]))
        for name, score in scores.items():
            obs = top1_error(y_test, score)
            errs_b, _ = bootstrap_bound_and_error(y_test, score, N_BOOT_CURVE)
            e_lo, e_hi = ci95(errs_b)
            rows.append({
                "cut_fraction": frac, "n_train": len(train_idx), "n_test": len(test_idx),
                "macro_f1": macro_f1, "mean_cross_auc": mean_cross_auc,
                "policy": name, "top1_accuracy": 1.0 - obs,
                "top1_accuracy_ci_lo": 1.0 - e_hi, "top1_accuracy_ci_hi": 1.0 - e_lo,
                "bound_top1_error": bound, "bound_ci_lo": b_lo, "bound_ci_hi": b_hi,
            })

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(RESULTS / "chronological_curve.csv", index=False)
    return curve_df


# ---------------------------------------------------------------------------
# E5 (partial): explanation agreement -- occlusion vs. permutation vs. grouped SHAP
# ---------------------------------------------------------------------------

def run_explanation_agreement(features: pd.DataFrame, targets: pd.DataFrame, groups: Dict[str, List[str]]) -> pd.DataFrame:
    import shap  # local import: optional dependency, only needed for this step

    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    gidx = p3.group_indices(features, groups)
    bundle = p3.train_models(features, targets, seed=17)
    model = bundle["model"]
    x_test = bundle["x_test"]
    x_train = bundle["x_train"]
    y_test = bundle["y_test"]

    baseline = np.mean(x_train, axis=0)
    occ = p3.occlusion_scores(model, x_test, baseline, gidx)  # (n, n_groups, n_components)
    occ_group_mean = occ.mean(axis=0)  # (n_groups, n_components) -- mean occlusion drop per group/component

    perm_df = p3.permutation_importance(model, x_test, y_test, gidx, seed=17)
    perm_pivot = perm_df.pivot_table(index="sensor", columns="component", values="importance").reindex(list(gidx.keys()))

    sensor_names = list(gidx.keys())
    shap_group = np.zeros((len(sensor_names), len(COMPONENTS)))
    for j, comp in enumerate(COMPONENTS):
        estimator = model.estimators_[j]
        explainer = shap.TreeExplainer(estimator)
        sv = explainer.shap_values(x_test)
        # shap_values: list [class0, class1] for binary RF, or an array with a trailing class axis.
        if isinstance(sv, list):
            sv1 = sv[1]
        else:
            sv1 = sv[..., 1] if sv.ndim == 3 else sv
        mean_abs = np.abs(sv1).mean(axis=0)  # per raw feature
        for s_idx, sensor in enumerate(sensor_names):
            cols = gidx[sensor]
            shap_group[s_idx, j] = mean_abs[cols].sum()

    rows = []
    for j, comp in enumerate(COMPONENTS):
        occ_vec = occ_group_mean[:, j]
        perm_vec = perm_pivot[comp].to_numpy()
        shap_vec = shap_group[:, j]
        rho_occ_perm, p_occ_perm = spearmanr(occ_vec, perm_vec)
        rho_occ_shap, p_occ_shap = spearmanr(occ_vec, shap_vec)
        rho_perm_shap, p_perm_shap = spearmanr(perm_vec, shap_vec)
        rows.append({
            "component": comp,
            "spearman_occlusion_permutation": float(rho_occ_perm), "p_occlusion_permutation": float(p_occ_perm),
            "spearman_occlusion_shap": float(rho_occ_shap), "p_occlusion_shap": float(p_occ_shap),
            "spearman_permutation_shap": float(rho_perm_shap), "p_permutation_shap": float(p_perm_shap),
        })
    agreement_df = pd.DataFrame(rows)
    agreement_df.to_csv(RESULTS / "explanation_agreement.csv", index=False)

    detail_rows = []
    for s_idx, sensor in enumerate(sensor_names):
        for j, comp in enumerate(COMPONENTS):
            detail_rows.append({"sensor": sensor, "component": comp,
                                 "occlusion_mean_drop": float(occ_group_mean[s_idx, j]),
                                 "permutation_ap_drop": float(perm_pivot[comp].iloc[s_idx]),
                                 "shap_mean_abs": float(shap_group[s_idx, j])})
    pd.DataFrame(detail_rows).to_csv(RESULTS / "explanation_agreement_detail.csv", index=False)
    return agreement_df


# ---------------------------------------------------------------------------
# LaTeX table writers
# ---------------------------------------------------------------------------

POLICY_SHORT = {
    "Probability only": "Probability only",
    "Severity-only policy": "Severity only",
    "Low-effort-only policy": "Low effort only",
    "Severity/effort heuristic": "Severity/effort",
    "Explanation confidence only": "Explanation conf.\\ only",
    "Local occlusion": "Local occlusion",
    "Priority score (prob + evidence + severity/effort)": "Priority score",
}


def write_bound_table(bound_df: pd.DataFrame, auc_df: pd.DataFrame) -> None:
    split_labels = {
        "random_stratified": "Random",
        "blocked_group_5fold": "Blocked 5-fold",
        "temporal_chronological_0.70": "Chronological",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-component AUC and the union-bound ceiling on top-1 root-cause "
        r"error (Sec.~\ref{sec:ceiling}), against each policy's observed top-1 error, with "
        r"2{,}000-resample bootstrap 95\% confidence intervals (in brackets). A policy sits "
        r"at the ceiling when its observed error interval overlaps the bound. Policy names "
        r"are abbreviated from Table~\ref{tab:ranking}; ``Priority score'' is the full "
        r"probability + evidence + severity/effort rule of Eq.~\ref{eq:priority}.}",
        r"\label{tab:ceilingbound}",
        r"\tiny",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{p{1.35cm}p{1.85cm}p{1.75cm}p{1.75cm}p{0.5cm}}",
        r"\hline",
        r"Split & Policy & Obs.\ error & Bound & At \\",
        r" & & [95\% CI] & [95\% CI] & ceil.\ \\",
        r"\hline",
    ]
    for split in ["random_stratified", "blocked_group_5fold", "temporal_chronological_0.70"]:
        sub = bound_df[bound_df["split"] == split]
        for _, r in sub.iterrows():
            at = "Y" if r["bound_holds"] else "N"
            lines.append(
                f"{split_labels[split]} & {POLICY_SHORT.get(r['policy'], p3.latex_escape(r['policy']))} & "
                f"{r['observed_top1_error']:.3f} [{r['observed_ci_lo']:.3f}, {r['observed_ci_hi']:.3f}] & "
                f"{r['bound']:.3f} [{r['bound_ci_lo']:.3f}, {r['bound_ci_hi']:.3f}] & {at} \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_ceiling_bound_table.tex").write_text("\n".join(lines), encoding="utf-8")

    auc_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-component AUC per component and split: the probability that the "
        r"best true-positive component's score exceeds this component's score on fault-containing "
        r"cycles where it is a true negative.}",
        r"\label{tab:crossauc}",
        r"\footnotesize",
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Split & Component & Cross-AUC & $n$ cycles \\",
        r"\hline",
    ]
    for split in ["random_stratified", "blocked_group_5fold", "temporal_chronological_0.70"]:
        sub = auc_df[auc_df["split"] == split]
        for _, r in sub.iterrows():
            auc_lines.append(f"{split_labels[split]} & {r['component'].capitalize()} & {r['cross_auc']:.3f} & {r['n_cycles']} \\\\")
        auc_lines.append(r"\hline")
    auc_lines += [r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_cross_auc_table.tex").write_text("\n".join(auc_lines), encoding="utf-8")


def write_chronological_table(curve_df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Chronological generalization curve: macro-F1, mean cross-component AUC, "
        r"and top-1 root-cause accuracy of two ranking policies as the training/test cut point "
        r"moves later in the cycle sequence. All rows use the leakage-safe blocked-model seed "
        r"(2026); the 0.70 row matches Table~\ref{tab:leakage_robustness}. At cut points 0.50-0.65 "
        r"the cooler-fault training fold contains no healthy-cooler cycles (Sec.~V), so the cooler "
        r"classifier degenerates to a constant score; this drags down every score that includes it "
        r"and is a training-fold pathology of the early cuts, not evidence about ranking quality.}",
        r"\label{tab:chronocurve}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{rrrrr}",
        r"\hline",
        r"Cut & Macro-F1 & Mean & Top-1 & Top-1 \\",
        r" & & cross-AUC & prob.\ only & priority \\",
        r"\hline",
    ]
    prob_only = curve_df[curve_df["policy"] == "Probability only"].set_index("cut_fraction")
    prio = curve_df[curve_df["policy"] == "Priority score (prob + evidence + severity/effort)"].set_index("cut_fraction")
    for frac in sorted(prob_only.index):
        r1 = prob_only.loc[frac]
        row2 = prio.loc[frac] if frac in prio.index else None
        top1_prio = f"{row2['top1_accuracy']:.3f}" if row2 is not None else "--"
        lines.append(
            f"{frac:.2f} & {r1['macro_f1']:.3f} & {r1['mean_cross_auc']:.3f} & "
            f"{r1['top1_accuracy']:.3f} & {top1_prio} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_chronological_curve_table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_agreement_table(agreement_df: pd.DataFrame) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Spearman rank agreement, across the 17 sensor groups, between grouped "
        r"occlusion, permutation importance, and grouped SHAP (\texttt{TreeExplainer}) attribution, "
        r"per component. Random-split test set, seed 17.}",
        r"\label{tab:explainagree}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Component & Occl.--Perm. & Occl.--SHAP & Perm.--SHAP \\",
        r"\hline",
    ]
    for _, r in agreement_df.iterrows():
        lines.append(
            f"{r['component'].capitalize()} & {r['spearman_occlusion_permutation']:.3f} & "
            f"{r['spearman_occlusion_shap']:.3f} & {r['spearman_permutation_shap']:.3f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_explanation_agreement_table.tex").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(bound_df: pd.DataFrame, curve_df: pd.DataFrame) -> None:
    p3.configure_plot_style()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    splits = ["random_stratified", "blocked_group_5fold", "temporal_chronological_0.70"]
    split_x = {s: i for i, s in enumerate(splits)}
    for policy in bound_df["policy"].unique():
        sub = bound_df[bound_df["policy"] == policy]
        xs = [split_x[s] + (hash(policy) % 7) * 0.02 for s in sub["split"]]
        ax.errorbar(xs, sub["observed_top1_error"],
                    yerr=[sub["observed_top1_error"] - sub["observed_ci_lo"], sub["observed_ci_hi"] - sub["observed_top1_error"]],
                    marker="o", linestyle="none", label=policy, alpha=0.85, capsize=2)
    for s, x in split_x.items():
        row = bound_df[bound_df["split"] == s].iloc[0]
        ax.plot([x - 0.3, x + 0.3], [row["bound"], row["bound"]], color="black", linewidth=2.0, zorder=5)
        ax.fill_between([x - 0.3, x + 0.3], row["bound_ci_lo"], row["bound_ci_hi"], color="black", alpha=0.12)
    ax.set_xticks(list(split_x.values()))
    ax.set_xticklabels(["Random\nstratified", "Blocked\ngroup 5-fold", "Chronological\n(0.70 cut)"])
    ax.set_ylabel("Top-1 root-cause error")
    ax.set_title("Observed error vs. the union-bound ceiling (black bar)")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ceiling_bound_vs_observed.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    prob_only = curve_df[curve_df["policy"] == "Probability only"].sort_values("cut_fraction")
    prio = curve_df[curve_df["policy"] == "Priority score (prob + evidence + severity/effort)"].sort_values("cut_fraction")
    sev_eff = curve_df[curve_df["policy"] == "Severity/effort heuristic"].sort_values("cut_fraction")

    ax = axes[0]
    ax.plot(prob_only["cut_fraction"], prob_only["macro_f1"], marker="o", label="Macro-F1")
    ax.plot(prob_only["cut_fraction"], prob_only["mean_cross_auc"], marker="s", label="Mean cross-AUC")
    ax.set_xlabel("Chronological cut fraction")
    ax.set_ylabel("Value")
    ax.set_title("Separability falls as the cut moves later")
    ax.legend(fontsize=8)

    ax = axes[1]
    for df, label, marker in [(prob_only, "Probability only", "o"), (prio, "Priority score", "^"), (sev_eff, "Severity/effort", "s")]:
        ax.errorbar(df["cut_fraction"], df["top1_accuracy"],
                    yerr=[df["top1_accuracy"] - df["top1_accuracy_ci_lo"], df["top1_accuracy_ci_hi"] - df["top1_accuracy"]],
                    marker=marker, label=label, capsize=2)
    ax.set_xlabel("Chronological cut fraction")
    ax.set_ylabel("Top-1 root-cause accuracy")
    ax.set_title("Ranking policies stay tied along the curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ceiling_chronological_curve.png", bbox_inches="tight")
    plt.close(fig)


def write_constants_table() -> None:
    """E1: disclose the severity/effort constants read directly from the code.
    Zero runtime. Note: alpha and beta (Eq. edgeweight, the occlusion/permutation
    graph mixing weights) are not implemented anywhere in the pipeline -- the
    evidence graph is built from occlusion means alone (save_figures in
    paper3_diagnostics_experiment.py) -- so no values for them can be disclosed
    without inventing numbers. That mismatch is flagged in the manuscript text,
    not silently fixed here."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Severity ($S_c$) and effort ($E_c$) constants used by the priority "
        r"score (Eq.~\ref{eq:priority}) and by the static policy baselines, read directly "
        r"from \texttt{code/paper3\_diagnostics\_experiment.py}. No value is reported for "
        r"the evidence-graph mixing weights $\alpha,\beta$ of Eq.~\ref{eq:edgeweight}: they "
        r"are not implemented in the released code, so no number exists to disclose (see "
        r"Sec.~\ref{sec:method}).}",
        r"\label{tab:constants}",
        r"\footnotesize",
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Component & $S_c$ & $E_c$ & $S_c/E_c$ \\",
        r"\hline",
    ]
    for c, s, e in zip(COMPONENTS, SEVERITY, EFFORT):
        lines.append(f"{c.capitalize()} & {s:.1f} & {e:.1f} & {s / e:.3f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_constants_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.time()
    write_constants_table()
    features, targets, groups = p3.build_feature_table()
    print(f"[ceiling_analysis] E2 bound check...", flush=True)
    bound_df, auc_df = run_bound_check(features, targets, groups)
    print(bound_df.to_string(index=False))
    print(auc_df.to_string(index=False))
    write_bound_table(bound_df, auc_df)

    print(f"[ceiling_analysis] chronological curve...", flush=True)
    curve_df = run_chronological_curve(features, targets, groups)
    print(curve_df.to_string(index=False))
    write_chronological_table(curve_df)

    print(f"[ceiling_analysis] explanation agreement (SHAP)...", flush=True)
    try:
        agreement_df = run_explanation_agreement(features, targets, groups)
        print(agreement_df.to_string(index=False))
        write_agreement_table(agreement_df)
    except ImportError as exc:
        print(f"[ceiling_analysis] SKIPPED explanation agreement: {exc}")

    make_figures(bound_df, curve_df)
    print(f"[ceiling_analysis] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
