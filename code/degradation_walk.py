"""
E3: full degradation walk (top10percentplan.md Sec. 5, E3; PHASE1_SKIPPED.md).

Four axes, five levels each, ten fixed seeds -- 200 forest fits total, matching the
plan's runtime budget. Every cell uses the SAME blocked-group train/test split (the
first fold of the 5-fold GroupKFold over 10 contiguous cycle blocks, identical
construction to ceiling_analysis.py's `blocked_group_5fold`), held fixed across axes
and seeds per the plan step 5 instruction ("holding the blocked-group split fixed and
varying ..."); only the classifier's random_state and the axis-specific degradation
(subsampling / family removal / label flip / raw-window truncation) change across
cells. Reuses ceiling_analysis.cross_component_auc, mean_union_bound and
bootstrap_bound_and_error for the union-bound machinery, and
paper3_diagnostics_experiment for feature loading, occlusion, and the ranking
policies, so every metric definition is identical to the ones already reported.

Sensor families (7): the plan specifies "sensor families removed {0..4 of 7}" without
naming the 7 families. This script discloses the grouping used, by physical
measurement type per data/uci_hydraulic_raw/documentation.txt: pressure (PS1-PS6),
volume flow (FS1, FS2), temperature (TS1-TS4), motor power (EPS1), vibration (VS1),
cooling (CE, CP), and efficiency factor (SE). This is a methodological choice made
explicit here, not a value read from any file; a different grouping would give a
different curve, and readers/reviewers can inspect FAMILIES below.

Outputs: results/degradation_walk.csv (one row per axis x level x seed x policy),
plus results/degradation_walk_summary.csv (mean/CI across seeds per axis x level x
policy) and figures/degradation_walk.png (one panel per axis: top-1 accuracy of
Probability-only and the Priority score vs. mean cross-component AUC, with the
undegraded baseline point marked).

Check (plan step 5): the undegraded cell (level 0 of every axis, i.e. no
degradation) must reproduce the blocked-group macro-F1 of 0.965 within seed
variance -- verified in `main()` and printed before the walk proceeds.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

import ceiling_analysis as ca
import paper3_diagnostics_experiment as p3

ROOT = p3.ROOT
RAW_DIR = p3.RAW_DIR
FIG_DIR = p3.FIG_DIR
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
RESULTS.mkdir(parents=True, exist_ok=True)

COMPONENTS = p3.COMPONENTS
SEVERITY = p3.SEVERITY
EFFORT = p3.EFFORT

# 10 fixed seeds: 17 and 2026 (matching the two seeds used elsewhere in this paper),
# plus eight more, fixed here so every rerun is identical.
SEEDS: List[int] = [17, 2026, 101, 202, 303, 404, 505, 606, 707, 808]

TRAIN_FRACTIONS = [1.0, 0.3, 0.1, 0.03, 0.01]
N_FAMILIES_REMOVED = [0, 1, 2, 3, 4]
LABEL_FLIP_RATES = [0.0, 0.05, 0.1, 0.2, 0.3]
CYCLE_WINDOW_FRACS = [1.0, 0.5, 0.25, 0.10, 0.05]

FAMILIES: Dict[str, List[str]] = {
    "pressure": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"],
    "flow": ["FS1", "FS2"],
    "temperature": ["TS1", "TS2", "TS3", "TS4"],
    "motor_power": ["EPS1"],
    "vibration": ["VS1"],
    "cooling": ["CE", "CP"],
    "efficiency_factor": ["SE"],
}
FAMILY_NAMES = list(FAMILIES.keys())
assert len(FAMILY_NAMES) == 7


def blocked_split(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The fixed blocked-group split: first fold of a 5-fold GroupKFold over 10
    contiguous cycle blocks. Identical block construction to ceiling_analysis.py."""
    idx = np.arange(n)
    blocks = (idx // max(1, n // 10)).clip(max=9)
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(idx, groups=blocks))
    return train_idx, test_idx


def fit_and_score(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                   gidx: Dict[str, List[int]], seed: int) -> Dict[str, object]:
    scaler = StandardScaler()
    xt = scaler.fit_transform(x_train)
    xs = scaler.transform(x_test)
    base = RandomForestClassifier(n_estimators=260, min_samples_leaf=2, random_state=seed,
                                   class_weight="balanced_subsample", n_jobs=-1)
    model = MultiOutputClassifier(base)
    model.fit(xt, y_train)
    prob = np.vstack([p3.positive_class_proba(e, xs) for e in model.estimators_]).T
    baseline = np.mean(xt, axis=0)
    occ = p3.occlusion_scores(model, xs, baseline, gidx)
    occ_conf = occ.mean(axis=1)
    pred = (prob >= 0.5).astype(int)
    macro_f1 = float(np.mean([f1_score(y_test[:, j], pred[:, j], zero_division=0) for j in range(len(COMPONENTS))]))
    scores = ca.policy_scores(prob, occ_conf)
    aucs = ca.cross_component_auc(y_test, prob)
    mean_cross_auc = float(np.nanmean([aucs[c][0] for c in COMPONENTS]))
    bound = ca.mean_union_bound(y_test, aucs)
    rows = []
    for name, score in scores.items():
        obs = ca.top1_error(y_test, score)
        rows.append({"policy": name, "top1_accuracy": 1.0 - obs, "bound_top1_error": bound})
    return {"rows": rows, "macro_f1": macro_f1, "mean_cross_auc": mean_cross_auc, "bound": bound}


def run_training_fraction(features: pd.DataFrame, targets: pd.DataFrame, groups: Dict[str, List[str]],
                           train_idx: np.ndarray, test_idx: np.ndarray) -> List[dict]:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    gidx = p3.group_indices(features, groups)
    x_all = features.to_numpy(float)
    y_test = y[test_idx]
    rows = []
    for frac in TRAIN_FRACTIONS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            if frac >= 1.0:
                sub_idx = train_idx
            else:
                n_sub = max(len(COMPONENTS) * 2, int(round(frac * len(train_idx))))
                n_sub = min(n_sub, len(train_idx))
                sub_idx = rng.choice(train_idx, size=n_sub, replace=False)
            result = fit_and_score(x_all[sub_idx], y[sub_idx], x_all[test_idx], y_test, gidx, seed)
            for r in result["rows"]:
                rows.append({"axis": "training_fraction", "level": frac, "seed": seed,
                             "n_train": len(sub_idx), "macro_f1": result["macro_f1"],
                             "mean_cross_auc": result["mean_cross_auc"], **r})
    return rows


def run_sensor_family_removal(features: pd.DataFrame, targets: pd.DataFrame, groups: Dict[str, List[str]],
                               train_idx: np.ndarray, test_idx: np.ndarray) -> List[dict]:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    rows = []
    for k in N_FAMILIES_REMOVED:
        for seed in SEEDS:
            rng = np.random.default_rng(seed + 90000)
            removed = list(rng.choice(FAMILY_NAMES, size=k, replace=False)) if k > 0 else []
            removed_sensors = set(s for fam in removed for s in FAMILIES[fam])
            keep_sensors = [s for s in groups if s not in removed_sensors]
            keep_cols = [c for s in keep_sensors for c in groups[s]]
            sub_features = features[keep_cols]
            sub_groups = {s: groups[s] for s in keep_sensors}
            gidx = p3.group_indices(sub_features, sub_groups)
            x_all = sub_features.to_numpy(float)
            result = fit_and_score(x_all[train_idx], y[train_idx], x_all[test_idx], y[test_idx], gidx, seed)
            for r in result["rows"]:
                rows.append({"axis": "sensor_families_removed", "level": k, "seed": seed,
                             "removed_families": ";".join(removed), "n_features": len(keep_cols),
                             "macro_f1": result["macro_f1"], "mean_cross_auc": result["mean_cross_auc"], **r})
    return rows


def run_label_flip(features: pd.DataFrame, targets: pd.DataFrame, groups: Dict[str, List[str]],
                    train_idx: np.ndarray, test_idx: np.ndarray) -> List[dict]:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    gidx = p3.group_indices(features, groups)
    x_all = features.to_numpy(float)
    y_test = y[test_idx]
    rows = []
    for rate in LABEL_FLIP_RATES:
        for seed in SEEDS:
            rng = np.random.default_rng(seed + 190000)
            y_train = y[train_idx].copy()
            if rate > 0:
                flip_mask = rng.random(y_train.shape) < rate
                y_train = np.where(flip_mask, 1 - y_train, y_train)
            result = fit_and_score(x_all[train_idx], y_train, x_all[test_idx], y_test, gidx, seed)
            for r in result["rows"]:
                rows.append({"axis": "label_flip_rate", "level": rate, "seed": seed,
                             "n_flipped": int((y_train != y[train_idx]).sum()),
                             "macro_f1": result["macro_f1"], "mean_cross_auc": result["mean_cross_auc"], **r})
    return rows


def truncated_feature_table(raw_matrices: Dict[str, np.ndarray], frac: float) -> pd.DataFrame:
    rows = []
    groups: Dict[str, List[str]] = {}
    for sensor, matrix in raw_matrices.items():
        n_cols = matrix.shape[1]
        keep = max(2, int(round(frac * n_cols)))
        truncated = matrix[:, :keep]
        feats = p3.row_features(truncated)
        names = [f"{sensor}_{name}" for name in
                 ["mean", "std", "rms", "min", "max", "ptp", "slope", "skew", "kurt", "energy", "crest"]]
        groups[sensor] = names
        rows.append(pd.DataFrame(feats, columns=names))
    features = pd.concat(rows, axis=1)
    return features, groups


def run_cycle_window(targets: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> List[dict]:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    y_test = y[test_idx]
    print("[degradation_walk] loading raw sensor streams for cycle-window axis ...", flush=True)
    raw_matrices = {s: np.loadtxt(RAW_DIR / f"{s}.txt") for s in p3.SENSORS}
    rows = []
    for frac in CYCLE_WINDOW_FRACS:
        features, groups = truncated_feature_table(raw_matrices, frac)
        gidx = p3.group_indices(features, groups)
        x_all = features.to_numpy(float)
        for seed in SEEDS:
            result = fit_and_score(x_all[train_idx], y[train_idx], x_all[test_idx], y_test, gidx, seed)
            for r in result["rows"]:
                rows.append({"axis": "cycle_window_retained", "level": frac, "seed": seed,
                             "macro_f1": result["macro_f1"], "mean_cross_auc": result["mean_cross_auc"], **r})
        print(f"  cycle_window_retained={frac:.2f}: macro_f1(last seed)={result['macro_f1']:.3f}", flush=True)
    return rows


def make_figures(df: pd.DataFrame) -> None:
    p3.configure_plot_style()
    axes_order = [
        ("training_fraction", "Training fraction retained"),
        ("sensor_families_removed", "Sensor families removed (of 7)"),
        ("label_flip_rate", "Training label-flip rate"),
        ("cycle_window_retained", "Cycle window retained (fraction)"),
    ]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.5))
    for ax, (axis, xlabel) in zip(axs.flat, axes_order):
        sub = df[df["axis"] == axis]
        for policy, marker in [("Probability only", "o"),
                                ("Priority score (prob + evidence + severity/effort)", "^"),
                                ("Severity/effort heuristic", "s")]:
            g = sub[sub["policy"] == policy].groupby("level")["top1_accuracy"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g["level"], g["mean"], yerr=g["std"], marker=marker, label=policy.split(" (")[0], capsize=2)
        g_auc = sub[sub["policy"] == "Probability only"].groupby("level")["mean_cross_auc"].mean().reset_index()
        ax2 = ax.twinx()
        ax2.plot(g_auc["level"], g_auc["mean_cross_auc"], color="black", linestyle="--", marker="x", label="Mean cross-AUC", alpha=0.6)
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Mean cross-AUC", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Top-1 root-cause accuracy")
        ax.set_ylim(-0.02, 1.05)
    axs.flat[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Degradation walk: top-1 accuracy vs. mean cross-component AUC, blocked-group split fixed")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "degradation_walk.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    features, targets, groups = p3.build_feature_table()
    n = len(features)
    train_idx, test_idx = blocked_split(n)

    # Check (plan step 5): undegraded cell reproduces blocked macro-F1 ~0.965.
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    gidx = p3.group_indices(features, groups)
    x_all = features.to_numpy(float)
    baseline_check = fit_and_score(x_all[train_idx], y[train_idx], x_all[test_idx], y[test_idx], gidx, seed=2026)
    print(f"[degradation_walk] undegraded blocked-split macro-F1 (seed 2026) = {baseline_check['macro_f1']:.4f} "
          f"(expected ~0.965 per leakage_robustness.py's blocked_group_5fold mean)", flush=True)

    all_rows: List[dict] = []
    print("[degradation_walk] axis 1/4: training fraction ...", flush=True)
    all_rows += run_training_fraction(features, targets, groups, train_idx, test_idx)
    print("[degradation_walk] axis 2/4: sensor families removed ...", flush=True)
    all_rows += run_sensor_family_removal(features, targets, groups, train_idx, test_idx)
    print("[degradation_walk] axis 3/4: label-flip rate ...", flush=True)
    all_rows += run_label_flip(features, targets, groups, train_idx, test_idx)
    print("[degradation_walk] axis 4/4: cycle window retained ...", flush=True)
    all_rows += run_cycle_window(targets, train_idx, test_idx)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "degradation_walk.csv", index=False)

    summary = (
        df.groupby(["axis", "level", "policy"])
        .agg(top1_mean=("top1_accuracy", "mean"), top1_std=("top1_accuracy", "std"),
             mean_cross_auc=("mean_cross_auc", "mean"), macro_f1_mean=("macro_f1", "mean"),
             bound_top1_error=("bound_top1_error", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )
    summary.to_csv(RESULTS / "degradation_walk_summary.csv", index=False)

    make_figures(df)

    print(summary.to_string(index=False))
    print(f"[degradation_walk] done in {time.time() - t0:.1f}s, {len(df)} rows "
          f"({df.groupby(['axis', 'level', 'seed']).ngroups} unique fits)")


if __name__ == "__main__":
    main()
