"""
E5 classifier comparators (top10percentplan.md Sec. 5; PHASE1_SKIPPED.md):
HistGradientBoosting and a 1-D CNN on raw per-cycle sensor streams, run on the
same three splits as ceiling_analysis.py's run_bound_check (random stratified
seed 17, blocked group 5-fold seed 2026, chronological 0.70 cut seed 2026), and
reporting beside the random forest using the identical union-bound machinery
(cross_component_auc, mean_union_bound, top1_error from ceiling_analysis.py).

Seeds: three per split (17, 2026, 4242) rather than the ten-seed budget used by
degradation_walk.py -- the existing ceiling-check table itself reports a single
seed per split, so three seeds here already adds a seed-variance estimate on top
of what E2 reports, without the added runtime of a full ten-seed sweep on a
second and third classifier family (see FIX_LOG.md 2026-09-10 for the reasoning).
Fixed here so every rerun is identical.

CNN input: the 17 raw sensor streams (data/uci_hydraulic_raw/*.txt) are
block-averaged down to a common rate of 60 samples/cycle (1 Hz, the native rate
of TS1-4, VS1, CE, CP, SE) -- PS1-6/EPS1 (100 Hz, 6000 samples) are averaged in
non-overlapping blocks of 100, FS1/FS2 (10 Hz, 600 samples) in blocks of 10. This
is a disclosed downsampling choice, not a value read from any file. The network
is a small multi-output 1-D CNN (two conv blocks, global average pool, one
sigmoid head per component), trained with BCEWithLogitsLoss, Adam, early
stopping on a held-out 15% of the training split.

Outputs: results/comparator_classifiers.csv (one row per split x seed x
classifier), results/comparator_classifiers_summary.csv (mean/std across seeds),
manuscript/generated_comparator_table.tex.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import ceiling_analysis as ca
import paper3_diagnostics_experiment as p3

ROOT = p3.ROOT
RAW_DIR = p3.RAW_DIR
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
RESULTS.mkdir(parents=True, exist_ok=True)

COMPONENTS = p3.COMPONENTS
SEEDS = [17, 2026, 4242]
SEED_BLOCKED_HEADLINE = 2026  # matches ceiling_analysis.SEED_BLOCKED


# ---------------------------------------------------------------------------
# Splits (identical construction to ceiling_analysis.run_bound_check)
# ---------------------------------------------------------------------------

def make_splits(n: int, targets: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    splits = {}
    tr, te = train_test_split(idx, test_size=0.30, random_state=17, stratify=targets["any_fault"])
    splits["random_stratified"] = (tr, te)
    blocks = (idx // max(1, n // 10)).clip(max=9)
    gkf = GroupKFold(n_splits=5)
    tr_b, te_b = next(gkf.split(idx, groups=blocks))
    splits["blocked_group_5fold"] = (tr_b, te_b)
    cut = int(0.70 * n)
    splits["temporal_chronological_0.70"] = (idx[:cut], idx[cut:])
    return splits


# ---------------------------------------------------------------------------
# HistGradientBoosting comparator
# ---------------------------------------------------------------------------

def hgb_positive_proba(estimator: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    proba = estimator.predict_proba(x)
    classes = list(estimator.classes_)
    if proba.shape[1] == 1:
        return np.full(x.shape[0], 1.0 if classes[0] == 1 else 0.0)
    return proba[:, classes.index(1)]


def fit_hgb(features: pd.DataFrame, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
            seed: int) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.iloc[train_idx].to_numpy(float))
    x_test = scaler.transform(features.iloc[test_idx].to_numpy(float))
    probs = []
    for j in range(y.shape[1]):
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, l2_regularization=0.02,
                                                random_state=seed)
        model.fit(x_train, y[train_idx, j])
        probs.append(hgb_positive_proba(model, x_test))
    return np.vstack(probs).T


# ---------------------------------------------------------------------------
# 1-D CNN comparator
# ---------------------------------------------------------------------------

def resample_common_rate(matrix: np.ndarray, target_len: int = 60) -> np.ndarray:
    n, cols = matrix.shape
    factor = cols // target_len
    if factor <= 1:
        return matrix[:, :target_len]
    usable = factor * target_len
    return matrix[:, :usable].reshape(n, target_len, factor).mean(axis=2)


def load_cnn_tensor() -> np.ndarray:
    """(n_cycles, n_sensors, 60) tensor, sensors in p3.SENSORS order."""
    channels = []
    for sensor in p3.SENSORS:
        matrix = np.loadtxt(RAW_DIR / f"{sensor}.txt")
        channels.append(resample_common_rate(matrix, target_len=60))
    return np.stack(channels, axis=1)  # (n, 17, 60)


class CycleCNN(nn.Module):
    def __init__(self, n_channels: int, n_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))


def fit_cnn(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int,
            max_epochs: int = 60, patience: int = 8) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # per-channel standardization on the training split
    mean = X[train_idx].mean(axis=(0, 2), keepdims=True)
    std = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-8
    Xn = (X - mean) / std

    n_train = len(train_idx)
    val_n = max(1, int(0.15 * n_train))
    perm = rng.permutation(n_train)
    val_local, fit_local = perm[:val_n], perm[val_n:]
    fit_idx = train_idx[fit_local]
    val_idx = train_idx[val_local]

    model = CycleCNN(n_channels=X.shape[1], n_outputs=y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    fit_ds = TensorDataset(torch.from_numpy(Xn[fit_idx]).float(), torch.from_numpy(y[fit_idx]).float())
    loader = DataLoader(fit_ds, batch_size=64, shuffle=True, drop_last=False,
                         generator=torch.Generator().manual_seed(seed))
    x_val = torch.from_numpy(Xn[val_idx]).float()
    y_val = y[val_idx]

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(loss_fn(val_logits, torch.from_numpy(y_val).float()))
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(Xn[test_idx]).float())
        prob = torch.sigmoid(test_logits).numpy()
    return prob


# ---------------------------------------------------------------------------
# Shared evaluation against the union bound
# ---------------------------------------------------------------------------

def evaluate_probs(y_test: np.ndarray, prob: np.ndarray) -> dict:
    pred = (prob >= 0.5).astype(int)
    macro_f1 = float(np.mean([f1_score(y_test[:, j], pred[:, j], zero_division=0) for j in range(y_test.shape[1])]))
    aucs = ca.cross_component_auc(y_test, prob)
    mean_cross_auc = float(np.nanmean([aucs[c][0] for c in COMPONENTS]))
    bound = ca.mean_union_bound(y_test, aucs)
    obs_top1_error = ca.top1_error(y_test, prob)
    return {"macro_f1": macro_f1, "mean_cross_auc": mean_cross_auc, "bound_top1_error": bound,
            "top1_accuracy": 1.0 - obs_top1_error}


def main() -> None:
    t0 = time.time()
    features, targets, groups = p3.build_feature_table()
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(int)
    n = len(features)
    splits = make_splits(n, targets)

    rows = []

    print("[comparators] HistGradientBoosting ...", flush=True)
    for split_name, (train_idx, test_idx) in splits.items():
        for seed in SEEDS:
            prob = fit_hgb(features, y, train_idx, test_idx, seed)
            metrics = evaluate_probs(y[test_idx], prob)
            rows.append({"split": split_name, "classifier": "HistGradientBoosting", "seed": seed, **metrics})
            print(f"  {split_name} seed={seed}: macro_f1={metrics['macro_f1']:.3f} "
                  f"top1={metrics['top1_accuracy']:.3f} cross_auc={metrics['mean_cross_auc']:.3f}", flush=True)

    print("[comparators] 1-D CNN on raw resampled streams ...", flush=True)
    X_cnn = load_cnn_tensor()
    for split_name, (train_idx, test_idx) in splits.items():
        for seed in SEEDS:
            prob = fit_cnn(X_cnn, y.astype(np.float32), train_idx, test_idx, seed)
            metrics = evaluate_probs(y[test_idx], prob)
            rows.append({"split": split_name, "classifier": "1D-CNN (raw, resampled 1 Hz)", "seed": seed, **metrics})
            print(f"  {split_name} seed={seed}: macro_f1={metrics['macro_f1']:.3f} "
                  f"top1={metrics['top1_accuracy']:.3f} cross_auc={metrics['mean_cross_auc']:.3f}", flush=True)

    print("[comparators] Random Forest (reference, from ceiling_analysis machinery) ...", flush=True)
    for split_name, (train_idx, test_idx) in splits.items():
        seed = SEED_BLOCKED_HEADLINE if split_name != "random_stratified" else 17
        for extra_seed in SEEDS:
            prob, _, _, _, _ = ca.fit_predict(features, y, train_idx, test_idx, extra_seed)
            metrics = evaluate_probs(y[test_idx], prob)
            rows.append({"split": split_name, "classifier": "Random Forest (headline)", "seed": extra_seed, **metrics})

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparator_classifiers.csv", index=False)

    summary = (
        df.groupby(["split", "classifier"])
        .agg(macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
             top1_mean=("top1_accuracy", "mean"), top1_std=("top1_accuracy", "std"),
             mean_cross_auc=("mean_cross_auc", "mean"), bound_top1_error=("bound_top1_error", "mean"),
             n_seeds=("seed", "nunique"))
        .reset_index()
    )
    summary.to_csv(RESULTS / "comparator_classifiers_summary.csv", index=False)
    print(summary.to_string(index=False))

    write_table(summary)
    print(f"[comparators] done in {time.time() - t0:.1f}s")


def write_table(summary: pd.DataFrame) -> None:
    split_labels = {
        "random_stratified": "Random",
        "blocked_group_5fold": "Blocked 5-fold",
        "temporal_chronological_0.70": "Chronological",
    }
    clf_order = ["Random Forest (headline)", "HistGradientBoosting", "1D-CNN (raw, resampled 1 Hz)"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Classifier comparators on the three headline splits (mean $\pm$ std over "
        r"three seeds: 17, 2026, 4242). HistGradientBoosting uses the same 187-feature cycle "
        r"table as the random forest; the 1-D CNN uses the 17 raw sensor streams "
        r"block-averaged to a common 60-sample/cycle rate (Sec.~\ref{sec:comparators}). "
        r"Mean cross-AUC and the union-bound top-1 error are computed identically to "
        r"Table~\ref{tab:ceilingbound}.}",
        r"\label{tab:comparators}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Split & Classifier & Macro-F1 & Top-1 acc.\ & Mean cross-AUC \\",
        r"\hline",
    ]
    for split in ["random_stratified", "blocked_group_5fold", "temporal_chronological_0.70"]:
        for clf in clf_order:
            row = summary[(summary["split"] == split) & (summary["classifier"] == clf)]
            if row.empty:
                continue
            r = row.iloc[0]
            lines.append(
                f"{split_labels[split]} & {p3.latex_escape(clf)} & "
                f"{r['macro_f1_mean']:.3f} $\\pm$ {r['macro_f1_std']:.3f} & "
                f"{r['top1_mean']:.3f} $\\pm$ {r['top1_std']:.3f} & {r['mean_cross_auc']:.3f} \\\\"
            )
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}", ""]
    (MANUSCRIPT / "generated_comparator_table.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
