"""
Paper 3 explainable diagnostics and maintenance priority experiment.

The experiment uses the UCI Condition Monitoring of Hydraulic Systems data set.
It is a real multi-sensor test-rig data set with component-condition labels for
cooler, valve, pump, and accumulator states. The method uses model-agnostic
occlusion and permutation explanations to build a diagnostic evidence graph and
a transparent component-priority score. The graph is not a causal discovery
claim; it is an evidence-weighted view of sensor-to-component support.
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    label_ranking_average_precision_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "uci_hydraulic_raw"
PROC_DIR = DATA_DIR / "processed"
FIG_DIR = ROOT / "figures"
DATA_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
ARCHIVE = DATA_DIR / "uci_hydraulic.zip"
SENSORS = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
COMPONENTS = ["cooler", "valve", "pump", "accumulator"]
SEVERITY = np.array([4.0, 3.0, 3.5, 3.2], dtype=float)
EFFORT = np.array([1.8, 1.5, 2.2, 1.4], dtype=float)


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.1,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.65,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
        }
    )


def download_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        urllib.request.urlretrieve(DATA_URL, ARCHIVE)
    if not RAW_DIR.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE, "r") as zf:
            zf.extractall(RAW_DIR)


def row_features(matrix: np.ndarray) -> np.ndarray:
    eps = 1e-10
    mean = np.mean(matrix, axis=1)
    std = np.std(matrix, axis=1)
    rms = np.sqrt(np.mean(matrix**2, axis=1))
    minimum = np.min(matrix, axis=1)
    maximum = np.max(matrix, axis=1)
    ptp = maximum - minimum
    slope = (matrix[:, -1] - matrix[:, 0]) / max(matrix.shape[1] - 1, 1)
    sk = skew(matrix, axis=1, bias=False)
    ku = kurtosis(matrix, axis=1, fisher=True, bias=False)
    energy = np.mean(matrix**2, axis=1)
    crest = np.max(np.abs(matrix), axis=1) / (rms + eps)
    return np.vstack([mean, std, rms, minimum, maximum, ptp, slope, sk, ku, energy, crest]).T


def build_feature_table(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    feature_path = PROC_DIR / "paper3_features.csv"
    target_path = PROC_DIR / "paper3_targets.csv"
    groups_path = PROC_DIR / "paper3_feature_groups.json"
    if feature_path.exists() and target_path.exists() and groups_path.exists() and not force:
        with open(groups_path, "r", encoding="utf-8") as f:
            groups = json.load(f)
        return pd.read_csv(feature_path), pd.read_csv(target_path), groups

    rows = []
    groups: Dict[str, List[str]] = {}
    for sensor in SENSORS:
        path = RAW_DIR / f"{sensor}.txt"
        matrix = np.loadtxt(path)
        feats = row_features(matrix)
        names = [f"{sensor}_{name}" for name in ["mean", "std", "rms", "min", "max", "ptp", "slope", "skew", "kurt", "energy", "crest"]]
        groups[sensor] = names
        rows.append(pd.DataFrame(feats, columns=names))
    features = pd.concat(rows, axis=1)

    profile = np.loadtxt(RAW_DIR / "profile.txt")
    targets = pd.DataFrame(profile, columns=["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure", "stable_flag"])
    targets["cooler_fault"] = (targets["cooler_condition"] < 100).astype(int)
    targets["valve_fault"] = (targets["valve_condition"] < 100).astype(int)
    targets["pump_fault"] = (targets["pump_leakage"] > 0).astype(int)
    targets["accumulator_fault"] = (targets["accumulator_pressure"] < 130).astype(int)
    targets["any_fault"] = targets[[f"{c}_fault" for c in COMPONENTS]].max(axis=1)

    features.to_csv(feature_path, index=False)
    targets.to_csv(target_path, index=False)
    with open(groups_path, "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)
    return features, targets, groups


def train_models(features: pd.DataFrame, targets: pd.DataFrame, seed: int) -> Dict[str, object]:
    y = targets[[f"{c}_fault" for c in COMPONENTS]].to_numpy(dtype=int)
    indices = np.arange(len(features))
    train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=seed, stratify=targets["any_fault"])
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features.iloc[train_idx].to_numpy(dtype=float))
    x_test = scaler.transform(features.iloc[test_idx].to_numpy(dtype=float))
    y_train = y[train_idx]
    y_test = y[test_idx]

    base = RandomForestClassifier(n_estimators=260, min_samples_leaf=2, random_state=seed, class_weight="balanced_subsample", n_jobs=-1)
    model = MultiOutputClassifier(base)
    model.fit(x_train, y_train)
    prob = np.vstack([positive_class_proba(est, x_test) for est in model.estimators_]).T
    pred = (prob >= 0.5).astype(int)
    return {
        "model": model,
        "scaler": scaler,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "prob": prob,
        "pred": pred,
    }


def group_indices(features: pd.DataFrame, groups: Dict[str, List[str]]) -> Dict[str, List[int]]:
    col_to_idx = {c: i for i, c in enumerate(features.columns)}
    return {sensor: [col_to_idx[c] for c in cols] for sensor, cols in groups.items()}


def positive_class_proba(estimator, x: np.ndarray) -> np.ndarray:
    """P(y=1) from a fitted binary classifier, robust to a training fold that saw
    only one class (predict_proba then has a single column). Never triggered by the
    original single 70/30 chronological split, which sees both classes for every
    component; needed once the chronological cut point moves earlier (Phase 1
    generalization curve), where a component can be single-class in the training
    fold."""
    proba = estimator.predict_proba(x)
    classes = list(estimator.classes_)
    if proba.shape[1] == 1:
        only_class = classes[0]
        return np.full(x.shape[0], 1.0 if only_class == 1 else 0.0)
    return proba[:, classes.index(1)]


def occlusion_scores(model: MultiOutputClassifier, x: np.ndarray, baseline: np.ndarray, gidx: Dict[str, List[int]]) -> np.ndarray:
    base_prob = np.vstack([positive_class_proba(est, x) for est in model.estimators_]).T
    scores = np.zeros((x.shape[0], len(gidx), len(COMPONENTS)), dtype=float)
    for s_idx, (_, idxs) in enumerate(gidx.items()):
        xp = x.copy()
        xp[:, idxs] = baseline[idxs]
        p = np.vstack([positive_class_proba(est, xp) for est in model.estimators_]).T
        scores[:, s_idx, :] = np.maximum(base_prob - p, 0.0)
    return scores


def permutation_importance(model: MultiOutputClassifier, x: np.ndarray, y: np.ndarray, gidx: Dict[str, List[int]], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_prob = np.vstack([positive_class_proba(est, x) for est in model.estimators_]).T
    base_ap = [average_precision_score(y[:, j], base_prob[:, j]) for j in range(len(COMPONENTS))]
    rows = []
    for sensor, idxs in gidx.items():
        xp = x.copy()
        for idx in idxs:
            xp[:, idx] = rng.permutation(xp[:, idx])
        p = np.vstack([positive_class_proba(est, xp) for est in model.estimators_]).T
        for j, component in enumerate(COMPONENTS):
            ap = average_precision_score(y[:, j], p[:, j])
            rows.append({"sensor": sensor, "component": component, "importance": max(0.0, base_ap[j] - ap)})
    return pd.DataFrame(rows)


def ranking_metric_row(name: str, y: np.ndarray, score: np.ndarray) -> Dict[str, float | int | str]:
    true_any = y.sum(axis=1) > 0
    top = np.argmax(score, axis=1)
    top_correct = []
    reciprocal_ranks = []
    for i in range(len(y)):
        if not true_any[i]:
            continue
        top_correct.append(y[i, top[i]] == 1)
        order = np.argsort(score[i])[::-1]
        true_positions = np.where(y[i, order] == 1)[0]
        reciprocal_ranks.append(1.0 / float(true_positions[0] + 1))
    if np.any(true_any):
        lrap_fault = label_ranking_average_precision_score(y[true_any], score[true_any])
    else:
        lrap_fault = math.nan
    return {
        "method": name,
        "fault_cycles": int(np.sum(true_any)),
        "label_ranking_ap_all_cycles": float(label_ranking_average_precision_score(y, score)),
        "label_ranking_ap_fault_cycles": float(lrap_fault),
        "top1_root_cause_accuracy": float(np.mean(top_correct)),
        "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
        "mean_priority_score": float(np.mean(np.max(score, axis=1))),
    }


def evaluate_rankings(y: np.ndarray, prob: np.ndarray, occlusion: np.ndarray, perm: pd.DataFrame, gidx: Dict[str, List[int]]) -> pd.DataFrame:
    true_any = y.sum(axis=1) > 0
    methods = {}
    methods["Probability only"] = prob
    methods["Severity-only policy"] = np.tile(SEVERITY, (len(y), 1))
    methods["Low-effort-only policy"] = np.tile(1.0 / EFFORT, (len(y), 1))
    methods["Severity/effort heuristic"] = np.tile(SEVERITY / EFFORT, (len(y), 1))
    perm_component = perm.groupby("component")["importance"].mean().reindex(COMPONENTS).fillna(0).to_numpy()
    methods["Permutation weighted"] = prob * (1.0 + perm_component)
    local_component = occlusion.mean(axis=1)
    methods["Explanation confidence only"] = local_component
    methods["Local occlusion"] = prob * (1.0 + local_component)
    confidence = np.maximum(local_component, 1e-6)
    methods["Priority score (prob + evidence + severity/effort)"] = prob * (1.0 + confidence) * (SEVERITY / EFFORT)

    return pd.DataFrame([ranking_metric_row(name, y, score) for name, score in methods.items()])


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    y_true = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-12, 1.0 - 1e-12)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    max_gap = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob > edges[i]) & (prob <= edges[i + 1])
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(y_true[mask])) - float(np.mean(prob[mask])))
        ece += float(np.mean(mask)) * gap
        max_gap = max(max_gap, gap)
    return ece, max_gap


def calibration_metrics(y: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows = []
    prob = np.clip(prob, 1e-12, 1.0 - 1e-12)
    for j, component in enumerate(COMPONENTS):
        ece, max_gap = expected_calibration_error(y[:, j], prob[:, j])
        rows.append(
            {
                "component": component,
                "positive_rate": float(np.mean(y[:, j])),
                "mean_probability": float(np.mean(prob[:, j])),
                "brier_score": float(brier_score_loss(y[:, j], prob[:, j])),
                "ece_10_bin": float(ece),
                "max_calibration_gap": float(max_gap),
                "negative_log_loss": float(log_loss(y[:, j], np.column_stack([1.0 - prob[:, j], prob[:, j]]), labels=[0, 1])),
            }
        )
    rows.append(
        {
            "component": "macro",
            "positive_rate": float(np.mean([r["positive_rate"] for r in rows])),
            "mean_probability": float(np.mean([r["mean_probability"] for r in rows])),
            "brier_score": float(np.mean([r["brier_score"] for r in rows])),
            "ece_10_bin": float(np.mean([r["ece_10_bin"] for r in rows])),
            "max_calibration_gap": float(np.mean([r["max_calibration_gap"] for r in rows])),
            "negative_log_loss": float(np.mean([r["negative_log_loss"] for r in rows])),
        }
    )
    return pd.DataFrame(rows)


def evaluate_priority_sensitivity(y: np.ndarray, prob: np.ndarray, occlusion: np.ndarray) -> pd.DataFrame:
    confidence = np.maximum(occlusion.mean(axis=1), 1e-6)
    scenarios = [
        ("Probability only", 0.0, 0.0, 0.0),
        ("Probability + evidence", 1.0, 0.0, 0.0),
        ("Current priority score", 1.0, 1.0, 1.0),
        ("Dampened effort penalty", 1.0, 1.0, 0.5),
        ("High severity emphasis", 1.0, 1.5, 1.0),
        ("High explanation emphasis", 2.0, 1.0, 1.0),
        ("Policy-only severity/effort", 0.0, 1.0, 1.0),
    ]
    rows = []
    for name, explanation_weight, severity_exponent, effort_exponent in scenarios:
        policy = np.power(SEVERITY, severity_exponent) / np.power(EFFORT, effort_exponent)
        if name == "Policy-only severity/effort":
            score = np.tile(policy, (len(y), 1))
        else:
            score = prob * (1.0 + explanation_weight * confidence) * policy
        row = ranking_metric_row(name, y, score)
        row.update(
            {
                "explanation_weight": explanation_weight,
                "severity_exponent": severity_exponent,
                "effort_exponent": effort_exponent,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def component_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, component in enumerate(COMPONENTS):
        rows.append(
            {
                "component": component,
                "accuracy": accuracy_score(y[:, j], pred[:, j]),
                "precision": precision_score(y[:, j], pred[:, j], zero_division=0),
                "recall": recall_score(y[:, j], pred[:, j], zero_division=0),
                "f1": f1_score(y[:, j], pred[:, j], zero_division=0),
                "average_precision": average_precision_score(y[:, j], prob[:, j]),
            }
        )
    rows.append(
        {
            "component": "macro",
            "accuracy": np.mean([r["accuracy"] for r in rows]),
            "precision": np.mean([r["precision"] for r in rows]),
            "recall": np.mean([r["recall"] for r in rows]),
            "f1": np.mean([r["f1"] for r in rows]),
            "average_precision": np.mean([r["average_precision"] for r in rows]),
        }
    )
    return pd.DataFrame(rows)


def plot_architecture() -> None:
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.axis("off")
    boxes = [
        (0.03, 0.58, 0.18, 0.22, "Multi-sensor\ncycle data"),
        (0.28, 0.58, 0.18, 0.22, "Failure\nclassifier"),
        (0.53, 0.58, 0.18, 0.22, "Occlusion and\npermutation XAI"),
        (0.76, 0.58, 0.20, 0.22, "Evidence-weighted\ndiagnostic graph"),
        (0.28, 0.18, 0.18, 0.22, "Component\nrisk scores"),
        (0.53, 0.18, 0.18, 0.22, "Priority\nscreening"),
        (0.76, 0.18, 0.20, 0.22, "Maintenance\naction list"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, ec="#222222", fc="#F2F5F7", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")
    for start, end in [
        ((0.21, 0.69), (0.28, 0.69)),
        ((0.46, 0.69), (0.53, 0.69)),
        ((0.71, 0.69), (0.76, 0.69)),
        ((0.37, 0.58), (0.37, 0.40)),
        ((0.46, 0.29), (0.53, 0.29)),
        ((0.71, 0.29), (0.76, 0.29)),
        ((0.86, 0.58), (0.86, 0.40)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_conceptual_architecture.png", bbox_inches="tight")
    plt.close(fig)


def save_figures(metrics: pd.DataFrame, rankings: pd.DataFrame, perm: pd.DataFrame, occlusion: np.ndarray, y: np.ndarray, prob: np.ndarray, gidx: Dict[str, List[int]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    configure_plot_style()
    plot_architecture()

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    m = metrics[metrics["component"] != "macro"]
    sns.barplot(data=m, x="component", y="f1", hue="component", palette="colorblind", legend=False, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Component")
    ax.set_ylabel("F1 score")
    ax.set_title("Component Fault Classification")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_component_f1.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    order = rankings.sort_values("top1_root_cause_accuracy", ascending=False)["method"]
    sns.barplot(data=rankings, x="top1_root_cause_accuracy", y="method", order=order, hue="method", palette="colorblind", legend=False, ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Top-1 root-cause accuracy")
    ax.set_ylabel("")
    ax.set_title("Root-Cause Ranking Comparison")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_root_cause_ranking.png", bbox_inches="tight")
    plt.close(fig)

    pivot = perm.pivot_table(index="sensor", columns="component", values="importance", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    sns.heatmap(pivot, cmap="rocket_r", annot=False, linewidths=0.3, ax=ax, cbar_kws={"label": "AP drop"})
    ax.set_title("Permutation Importance by Sensor and Component")
    ax.set_xlabel("Component")
    ax.set_ylabel("Sensor")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_importance_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    avg_edges = occlusion.mean(axis=0)
    graph = nx.DiGraph()
    for sensor in gidx:
        graph.add_node(sensor, bipartite=0)
    for component in COMPONENTS:
        graph.add_node(component, bipartite=1)
    for i, sensor in enumerate(gidx):
        for j, component in enumerate(COMPONENTS):
            weight = avg_edges[i, j]
            if weight > np.percentile(avg_edges, 75):
                graph.add_edge(sensor, component, weight=weight)
    pos = {}
    for idx, sensor in enumerate(gidx):
        pos[sensor] = (0, -idx)
    for idx, component in enumerate(COMPONENTS):
        pos[component] = (3, -idx * 4)
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    widths = [1.0 + 8.0 * graph[u][v]["weight"] for u, v in graph.edges()]
    nx.draw_networkx_nodes(graph, pos, nodelist=list(gidx.keys()), node_color="#9ecae1", node_size=700, ax=ax)
    nx.draw_networkx_nodes(graph, pos, nodelist=COMPONENTS, node_color="#fdae6b", node_size=1100, ax=ax)
    nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.75, arrows=True, arrowstyle="-|>", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    ax.set_title("Evidence-Weighted Sensor-to-Component Diagnostic Graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_evidence_graph.png", bbox_inches="tight")
    plt.close(fig)

    sample_idx = int(np.argmax(np.sum(prob * y, axis=1)))
    priority = prob[sample_idx] * (SEVERITY / EFFORT)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    order_idx = np.argsort(priority)[::-1]
    sns.barplot(x=priority[order_idx], y=np.array(COMPONENTS)[order_idx], hue=np.array(COMPONENTS)[order_idx], palette="colorblind", legend=False, ax=ax)
    ax.set_xlabel("Maintenance priority score")
    ax.set_ylabel("")
    ax.set_title("Example Maintenance Priority Screening")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_maintenance_priority_example.png", bbox_inches="tight")
    plt.close(fig)


def latex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def write_latex_tables(metrics: pd.DataFrame, rankings: pd.DataFrame, calibration: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    manuscript_dir = ROOT / "manuscript"
    if not manuscript_dir.exists():
        return

    component_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Component fault classification metrics on the random stratified test split. The split is intentionally reported as an optimistic reference; leakage-safe splits are reported separately.}",
        r"\label{tab:componentmetrics}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Component & Accuracy & Precision & Recall & F1 & AP \\",
        r"\hline",
    ]
    for _, r in metrics.iterrows():
        component = "Macro" if r["component"] == "macro" else str(r["component"]).capitalize()
        component_lines.append(
            f"{component} & {r['accuracy']:.3f} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} & {r['average_precision']:.3f} \\\\"
        )
    component_lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (manuscript_dir / "generated_component_metrics_table.tex").write_text("\n".join(component_lines), encoding="utf-8")

    ranking_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Component-fault ranking on fault-containing test cycles. Static maintenance policies are included to test whether the priority rule adds information beyond severity and effort constants.}",
        r"\label{tab:ranking}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{Xrrr}",
        r"\hline",
        r"Method & LRAP & Top-1 & MRR \\",
        r"\hline",
    ]
    for _, r in rankings.iterrows():
        ranking_lines.append(
            f"{latex_escape(r['method'])} & {r['label_ranking_ap_fault_cycles']:.3f} & {r['top1_root_cause_accuracy']:.3f} & {r['mean_reciprocal_rank']:.3f} & {r['mean_priority_score']:.3f} \\\\"
        )
    ranking_lines += [r"\hline", r"\end{tabularx}", r"\end{table}", ""]
    (manuscript_dir / "generated_ranking_baselines_table.tex").write_text("\n".join(ranking_lines), encoding="utf-8")

    calibration_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Probability calibration on the random stratified test split. Lower Brier score, ECE, and negative log loss indicate better calibrated probabilities for priority scoring.}",
        r"\label{tab:calibration}",
        r"\footnotesize",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Component & Pos. rate & Mean $p$ & Brier & ECE & NLL \\",
        r"\hline",
    ]
    for _, r in calibration.iterrows():
        component = "Macro" if r["component"] == "macro" else str(r["component"]).capitalize()
        calibration_lines.append(
            f"{component} & {r['positive_rate']:.3f} & {r['mean_probability']:.3f} & {r['brier_score']:.4f} & {r['ece_10_bin']:.4f} & {r['negative_log_loss']:.4f} \\\\"
        )
    calibration_lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    (manuscript_dir / "generated_calibration_table.tex").write_text("\n".join(calibration_lines), encoding="utf-8")

    sensitivity_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of maintenance-priority ranking to explanation, severity, and effort weights. The identical top-1 values for several probability-based variants show that this benchmark is too separable to prove superiority of a particular priority weighting.}",
        r"\label{tab:sensitivity}",
        r"\footnotesize",
        r"\begin{tabularx}{\textwidth}{Xrrrrr}",
        r"\hline",
        r"Scenario & $\gamma$ & $S$ exp. & $E$ exp. & Top-1 & MRR \\",
        r"\hline",
    ]
    for _, r in sensitivity.iterrows():
        sensitivity_lines.append(
            f"{latex_escape(r['method'])} & {r['explanation_weight']:.1f} & {r['severity_exponent']:.1f} & {r['effort_exponent']:.1f} & {r['top1_root_cause_accuracy']:.3f} & {r['mean_reciprocal_rank']:.3f} \\\\"
        )
    sensitivity_lines += [r"\hline", r"\end{tabularx}", r"\end{table}", ""]
    (manuscript_dir / "generated_priority_sensitivity_table.tex").write_text("\n".join(sensitivity_lines), encoding="utf-8")

    ranking_order = rankings.sort_values("label_ranking_ap_fault_cycles", ascending=False)["method"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.barplot(data=rankings, x="label_ranking_ap_fault_cycles", y="method", order=ranking_order, hue="method", palette="colorblind", legend=False, ax=ax)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Label ranking average precision")
    ax.set_ylabel("")
    ax.set_title("Maintenance Ranking Quality")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "paper3_ranking_average_precision.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    download_dataset()
    features, targets, groups = build_feature_table(force=args.force_features)
    bundle = train_models(features, targets, args.seed)
    gidx = group_indices(features, groups)
    baseline = np.mean(bundle["x_train"], axis=0)
    occ = occlusion_scores(bundle["model"], bundle["x_test"], baseline, gidx)
    perm = permutation_importance(bundle["model"], bundle["x_test"], bundle["y_test"], gidx, args.seed)
    metrics = component_metrics(bundle["y_test"], bundle["pred"], bundle["prob"])
    rankings = evaluate_rankings(bundle["y_test"], bundle["prob"], occ, perm, gidx)
    calibration = calibration_metrics(bundle["y_test"], bundle["prob"])
    sensitivity = evaluate_priority_sensitivity(bundle["y_test"], bundle["prob"], occ)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(PROC_DIR / "paper3_component_metrics.csv", index=False)
    rankings.to_csv(PROC_DIR / "paper3_ranking_metrics.csv", index=False)
    calibration.to_csv(PROC_DIR / "paper3_calibration_metrics.csv", index=False)
    sensitivity.to_csv(PROC_DIR / "paper3_priority_sensitivity.csv", index=False)
    perm.to_csv(PROC_DIR / "paper3_permutation_importance.csv", index=False)
    pd.DataFrame(bundle["prob"], columns=[f"{c}_probability" for c in COMPONENTS]).to_csv(PROC_DIR / "paper3_test_probabilities.csv", index=False)
    summary = {
        "dataset": "UCI Condition Monitoring of Hydraulic Systems",
        "instances": int(len(features)),
        "features": int(features.shape[1]),
        "test_instances": int(len(bundle["test_idx"])),
        "macro_f1": float(metrics.loc[metrics["component"] == "macro", "f1"].iat[0]),
        "macro_brier_score": float(calibration.loc[calibration["component"] == "macro", "brier_score"].iat[0]),
        "macro_ece_10_bin": float(calibration.loc[calibration["component"] == "macro", "ece_10_bin"].iat[0]),
        "best_ranking_method": rankings.sort_values("top1_root_cause_accuracy", ascending=False).iloc[0].to_dict(),
    }
    with open(PROC_DIR / "paper3_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    save_figures(metrics, rankings, perm, occ, bundle["y_test"], bundle["prob"], gidx)
    write_latex_tables(metrics, rankings, calibration, sensitivity)
    print(json.dumps(summary, indent=2))
    print(metrics.to_string(index=False))
    print(rankings.to_string(index=False))
    print(calibration.to_string(index=False))
    print(sensitivity.to_string(index=False))


if __name__ == "__main__":
    main()
