"""Redraws the data figures whose axis text used the term "root cause" (reviewer R1) and
adds the new protocols to the split-sensitivity figure (R4). Reads committed CSVs only."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                            "mathtext.fontset": "dejavuserif", "font.size": 7})
ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"
PRIO = "Priority score (prob + evidence + severity/effort)"
SHORT = {"Probability only": "Probability only", "Severity-only policy": "Severity only",
         "Low-effort-only policy": "Low effort only", "Severity/effort heuristic": "Severity/effort",
         "Explanation confidence only": "Explanation conf. only", "Local occlusion": "Local occlusion",
         PRIO: "Priority score"}
COL = {"Probability only": "#1f5a99", "Severity only": "#d95f02", "Low effort only": "#7570b3",
       "Severity/effort": "#e7298a", "Explanation conf. only": "#66a61e", "Local occlusion": "#a6761d",
       "Priority score": "#000000"}


def ranking_random():
    vals = [("Severity only", 0.697), ("Low effort only", 0.735), ("Severity/effort", 0.735),
            ("Explanation conf. only", 0.986), ("Probability only", 1.000), ("Permutation weighted", 1.000),
            ("Local occlusion", 1.000), ("Priority score", 1.000)]
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ys = np.arange(len(vals))
    ax.barh(ys, [v for _, v in vals], color=["#fdbf6f"] * 3 + ["#c6dbef"] + ["#6baed6"] * 4, ec="0.3", lw=0.4)
    for y_, (_, v) in zip(ys, vals):
        ax.text(v + 0.01, y_, f"{v:.3f}", va="center", fontsize=6)
    ax.set_yticks(ys); ax.set_yticklabels([k for k, _ in vals], fontsize=6.5)
    ax.set_xlim(0, 1.12); ax.set_xlabel("Top-1 component-fault ranking accuracy (random split)")
    ax.axvline(0.735, color="#b2182b", ls="--", lw=0.8)
    ax.text(0.74, len(vals) - 0.5, "best static policy", color="#b2182b", fontsize=5.5, va="bottom")
    fig.tight_layout(); fig.savefig(FIG / "paper3_root_cause_ranking.pdf", bbox_inches="tight"); plt.close(fig)


def ceiling():
    d = pd.read_csv(RES / "ceiling_bound_check_runclustered.csv")
    splits = ["random_stratified", "blocked_group_5fold", "temporal_chronological_0.70"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    pols = [p for p in d.quantity.unique() if p in SHORT]
    for si, sp in enumerate(splits):
        b = d[(d.split == sp) & (d.quantity == "bound")].iloc[0]
        ax.fill_between([si - 0.35, si + 0.35], b.clu_lo, b.clu_hi, color="0.85")
        ax.plot([si - 0.35, si + 0.35], [b.est] * 2, color="black", lw=1.6)
        for pi, p in enumerate(pols):
            r = d[(d.split == sp) & (d.quantity == p)].iloc[0]
            x = si - 0.27 + pi * 0.09
            ax.errorbar(x, r.est, yerr=[[r.est - r.clu_lo], [r.clu_hi - r.est]], fmt="o", ms=2.6,
                        color=COL[SHORT[p]], capsize=1.5, lw=0.7, label=SHORT[p] if si == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Random", "Blocked 5-fold", "Chronological 0.70"])
    ax.set_ylabel("Top-1 component-fault ranking error")
    ax.legend(fontsize=5, ncol=2, loc="upper left", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "ceiling_bound_vs_observed.pdf", bbox_inches="tight"); plt.close(fig)


def chrono():
    d = pd.read_csv(RES / "chronological_curve_runclustered.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3))
    po = d[d.policy == "Probability only"].sort_values("cut_fraction")
    axes[0].plot(po.cut_fraction, po.macro_f1, "o-", ms=3, label="Macro-F1 (all four)")
    axes[0].plot(po.cut_fraction, po.mean_cross_auc, "s-", ms=3, label="Mean cross-component AUC")
    axes[0].set_xlabel("Chronological cut fraction"); axes[0].legend(fontsize=6, frameon=False)
    for p, m in [("Probability only", "o"), (PRIO, "^"), ("Severity/effort heuristic", "s")]:
        s = d[d.policy == p].sort_values("cut_fraction")
        axes[1].errorbar(s.cut_fraction, s.top1_accuracy,
                         yerr=[s.top1_accuracy - s.top1_accuracy_ci_lo, s.top1_accuracy_ci_hi - s.top1_accuracy],
                         marker=m, ms=3, capsize=1.5, lw=0.8, label=SHORT[p], color=COL[SHORT[p]])
    axes[1].set_xlabel("Chronological cut fraction"); axes[1].set_ylabel("Top-1 component-fault accuracy")
    axes[1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "ceiling_chronological_curve.pdf", bbox_inches="tight"); plt.close(fig)


def degradation():
    d = pd.read_csv(RES / "degradation_walk_summary.csv")
    axes_ = [("training_fraction", "Training fraction retained"), ("sensor_families_removed", "Sensor families removed (of 7)"),
             ("label_flip_rate", "Training label-flip rate"), ("cycle_window_retained", "Cycle window retained")]
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.9), sharey=True)
    for ax, (a, lab) in zip(axs, axes_):
        for p in ["Probability only", PRIO, "Severity/effort heuristic"]:
            s = d[(d.axis == a) & (d.policy == p)].sort_values("level")
            ax.plot(s.level, s.top1_mean, "o-", ms=2.5, lw=0.8, color=COL[SHORT[p]], label=SHORT[p])
            ax.fill_between(s.level, s.top1_mean - s.top1_std, s.top1_mean + s.top1_std, color=COL[SHORT[p]], alpha=0.15)
        ax.set_xlabel(lab, fontsize=6); ax.tick_params(labelsize=6); ax.set_ylim(0.4, 1.02)
    axs[0].set_ylabel("Top-1 component-fault accuracy", fontsize=6)
    axs[0].legend(fontsize=5, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "degradation_walk.pdf", bbox_inches="tight"); plt.close(fig)


def split_sensitivity(protocols):
    """protocols: list of (label, macro_f1, sd)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    xs = np.arange(len(protocols))
    cols = ["white"] + ["#6baed6"] * (len(protocols) - 2) + ["#fdbf6f"]
    ax.bar(xs, [p[1] for p in protocols], yerr=[p[2] for p in protocols], color=cols, ec="0.3", lw=0.5,
           hatch=["//"] + [""] * (len(protocols) - 1), capsize=2)
    for x, p in zip(xs, protocols):
        ax.text(x, p[1] + 0.02, f"{p[1]:.3f}", ha="center", fontsize=6)
    ax.set_xticks(xs); ax.set_xticklabels([p[0] for p in protocols], fontsize=5.6)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Macro-F1 (all four components)")
    ax.set_title("(a) Macro-F1 by evaluation protocol", fontsize=7)
    sig = [0, 0.25, 0.5, 1.0, 2.0]; f1 = [0.667, 0.601, 0.532, 0.482, 0.447]
    axes[1].plot(sig, f1, "o-", ms=3, color="#1f5a99")
    axes[1].set_xlabel(r"Noise s.d. $\sigma$ (units of training-set s.d. per feature)", fontsize=6)
    axes[1].set_ylabel("Macro-F1, chronological split", fontsize=6)
    axes[1].set_title("(b) Feature-noise stress", fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "paper3_split_sensitivity.pdf", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    ranking_random(); ceiling(); chrono(); degradation()
