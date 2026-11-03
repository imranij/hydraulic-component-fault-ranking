"""Redraws the three schematic figures (Figs. 1-3) so that they match the released
implementation: the evidence graph is built from grouped occlusion only, permutation
importance is a separate global view, and the ranking audit quotes the leakage-safe split."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIG = Path(__file__).resolve().parent.parent / "figures"
NAVY, LIGHT, EDGE = "#1f3b63", "#eef2f8", "#1f3b63"


def band(ax, y, h, title, boxes):
    ax.add_patch(FancyBboxPatch((0.02, y), 0.96, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                fc="white", ec=EDGE, lw=0.7))
    ax.text(0.5, y + h - 0.018, title, ha="center", va="top", fontsize=6.6, fontweight="bold", color=NAVY)
    k = len(boxes)
    w = (0.92 - 0.03 * (k - 1)) / k
    for i, (head, body) in enumerate(boxes):
        x = 0.04 + i * (w + 0.03)
        by, bh = y + 0.012, h - 0.072
        ax.add_patch(FancyBboxPatch((x, by), w, bh, boxstyle="round,pad=0.003,rounding_size=0.008",
                                    fc=LIGHT, ec=EDGE, lw=0.5))
        ax.add_patch(FancyBboxPatch((x, by + bh - 0.03), w, 0.03, boxstyle="square,pad=0",
                                    fc=NAVY, ec=NAVY, lw=0))
        ax.text(x + w / 2, by + bh - 0.015, head, ha="center", va="center", fontsize=5.6, color="white",
                fontweight="bold")
        ax.text(x + w / 2, by + (bh - 0.03) / 2, body, ha="center", va="center", fontsize=5.2,
                color="black", wrap=True, linespacing=1.25)


def arrow(ax, y_top, y_bot, label):
    ax.annotate("", xy=(0.5, y_bot), xytext=(0.5, y_top),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6))
    ax.text(0.53, (y_top + y_bot) / 2, label, ha="left", va="center", fontsize=5.2, style="italic")


def draw(name, bands, labels):
    fig = plt.figure(figsize=(3.5, 0.95 * len(bands) + 0.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n = len(bands)
    gap = 0.045
    h = (1 - 0.02 - gap * (n - 1)) / n
    y = 1 - 0.01 - h
    prev = None
    for i, (title, boxes) in enumerate(bands):
        if prev is not None:
            arrow(ax, prev, y + h, labels[i - 1])
        band(ax, y, h, title, boxes)
        prev = y
        y -= h + gap
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(blk_prob=0.022, blk_prio=0.051, static_lo=0.265, static_hi=0.330):
    draw("paper3_conceptual_architecture", [
        ("UCI hydraulic test rig", [("2205 cycles", "60 s each,\n17 sensor channels"),
                                    ("187 features", "11 cycle-level\ndescriptors per channel")]),
        ("Multi-output classification", [("Random forest", "one binary forest\nper component"),
                                         ("Four probabilities", "cooler, valve,\npump, accumulator")]),
        ("Two explanation views, computed independently",
         [("Grouped occlusion", "local: drop in $p_c$ when a\ngroup is mean-imputed"),
          ("Permutation importance", "global: drop in AP when\na group is permuted")]),
        ("Diagnostic evidence graph (occlusion only)",
         [("Sensor families to components", "edge = mean occlusion drop;\nmodel sensitivity, not causation")]),
        ("Component-fault ranking",
         [("Priority score", "$p_c$, mean occlusion,\nfixed $S_c/E_c$"),
          ("Audited against", "probability-only and\nstatic policies")]),
    ], ["one row per cycle", "four probabilities", "occlusion drops", "a ranked list"])

    draw("paper3_explanation_evidence", [
        ("One cycle, one component", [("The classifier emits $p_c(x)$, Eq. (1)",
                                       "the explanations ask what that output\nis sensitive to, one sensor group at a time")]),
        ("Two views of the same fitted model",
         [("Local, per cycle", "mean-impute the group,\nre-query, take the drop, Eq. (3)"),
          ("Global, per test set", "permute the group,\ntake the AP drop, Eq. (4)")]),
        ("What each view is used for",
         [("Occlusion", "graph edges (Fig. 9) and\nthe $\\bar a_c$ term of Eq. (5)"),
          ("Permutation", "Fig. 10(b) and the permutation-\nweighted baseline only")]),
        ("What the graph does not claim",
         [("A model-sensitivity map", "mean imputation queries the model off\nthe data manifold; edges are ordinal\n"
                                      "evidence about the model, not physical\nsupport or causal effect")]),
    ], ["two attributions", "kept separate", "one graph"])

    draw("paper3_priority_rule", [
        ("Three inputs, per component",
         [("Probability", "$p_c(x)$ from the\nmulti-output classifier"),
          ("Explanation term", "$\\bar a_c(x)$, mean\nocclusion drop"),
          ("Policy constants", "$S_c$, $E_c$: illustrative,\nnot elicited")]),
        ("The score", [("$M_c = p_c\\,(1+\\bar a_c)\\,S_c/E_c$, Eq. (5)",
                        "kept simple so an operator can see\nwhich term produced an ordering")]),
        ("The audit, leakage-safe blocked split",
         [("Probability only", f"top-1 error {blk_prob:.3f}"),
          ("Priority score", f"top-1 error {blk_prio:.3f}"),
          ("Static policies", f"top-1 error\n{static_lo:.3f}-{static_hi:.3f}")]),
        ("What this benchmark settles",
         [("Reweighting reorders components", "and raises top-1 error here; the priority\nweighting is a display "
                                              "convention until\ncost data exist to evaluate it")]),
    ], ["one score per component", "scored on held-out runs", "reading the audit"])


if __name__ == "__main__":
    main()
