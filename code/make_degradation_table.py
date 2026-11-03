"""Render the degradation-walk summary (results/degradation_walk_summary.csv,
produced by degradation_walk.py) into manuscript/generated_degradation_walk_table.tex.

Kept as a separate, cheap script (rather than folded into degradation_walk.py's own
table writer) so the ~35-minute experiment does not need to be re-run every time the
table's formatting changes. Reads only the already-committed summary CSV; no numbers
are computed or altered here.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "degradation_walk_summary.csv"
OUT_TEX = ROOT / "manuscript" / "generated_degradation_walk_table.tex"

POLICY = "Priority score (prob + evidence + severity/effort)"

AXIS_LABELS = {
    "training_fraction": "Training fraction retained",
    "sensor_families_removed": "Sensor families removed (of 7)",
    "label_flip_rate": "Training label-flip rate",
    "cycle_window_retained": "Cycle window retained",
}
AXIS_ORDER = ["training_fraction", "sensor_families_removed", "label_flip_rate", "cycle_window_retained"]


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["policy"] == POLICY].copy()

    lines = []
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Degradation walk (priority-score policy, 10 seeds per level, "
        r"blocked-group split held fixed): mean top-1 root-cause accuracy, its "
        r"bootstrap union-bound top-1 error, mean cross-component AUC, and macro-F1, "
        r"by axis and level. Code: \texttt{code/degradation\_walk.py}; data: "
        r"\texttt{results/degradation\_walk\_summary.csv}.}"
    )
    lines.append(r"\label{tab:degradationwalk}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\hline")
    lines.append(r"Axis & Level & Top-1 & Bound err.\ & Cross-AUC & Macro-F1 \\")
    lines.append(r"\hline")
    for axis in AXIS_ORDER:
        sub = df[df["axis"] == axis].sort_values("level")
        for _, row in sub.iterrows():
            level = row["level"]
            level_str = f"{level:.2f}" if axis != "sensor_families_removed" else f"{int(level)}"
            lines.append(
                f"{AXIS_LABELS[axis]} & {level_str} & {row['top1_mean']:.3f} & "
                f"{row['bound_top1_error']:.3f} & {row['mean_cross_auc']:.3f} & "
                f"{row['macro_f1_mean']:.3f} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(f"[make_degradation_table] wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
