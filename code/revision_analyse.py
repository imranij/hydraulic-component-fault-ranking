"""Stage 2 of the revision analyses: reads results/revision/cache/*.npz only.
Writes results/revision/*.csv, manuscript/generated_rev_*.tex and figures/rev_*.pdf."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import brier_score_loss, f1_score, log_loss

from revision_analyses import (CACHE, OUT, FIG, MAN, COMP, SEV, EFF, SEEDS10, SEEDS3, PRIO, FAMILIES,
                               CHANNEL_INFO, GRID, N_BOOT, cached, load, split_A, split_B, split_C,
                               weighted_score, ca, p3)

RNG_SEED = 20260912
CNAME = [c.capitalize() for c in COMP]
POL = ["Probability only", "Local occlusion", PRIO, "Severity/effort heuristic"]
POL_SHORT = {"Probability only": "Probability only", "Local occlusion": "Local occlusion",
             PRIO: "Priority score", "Severity/effort heuristic": "Severity/effort"}


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def ci_str(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------------------
# bound helpers
# ---------------------------------------------------------------------------

def conservative_bound(y, s):
    """Union bound with ties counted as full misranks: mean_x sum_{c in N(x)} 1[s_c >= max_P s]."""
    fc = y.sum(axis=1) > 0
    yy, ss = y[fc], s[fc]
    best = np.where(yy == 1, ss, -np.inf).max(axis=1)
    beats = (ss >= best[:, None]) & (yy == 0)
    return float(beats.sum(axis=1).mean())


def prob_bound(y, p):
    return ca.mean_union_bound(y, ca.cross_component_auc(y, p))


def order_stats(y, p, s):
    """Pairwise order preservation of score s relative to probability p, on fault-containing cycles."""
    fc = y.sum(axis=1) > 0
    p, s = p[fc], s[fc]
    full = np.ones(len(p), bool)
    for i in range(4):
        for j in range(i + 1, 4):
            full &= np.sign(p[:, i] - p[:, j]) == np.sign(s[:, i] - s[:, j])
    top_same = np.argmax(p, 1) == np.argmax(s, 1)
    return float(full.mean()), float(top_same.mean())


def reversal_frac(y, p, s):
    """Share of fault-containing cycles with at least one order violation: p_i > p_j but s_i <= s_j."""
    fc = y.sum(axis=1) > 0
    p, s = p[fc], s[fc]
    rev = np.zeros(len(p), bool)
    for i in range(4):
        for j in range(4):
            if i != j:
                rev |= (p[:, i] > p[:, j]) & (s[:, i] <= s[:, j])
    return float(rev.mean())


def tie_stats(y, p):
    fc = y.sum(axis=1) > 0
    yy, pp = y[fc], p[fc]
    mx = pp.max(axis=1, keepdims=True)
    tied = (pp == mx).sum(axis=1) > 1
    mixed = tied & (((pp == mx) & (yy == 1)).any(1)) & (((pp == mx) & (yy == 0)).any(1))
    return int(fc.sum()), int(tied.sum()), int(mixed.sum())


def clustered_boot(cl, stat_fn, n_boot=N_BOOT, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(cl == u) for u in np.unique(cl)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        ix = np.concatenate([members[j] for j in pick])
        v = stat_fn(ix)
        if v is not None:
            out.append(v)
    out = np.array(out, float)
    return np.nanpercentile(out, [2.5, 97.5], axis=0)


# ---------------------------------------------------------------------------

def main():
    f, t, g, y, run, prof = load()
    n = len(y)
    A, blocksA = split_A(n)
    B, blocksB, purged = split_B(n, run)
    C = split_C(n, prof)
    cut = int(0.70 * n)
    res = {}

    # ---------------- R2 label mapping -----------------------------------
    rows = []
    spec = [("Cooler", "cooler_condition", "efficiency (%)", lambda v: v < 100),
            ("Valve", "valve_condition", "switching condition (%)", lambda v: v < 100),
            ("Pump", "pump_leakage", "internal leakage level", lambda v: v > 0),
            ("Accumulator", "accumulator_pressure", "pressure (bar)", lambda v: v < 130)]
    for comp, col, desc, rule in spec:
        for v, cnt in sorted(t[col].value_counts().items(), key=lambda kv: -kv[0]):
            rows.append({"component": comp, "quantity": desc, "level": int(v), "cycles": int(cnt),
                         "binary_label": int(rule(v))})
    lab = pd.DataFrame(rows)
    lab.to_csv(OUT / "label_mapping.csv", index=False)
    stab = t.stable_flag.value_counts().to_dict()
    res["stable_flag_counts"] = {int(k): int(v) for k, v in stab.items()}
    res["n_fault_counts"] = {int(k): int(v) for k, v in pd.Series(y.sum(1)).value_counts().sort_index().items()}

    # ---------------- split diagnostics (R3, R4) -------------------------
    def split_diag(folds, name):
        shared_prof, shared_run, adj = [], [], []
        for tr, te in folds:
            trp, trr = set(prof[tr]), set(run[tr])
            shared_prof.append(float(np.mean([p_ in trp for p_ in prof[te]])))
            shared_run.append(float(np.mean([r_ in trr for r_ in run[te]])))
            trs = set(tr)
            adj.append(int(sum(((i - 1) in trs) or ((i + 1) in trs) for i in te)))
        return {"split": name, "test_cycles_with_profile_in_train": float(np.mean(shared_prof)),
                "test_cycles_with_run_in_train": float(np.mean(shared_run)),
                "test_cycles_adjacent_to_train_cycle_total": int(np.sum(adj))}
    diag = pd.DataFrame([split_diag(A, "A"), split_diag(B, "B"), split_diag(C, "C"),
                         split_diag([(np.arange(cut), np.arange(cut, n))], "T")])
    diag.to_csv(OUT / "split_diagnostics.csv", index=False)
    res["runs_split_across_A_blocks"] = int(sum(len(set(blocksA[run == r])) > 1 for r in np.unique(run)))
    res["B_purged_per_fold"] = purged
    res["A_fold_test_blocks"] = [sorted(set(int(b) for b in blocksA[te])) for _, te in A]

    # ---------------- R13 reconciliation ---------------------------------
    rows = []
    te0 = A[0][1]
    for s in SEEDS10:
        d = cached(f"A_rf_{s}")
        P, O = d["P"], d["O"]
        sc = ca.policy_scores(P, O)
        rows.append({"seed": s,
                     "fold0_prob_top1_acc": 1 - ca.top1_error(y[te0], sc["Probability only"][te0]),
                     "fold0_priority_top1_acc": 1 - ca.top1_error(y[te0], sc[PRIO][te0]),
                     "fold0_macro_f1": float(d["fold_f1"][0].mean()),
                     "pooled_prob_top1_err": ca.top1_error(y, sc["Probability only"]),
                     "pooled_priority_top1_err": ca.top1_error(y, sc[PRIO]),
                     "pooled_bound": prob_bound(y, P),
                     "foldmean_macro_f1": float(d["fold_f1"].mean())})
    r13 = pd.DataFrame(rows)
    r13.to_csv(OUT / "r13_seed_reconciliation.csv", index=False)
    res["r13_means"] = r13.drop(columns="seed").mean().to_dict()

    # ---------------- per-split classification (R3, R4, R5) -------------
    def cls_summary(d, folds):
        P = d["P"]
        mask = ~np.isnan(P[:, 0])
        yy, pp = y[mask], P[mask]
        pred = (pp >= 0.5).astype(int)
        comp_rows = []
        for j in range(4):
            npos, nneg = int(yy[:, j].sum()), int((1 - yy[:, j]).sum())
            f1 = f1_score(yy[:, j], pred[:, j]) if npos > 0 else np.nan
            comp_rows.append((npos, nneg, f1))
        fold_f1 = d["fold_f1"]
        # fold-mean macro-F1 over components defined in that fold
        fm = []
        for k, (tr, te) in enumerate(folds):
            defined = [j for j in range(4) if y[te, j].sum() > 0]
            fm.append(np.mean([fold_f1[k, j] for j in defined]))
        return comp_rows, float(np.mean(fold_f1)), float(np.mean(fm))
    split_rows = []
    T = [(np.arange(cut), np.arange(cut, n))]
    for key, folds, label in [("A", A, "Blocked (original, 10 equal blocks)"),
                              ("B", B, "Blocked, run-aligned + purged"),
                              ("C", C, "Condition-profile holdout"),
                              ("T", T, "Chronological 70/30")]:
        seeds = SEEDS3 if key in "BC" else ([17, 2026] if key == "A" else [2026])
        mf_all, mf_def = [], []
        for s in seeds:
            d = cached(f"{key}_rf_{s}")
            comp_rows, mf, mfd = cls_summary(d, folds)
            mf_all.append(mf)
            mf_def.append(mfd)
            if s == 2026:
                comp26 = comp_rows
        split_rows.append({"key": key, "split": label, "seeds": len(seeds),
                           "macro_f1_all4": float(np.mean(mf_all)), "macro_f1_all4_sd": float(np.std(mf_all)),
                           "macro_f1_defined": float(np.mean(mf_def)),
                           **{f"{COMP[j]}_npos": comp26[j][0] for j in range(4)},
                           **{f"{COMP[j]}_nneg": comp26[j][1] for j in range(4)},
                           **{f"{COMP[j]}_f1": comp26[j][2] for j in range(4)}})
    spl = pd.DataFrame(split_rows)
    spl.to_csv(OUT / "split_classification.csv", index=False)

    # ---------------- ranking, bound, order preservation (R4, R6) --------
    rank_rows, order_rows = [], []
    paired = {}
    for key, label in [("A", "Blocked (original)"), ("B", "Blocked, run-aligned + purged"),
                       ("C", "Condition-profile holdout"), ("T", "Chronological 70/30")]:
        d = cached(f"{key}_rf_2026")
        mask = ~np.isnan(d["P"][:, 0])
        yy, P, O, cl = y[mask], d["P"][mask], d["O"][mask], run[mask]
        sc = ca.policy_scores(P, O)
        bnd = prob_bound(yy, P)
        bnd_c = conservative_bound(yy, P)
        nfc, ntied, nmixed = tie_stats(yy, P)

        def st(ix, yy=yy, P=P, sc=sc):
            yb = yy[ix]
            if (yb.sum(1) > 0).sum() == 0:
                return None
            b = prob_bound(yb, P[ix])
            e = {k: ca.top1_error(yb, sc[k][ix]) for k in POL}
            return [e[PRIO] - e["Probability only"], e["Probability only"] - b, e["Local occlusion"] - b,
                    e[PRIO] - b, b] + [e[k] for k in POL]
        lo, hi = clustered_boot(cl, st)
        paired[key] = (lo, hi)
        for i, k in enumerate(POL):
            e = ca.top1_error(yy, sc[k])
            full, top = order_stats(yy, P, sc[k])
            rank_rows.append({"split_key": key, "split": label, "policy": POL_SHORT[k],
                              "top1_error": e, "err_ci_lo": lo[5 + i], "err_ci_hi": hi[5 + i],
                              "bound_prob": bnd, "bound_ci_lo": lo[4], "bound_ci_hi": hi[4],
                              "bound_prob_conservative": bnd_c, "bound_own_score": conservative_bound(yy, sc[k]),
                              "order_preserving_frac": full, "top1_agrees_with_prob_frac": top,
                              "strict_reversal_frac": reversal_frac(yy, P, sc[k])})
        order_rows.append({"split_key": key, "fault_cycles": nfc, "cycles_with_tied_top_prob": ntied,
                           "tied_top_mixing_pos_neg": nmixed,
                           "priority_minus_prob": ca.top1_error(yy, sc[PRIO]) - ca.top1_error(yy, sc["Probability only"]),
                           "priority_minus_prob_ci_lo": lo[0], "priority_minus_prob_ci_hi": hi[0],
                           "probonly_minus_bound": ca.top1_error(yy, sc["Probability only"]) - bnd,
                           "probonly_minus_bound_ci_lo": lo[1], "probonly_minus_bound_ci_hi": hi[1],
                           "localocc_minus_bound": ca.top1_error(yy, sc["Local occlusion"]) - bnd,
                           "localocc_minus_bound_ci_lo": lo[2], "localocc_minus_bound_ci_hi": hi[2],
                           "priority_minus_bound": ca.top1_error(yy, sc[PRIO]) - bnd,
                           "priority_minus_bound_ci_lo": lo[3], "priority_minus_bound_ci_hi": hi[3]})
    rk = pd.DataFrame(rank_rows)
    rk.to_csv(OUT / "ranking_by_split.csv", index=False)
    od = pd.DataFrame(order_rows)
    od.to_csv(OUT / "paired_and_bound_differences.csv", index=False)

    # single-fault subset (R1): cycles with exactly one faulted component
    sf = []
    for key in ["A", "B", "C"]:
        d = cached(f"{key}_rf_2026")
        sc = ca.policy_scores(d["P"], d["O"])
        m1 = y.sum(1) == 1
        for k in POL:
            sf.append({"split_key": key, "policy": POL_SHORT[k], "n_cycles": int(m1.sum()),
                       "top1_error": ca.top1_error(y[m1], sc[k][m1])})
    pd.DataFrame(sf).to_csv(OUT / "single_fault_subset.csv", index=False)
    # stable-flag subsets (R2): evaluation restricted by the fifth profile column (models unchanged)
    st_rows = []
    for key in ["A", "B", "C"]:
        d = cached(f"{key}_rf_2026")
        sc = ca.policy_scores(d["P"], d["O"])
        for flag in (0, 1):
            m_ = (t.stable_flag.to_numpy() == flag)
            pred_ = (d["P"][m_] >= 0.5).astype(int)
            st_rows.append({"split_key": key, "stable_flag": flag, "n_cycles": int(m_.sum()),
                            "macro_f1": float(np.mean([f1_score(y[m_, j], pred_[:, j], zero_division=0) for j in range(4)])),
                            "prob_top1_error": ca.top1_error(y[m_], sc["Probability only"][m_]),
                            "priority_top1_error": ca.top1_error(y[m_], sc[PRIO][m_])})
    pd.DataFrame(st_rows).to_csv(OUT / "stable_flag_subsets.csv", index=False)

    # order preservation across the 10 seeds on the degradation-walk fold (fold 0), for Sec. VI-I
    op = []
    for s in SEEDS10:
        d = cached(f"A_rf_{s}")
        sc = ca.policy_scores(d["P"][te0], d["O"][te0])
        for k in ["Local occlusion", PRIO]:
            full, top = order_stats(y[te0], d["P"][te0], sc[k])
            op.append({"seed": s, "policy": POL_SHORT[k], "order_preserving_frac": full,
                       "top1_agrees_frac": top, "strict_reversal_frac": reversal_frac(y[te0], d["P"][te0], sc[k])})
    pd.DataFrame(op).to_csv(OUT / "order_preservation_fold0_10seeds.csv", index=False)
    _g = pd.DataFrame(op).groupby("policy")[["order_preserving_frac", "top1_agrees_frac", "strict_reversal_frac"]].agg(["min", "mean"])
    res["fold0_order_pres"] = {p_: {f"{a}_{b}": float(_g.loc[p_, (a, b)]) for a, b in _g.columns} for p_ in _g.index}

    # ---------------- R10 calibration on leakage-safe OOF ----------------
    cal_rows = []
    for key in ["A", "B", "C"]:
        d = cached(f"{key}_rf_2026")
        P = np.clip(d["P"], 1e-12, 1 - 1e-12)
        for j in range(4):
            yj, pj = y[:, j], P[:, j]
            ece = p3.expected_calibration_error(yj, pj)[0]
            lo, hi = clustered_boot(run, lambda ix: [brier_score_loss(yj[ix], pj[ix]),
                                                     p3.expected_calibration_error(yj[ix], pj[ix])[0]],
                                    n_boot=1000)
            cal_rows.append({"split_key": key, "component": CNAME[j], "pos_rate": yj.mean(), "mean_p": pj.mean(),
                             "brier": brier_score_loss(yj, pj), "brier_lo": lo[0], "brier_hi": hi[0],
                             "ece": ece, "ece_lo": lo[1], "ece_hi": hi[1],
                             "nll": log_loss(yj, np.c_[1 - pj, pj], labels=[0, 1])})
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(OUT / "calibration_oof.csv", index=False)
    # reliability diagram, split A
    d = cached("A_rf_2026")
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.1), sharey=True)
    edges = np.linspace(0, 1, 11)
    for j, ax in enumerate(axes):
        pj, yj = d["P"][:, j], y[:, j]
        b = np.clip(np.digitize(pj, edges[1:-1]), 0, 9)
        xs, ys, ns = [], [], []
        for k in range(10):
            m = b == k
            if m.sum():
                xs.append(pj[m].mean()); ys.append(yj[m].mean()); ns.append(m.sum())
        ax.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
        ax.scatter(xs, ys, s=np.clip(np.array(ns) / 6, 6, 80), color="#1f5a99", zorder=3)
        ax.plot(xs, ys, color="#1f5a99", lw=0.9)
        r = cal[(cal.split_key == "A") & (cal.component == CNAME[j])].iloc[0]
        ax.set_title(f"{CNAME[j]}\nECE {r.ece:.3f}", fontsize=7)
        ax.set_xlabel("Mean predicted $p_c$", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Observed fault rate", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "rev_reliability_blocked.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------- R11 weight sensitivity on blocked OOF --------------
    d = cached("A_rf_2026")
    P, O = d["P"], d["O"]
    grid_rows = []
    for w in GRID:
        grid_rows.append({"gamma": w[0], "s_exp": w[1], "e_exp": w[2],
                          "top1_error": ca.top1_error(y, weighted_score(P, O, *w))})
    for a, b in [(1.0, 1.0)]:
        grid_rows.append({"gamma": np.nan, "s_exp": a, "e_exp": b,
                          "top1_error": ca.top1_error(y, np.tile(SEV ** a / EFF ** b, (n, 1)))})
    gr = pd.DataFrame(grid_rows)
    gr.to_csv(OUT / "weight_grid_blocked.csv", index=False)
    nested = cached("nested_A_2026")["score"]
    choices = pd.read_csv(OUT / "nested_weight_choices.csv")
    po = P
    e_nest = ca.top1_error(y, nested)
    lo, hi = clustered_boot(run, lambda ix: [ca.top1_error(y[ix], nested[ix]) - ca.top1_error(y[ix], po[ix])])
    fo = d["fold_of"]
    res["nested_per_fold"] = [{"fold": k, "nested_err": ca.top1_error(y[fo == k], nested[fo == k]),
                               "prob_err": ca.top1_error(y[fo == k], po[fo == k])} for k in range(5)]
    res["nested"] = {"error": e_nest, "minus_prob": e_nest - ca.top1_error(y, po), "ci": [lo[0], hi[0]],
                     "choices": choices.to_dict("records")}
    # paired CIs for the seven original scenarios
    scen = [("Probability only", 0.0, 0.0, 0.0), ("Probability + evidence", 1.0, 0.0, 0.0),
            ("Current priority score", 1.0, 1.0, 1.0), ("Dampened effort penalty", 1.0, 1.0, 0.5),
            ("High severity emphasis", 1.0, 1.5, 1.0), ("High explanation emphasis", 2.0, 1.0, 1.0),
            ("Probability x S/E (Fig. 11 rule)", 0.0, 1.0, 1.0)]
    srows = []
    for name, gam, a, b in scen:
        s_ = weighted_score(P, O, gam, a, b)
        e = ca.top1_error(y, s_)
        lo, hi = clustered_boot(run, lambda ix, s_=s_: [ca.top1_error(y[ix], s_[ix]) - ca.top1_error(y[ix], po[ix])],
                                n_boot=1000)
        srows.append({"scenario": name, "gamma": gam, "s_exp": a, "e_exp": b, "top1_error": e,
                      "minus_prob": e - ca.top1_error(y, po), "lo": lo[0], "hi": hi[0]})
    static = ca.top1_error(y, np.tile(SEV / EFF, (n, 1)))
    srows.append({"scenario": "Policy-only severity/effort", "gamma": np.nan, "s_exp": 1, "e_exp": 1,
                  "top1_error": static, "minus_prob": static - ca.top1_error(y, po), "lo": np.nan, "hi": np.nan})
    sens = pd.DataFrame(srows)
    sens.to_csv(OUT / "priority_sensitivity_blocked.csv", index=False)
    res["grid_min"] = gr.top1_error.min()
    res["grid_n_better_than_prob"] = int((gr.top1_error < ca.top1_error(y, po) - 1e-12).sum())
    res["grid_best"] = gr.loc[gr.top1_error.idxmin()].to_dict()

    # ---------------- R14 HistGradientBoosting --------------------------
    hrows = []
    for s in SEEDS3:
        dh = cached(f"A_hgb_{s}")
        sc = ca.policy_scores(dh["P"], dh["O"])
        hrows.append({"seed": s, "macro_f1_foldmean": float(dh["fold_f1"].mean()),
                      **{POL_SHORT[k]: ca.top1_error(y, sc[k]) for k in POL},
                      "bound": prob_bound(y, dh["P"])})
    hgb = pd.DataFrame(hrows)
    dh = cached("A_hgb_2026")
    sch = ca.policy_scores(dh["P"], dh["O"])
    lo, hi = clustered_boot(run, lambda ix: [ca.top1_error(y[ix], sch[PRIO][ix]) - ca.top1_error(y[ix], sch["Probability only"][ix])])
    res["hgb_paired"] = {"diff": ca.top1_error(y, sch[PRIO]) - ca.top1_error(y, sch["Probability only"]), "ci": [lo[0], hi[0]]}
    hgb.to_csv(OUT / "hgb_ranking_blocked.csv", index=False)
    drf = cached("A_rf_2026")
    w = np.array([len(te) for _, te in A], float)
    rf_occ_fam = np.average(drf["occ_fam"], axis=0, weights=w)
    hg_occ_fam = np.average(dh["occ_fam"], axis=0, weights=w)
    rf_occ_ch = np.average(drf["occ_ch"], axis=0, weights=w)
    hg_occ_ch = np.average(dh["occ_ch"], axis=0, weights=w)
    res["hgb_vs_rf_occ_spearman_channel"] = [float(spearmanr(rf_occ_ch[:, j], hg_occ_ch[:, j])[0]) for j in range(4)]
    res["hgb_vs_rf_occ_spearman_family"] = [float(spearmanr(rf_occ_fam[:, j], hg_occ_fam[:, j])[0]) for j in range(4)]
    res["hgb_top_family"] = [list(FAMILIES)[int(np.argmax(hg_occ_fam[:, j]))] for j in range(4)]
    res["rf_top_family"] = [list(FAMILIES)[int(np.argmax(rf_occ_fam[:, j]))] for j in range(4)]

    # ---------------- R9 explanation stability --------------------------
    mats = {"occ_ch": [], "perm_ch": [], "occ_fam": [], "perm_fam": []}
    for s in [17, 2026]:
        d_ = cached(f"A_rf_{s}")
        for k in mats:
            mats[k] += list(d_[k])
    stab_rows = []
    for k, L in mats.items():
        for j in range(4):
            rhos = [spearmanr(L[a][:, j], L[b][:, j])[0] for a in range(len(L)) for b in range(a + 1, len(L))]
            rhos = np.array(rhos, float)
            stab_rows.append({"quantity": k, "component": CNAME[j], "n_fits": len(L),
                              "median_rho": float(np.nanmedian(rhos)), "min_rho": float(np.nanmin(rhos)),
                              "frac_nan": float(np.isnan(rhos).mean())})
    shap_ch = cached("shap_A_2026")["shap_ch"]
    for j in range(4):
        ro = [spearmanr(drf["occ_ch"][k][:, j], shap_ch[k][:, j])[0] for k in range(5)]
        rp = [spearmanr(drf["perm_ch"][k][:, j], shap_ch[k][:, j])[0] for k in range(5)]
        rop = [spearmanr(drf["occ_ch"][k][:, j], drf["perm_ch"][k][:, j])[0] for k in range(5)]
        ss = [spearmanr(shap_ch[a][:, j], shap_ch[b][:, j])[0] for a in range(5) for b in range(a + 1, 5)]
        stab_rows += [{"quantity": "occ_vs_shap_perfold", "component": CNAME[j], "n_fits": 5,
                       "median_rho": float(np.nanmedian(ro)), "min_rho": float(np.nanmin(ro)), "frac_nan": 0.0},
                      {"quantity": "perm_vs_shap_perfold", "component": CNAME[j], "n_fits": 5,
                       "median_rho": float(np.nanmedian(rp)), "min_rho": float(np.nanmin(rp)),
                       "frac_nan": float(np.isnan(rp).mean())},
                      {"quantity": "occ_vs_perm_perfold", "component": CNAME[j], "n_fits": 5,
                       "median_rho": float(np.nanmedian(rop)), "min_rho": float(np.nanmin(rop)),
                       "frac_nan": float(np.isnan(rop).mean())},
                      {"quantity": "shap_ch", "component": CNAME[j], "n_fits": 5,
                       "median_rho": float(np.nanmedian(ss)), "min_rho": float(np.nanmin(ss)), "frac_nan": 0.0}]
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT / "explanation_stability.csv", index=False)
    # family-level top-family agreement across fits
    topfam = []
    for j in range(4):
        tops = [int(np.argmax(m[:, j])) for m in mats["occ_fam"]]
        topfam.append({"component": CNAME[j], "modal_top_family": list(FAMILIES)[max(set(tops), key=tops.count)],
                       "agree_frac": tops.count(max(set(tops), key=tops.count)) / len(tops)})
    pd.DataFrame(topfam).to_csv(OUT / "occlusion_top_family_agreement.csv", index=False)

    # ---------------- R8 family-level figures ----------------------------
    rf_perm_fam = np.average(drf["perm_fam"], axis=0, weights=w)
    fams = list(FAMILIES)
    pd.DataFrame(rf_occ_fam, index=fams, columns=CNAME).to_csv(OUT / "family_occlusion_blocked.csv")
    pd.DataFrame(rf_perm_fam, index=fams, columns=CNAME).to_csv(OUT / "family_permutation_blocked.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9))
    for ax, M, title, lab_ in [(axes[0], rf_occ_fam, "(a) Grouped occlusion", "mean drop in $p_c$"),
                               (axes[1], rf_perm_fam, "(b) Grouped permutation", "drop in AP$_c$")]:
        im = ax.imshow(M, cmap="Blues", aspect="auto", vmin=0, vmax=max(M.max(), 1e-9))
        ax.set_xticks(range(4)); ax.set_xticklabels(CNAME, fontsize=7)
        ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=7)
        for i in range(M.shape[0]):
            for j in range(4):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if M[i, j] > 0.6 * M.max() else "black")
        ax.set_title(title, fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6); cb.set_label(lab_, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "rev_family_importance.pdf", bbox_inches="tight")
    plt.close(fig)
    # evidence graph from family occlusion
    thr = 0.01
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ys_f = np.linspace(1, 0, len(fams)); ys_c = np.linspace(0.9, 0.1, 4)
    for i in range(len(fams)):
        for j in range(4):
            v = rf_occ_fam[i, j]
            if v >= thr:
                ax.plot([0.08, 0.9], [ys_f[i], ys_c[j]], color="#1f5a99", lw=0.6 + 25 * v, alpha=0.75,
                        solid_capstyle="round", zorder=1)
    for i, fam in enumerate(fams):
        ax.text(0.06, ys_f[i], fam, ha="right", va="center", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="#e8eef6", ec="#1f5a99", lw=0.6))
    for j, c in enumerate(CNAME):
        ax.text(0.92, ys_c[j], c, ha="left", va="center", fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fdebd3", ec="#b5651d", lw=0.6))
    ax.set_xlim(-0.55, 1.35); ax.set_ylim(-0.08, 1.08); ax.axis("off")
    fig.savefig(FIG / "rev_evidence_graph.pdf", bbox_inches="tight")
    plt.close(fig)
    res["graph_threshold"] = thr
    res["graph_n_edges"] = int((rf_occ_fam >= thr).sum())

    # ---------------- R12 label-free example ----------------------------
    d = cached("A_rf_2026")
    P, O = d["P"], d["O"]
    k2 = np.flatnonzero(((P >= 0.5).sum(1) == 2))
    ex = int(k2[0])
    M = P[ex] * (1 + np.maximum(O[ex], 1e-6)) * SEV / EFF
    order = np.argsort(M)[::-1]
    res["example"] = {"cycle": ex, "fold": int(d["fold_of"][ex]), "p": P[ex].tolist(), "abar": O[ex].tolist(),
                      "M": M.tolist(), "labels": y[ex].tolist(), "order": [COMP[i] for i in order],
                      "prob_order": [COMP[i] for i in np.argsort(P[ex])[::-1]], "n_candidates": int(len(k2))}
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    cols = ["#1f5a99" if y[ex, i] else "#bdbdbd" for i in order]
    ax.barh(range(4), M[order][::-1], color=cols[::-1])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"{CNAME[i]} ({'fault' if y[ex, i] else 'healthy'})" for i in order][::-1], fontsize=7)
    for r_, i in enumerate(order[::-1]):
        ax.text(M[i] + 0.03, r_, f"$M_c$={M[i]:.3f}  $p_c$={P[ex, i]:.3f}", va="center", fontsize=6)
    ax.set_xlim(0, max(M) * 1.9)
    ax.set_xlabel("Priority score $M_c(x)$, Eq. (5)", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "rev_priority_example.pdf", bbox_inches="tight")
    plt.close(fig)

    json.dump(res, open(OUT / "revision_summary.json", "w"), indent=2, default=float)
    write_tables(lab, spl, rk, od, cal, sens, res, hgb, stab, diag, r13)
    print(json.dumps(res, indent=1, default=float)[:6000])


def fit_width(path):
    """Wrap the tabular of a generated table in \\resizebox so it never exceeds the column/text width."""
    t = path.read_text(encoding="utf-8")
    if "\\resizebox" in t:
        return
    i, j = t.index("\\begin{tabular}"), t.index("\\end{tabular}") + len("\\end{tabular}")
    path.write_text(t[:i] + "\\resizebox{\\linewidth}{!}{%\n" + t[i:j] + "}" + t[j:], encoding="utf-8")


def write_tables(lab, spl, rk, od, cal, sens, res, hgb, stab, diag, r13):
    # label mapping + channel mapping
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Binary component-fault labels derived from the four condition columns of \texttt{profile.txt}. "
         r"Each level is listed with its cycle count; the nominal level is the only one mapped to 0. The fifth "
         r"column (stable flag: 0 for 1449 cycles, 1 for 756) is retained in every split, is not used to "
         r"define labels or features, and is constant within every condition-run, so the 194 runs are the same "
         r"with or without it.}",
         r"\label{tab:labelmap}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{llrrr}", r"\toprule", r"Component & Quantity & Level & Cycles & Label \\", r"\midrule"]
    for comp, sub in lab.groupby("component", sort=False):
        first = True
        for _, r in sub.iterrows():
            L.append(f"{comp if first else ''} & {r.quantity if first else ''} & {r.level} & {r.cycles} & {r.binary_label} \\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_labelmap_table.tex").write_text("\n".join(L), encoding="utf-8")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Mapping of the 17 sensor channels to the seven physical families. Occlusion, permutation "
         r"and SHAP attributions in Table~\ref{tab:explainagree} and in the priority score use the 17 channels "
         r"as groups; Figs.~\ref{fig:evidencegraph} and~\ref{fig:importance} occlude or permute each family "
         r"as one group. Motor power and vibration are separate families; cooling efficiency and cooling "
         r"power form one family; the efficiency factor is its own family.}",
         r"\label{tab:channelmap}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lllr}", r"\toprule", r"Family & Channels & Unit & Rate (Hz) \\", r"\midrule"]
    for fam, chs in FAMILIES.items():
        u = sorted(set(CHANNEL_INFO[c][1] for c in chs)); hz = sorted(set(CHANNEL_INFO[c][2] for c in chs))
        L.append(f"{fam} & {', '.join(chs)} & {', '.join(u).replace('degC', r'$^\circ$C')} & {', '.join(map(str, hz))} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_channelmap_table.tex").write_text("\n".join(L), encoding="utf-8")

    # split comparison (R3 R4 R5)
    def cell(r, c):
        npos = int(r[f"{c}_npos"])
        return "n/a" if npos == 0 else f"{r[f'{c}_f1']:.3f}"
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Component-fault classification under four evaluation protocols (seed 2026 per-component "
         r"F1 on pooled held-out predictions; macro-F1 is the mean over folds of the per-fold macro-F1, averaged "
         r"over the stated seeds). Positive/negative test support is given in parentheses. F1 is reported as "
         r"n/a when the held-out set contains no positive case, because recall is then undefined; ``Macro-F1, "
         r"defined'' averages only the components whose F1 is defined in each fold. Diagnostics: share of "
         r"held-out cycles whose condition profile also occurs in training, and held-out cycles that are "
         r"immediately adjacent in time to a training cycle.}",
         r"\label{tab:splitcompare}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"Protocol & Seeds & Macro-F1 & Macro-F1, & Cooler & Valve & Pump & Accum. & Profile in train / \\",
         r" & & (all 4) & defined & F1 (+/$-$) & F1 (+/$-$) & F1 (+/$-$) & F1 (+/$-$) & adjacent cycles \\",
         r"\midrule"]
    for _, r in spl.iterrows():
        dg = diag[diag.split == r.key].iloc[0]
        cells = " & ".join(f"{cell(r, c)} ({int(r[c + '_npos'])}/{int(r[c + '_nneg'])})" for c in COMP)
        L.append(f"{r.split} & {int(r.seeds)} & {r.macro_f1_all4:.3f} & {r.macro_f1_defined:.3f} & {cells} & "
                 f"{dg.test_cycles_with_profile_in_train:.2f} / {int(dg.test_cycles_adjacent_to_train_cycle_total)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (MAN / "generated_rev_split_table.tex").write_text("\n".join(L), encoding="utf-8")

    # ranking + bound (R6)
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Top-1 component-fault ranking error on fault-containing held-out cycles, the probability-derived "
         r"union bound $B_p$ of \eqref{eq:ceilingbound}, and the order-preservation check of "
         r"Proposition~\ref{prop:bound}. ``Violation'' is the share of cycles with at least one pair $i,j$ such that "
         r"$p_i>p_j$ but $s_i\le s_j$; $B_p$ is guaranteed to cover a policy only when this share is zero. "
         r"``Top-1 = prob.'' is the share of cycles on which the policy's first-ranked component equals probability-only's. "
         r"$B_s$ is the bound recomputed from the policy's own score with ties counted as misranks, which holds "
         r"for every policy by construction. Seed 2026; 95\% intervals are run-clustered (2{,}000 resamples).}",
         r"\label{tab:boundcheck}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llrlrrrr}", r"\toprule",
         r"Protocol & Policy & Error & [95\% CI] & $B_p$ & $B_s$ & Violation & Top-1 = prob. \\", r"\midrule"]
    for key, sub in rk.groupby("split_key", sort=False):
        first = True
        for _, r in sub.iterrows():
            L.append(f"{r.split if first else ''} & {r.policy} & {r.top1_error:.3f} & {ci_str(r.err_ci_lo, r.err_ci_hi)} & "
                     f"{r.bound_prob:.3f} & {r.bound_own_score:.3f} & {r.strict_reversal_frac:.3f} & "
                     f"{r.top1_agrees_with_prob_frac:.3f} \\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table*}", ""]
    (MAN / "generated_rev_boundcheck_table.tex").write_text("\n".join(L), encoding="utf-8")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Run-clustered paired bootstrap differences (seed 2026, 2{,}000 resamples, 95\% intervals). "
         r"$\Delta_{\mathrm{prio}}$: priority-score error minus probability-only error. "
         r"$E-B_p$: a policy's error minus the probability-derived bound, computed on the same resample.}",
         r"\label{tab:paireddiff}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{lll}", r"\toprule", r"Protocol & Quantity & Estimate [95\% CI] \\", r"\midrule"]
    lbl = {"A": "Blocked (orig.)", "B": "Blocked, purged", "C": "Profile holdout", "T": "Chronological"}
    for _, r in od.iterrows():
        L.append(f"{lbl[r.split_key]} & $\\Delta_{{\\mathrm{{prio}}}}$ & {r.priority_minus_prob:+.3f} {ci_str(r.priority_minus_prob_ci_lo, r.priority_minus_prob_ci_hi)} \\\\")
        L.append(f" & $E-B_p$, prob.\\ only & {r.probonly_minus_bound:+.3f} {ci_str(r.probonly_minus_bound_ci_lo, r.probonly_minus_bound_ci_hi)} \\\\")
        L.append(f" & $E-B_p$, local occl. & {r.localocc_minus_bound:+.3f} {ci_str(r.localocc_minus_bound_ci_lo, r.localocc_minus_bound_ci_hi)} \\\\")
        L.append(f" & $E-B_p$, priority & {r.priority_minus_bound:+.3f} {ci_str(r.priority_minus_bound_ci_lo, r.priority_minus_bound_ci_hi)} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_paired_table.tex").write_text("\n".join(L), encoding="utf-8")

    # calibration
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Calibration of the raw random-forest probabilities (no calibrator is fitted) on pooled "
         r"out-of-fold predictions for the three leakage-controlled protocols, seed 2026. Brackets: run-clustered "
         r"95\% intervals (1{,}000 resamples).}",
         r"\label{tab:calibrationoof}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{llrll}", r"\toprule", r"Protocol & Comp. & Pos. & Brier [95\% CI] & ECE [95\% CI] \\", r"\midrule"]
    for key, sub in cal.groupby("split_key", sort=False):
        first = True
        for _, r in sub.iterrows():
            L.append(f"{lbl[key] if first else ''} & {r.component} & {r.pos_rate:.2f} & {r.brier:.3f} {ci_str(r.brier_lo, r.brier_hi)} & "
                     f"{r.ece:.3f} {ci_str(r.ece_lo, r.ece_hi)} \\\\")
            first = False
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_calibration_table.tex").write_text("\n".join(L), encoding="utf-8")

    # sensitivity on blocked
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Priority-weight sensitivity on the leakage-safe blocked split (pooled out-of-fold, seed 2026). "
         r"Score $p_c(1+\gamma\bar a_c)S_c^{a}/E_c^{b}$. $\Delta$: top-1 error minus probability-only error, with a "
         r"run-clustered paired 95\% interval. The last row selects $(\gamma,a,b)$ from a 48-point grid by inner "
         r"cross-validation on the training blocks of each outer fold only.}",
         r"\label{tab:sensitivity}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{lrrrrl}", r"\toprule", r"Scenario & $\gamma$ & $a$ & $b$ & Error & $\Delta$ [95\% CI] \\", r"\midrule"]
    for _, r in sens.iterrows():
        g_ = "--" if np.isnan(r.gamma) else f"{r.gamma:.1f}"
        dl = "--" if np.isnan(r.lo) else f"{r.minus_prob:+.3f} {ci_str(r.lo, r.hi)}"
        if r.scenario == "Probability only":
            dl = "0 (reference)"
        L.append(f"{r.scenario} & {g_} & {r.s_exp:.1f} & {r.e_exp:.1f} & {r.top1_error:.3f} & {dl} \\\\")
    nz = res["nested"]
    L.append(r"\midrule")
    L.append(f"Nested selection & \\multicolumn{{3}}{{c}}{{per fold}} & {nz['error']:.3f} & {nz['minus_prob']:+.3f} {ci_str(*nz['ci'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_sensitivity_table.tex").write_text("\n".join(L), encoding="utf-8")

    # R13
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Reconciliation of the blocked-split top-1 figures. Table~\ref{tab:ceilingbound} pools the "
         r"out-of-fold predictions of all five folds (2205 cycles, seed 2026). The degradation sweep "
         r"(Table~\ref{tab:degradationwalk}) evaluates only fold~0 (blocks 0 and 9, 445 cycles) and averages "
         r"ten seeds. Top-1 accuracy is shown for both; ``pooled'' columns give top-1 error.}",
         r"\label{tab:reconcile}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{rrrrr}", r"\toprule",
         r"Seed & Fold 0 acc. & Fold 0 acc. & Pooled err. & Pooled err. \\",
         r" & prob.\ only & priority & prob.\ only & priority \\", r"\midrule"]
    for _, r in r13.iterrows():
        L.append(f"{int(r.seed)} & {r.fold0_prob_top1_acc:.3f} & {r.fold0_priority_top1_acc:.3f} & "
                 f"{r.pooled_prob_top1_err:.3f} & {r.pooled_priority_top1_err:.3f} \\\\")
    m = r13.mean()
    L += [r"\midrule", f"Mean & {m.fold0_prob_top1_acc:.3f} & {m.fold0_priority_top1_acc:.3f} & "
          f"{m.pooled_prob_top1_err:.3f} & {m.pooled_priority_top1_err:.3f} \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_reconcile_table.tex").write_text("\n".join(L), encoding="utf-8")

    # explanation stability (R9)
    q = [("occ_ch", r"Occlusion, 17 channels, fit vs.\ fit"), ("perm_ch", r"Permutation, 17 channels, fit vs.\ fit"),
         ("shap_ch", r"SHAP, 17 channels, fold vs.\ fold"), ("occ_fam", r"Occlusion, 7 families, fit vs.\ fit"),
         ("perm_fam", r"Permutation, 7 families, fit vs.\ fit"), ("occ_vs_shap_perfold", r"Occlusion vs.\ SHAP, same fold"),
         ("occ_vs_perm_perfold", r"Occlusion vs.\ permutation, same fold")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Stability of the grouped attributions on the blocked split: median (minimum) Spearman correlation "
         r"between importance vectors. ``Fit vs.\ fit'' compares all pairs among ten fits (five folds $\times$ seeds 17 "
         r"and 2026); SHAP rows use seed 2026 and 120 held-out cycles per fold. Pairs with a constant vector are "
         r"excluded (up to 38\% of permutation pairs for cooler and accumulator, where most groups have zero drop).}",
         r"\label{tab:stability}", r"\footnotesize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{lrrrr}", r"\toprule", r"Comparison & Cooler & Valve & Pump & Accum. \\", r"\midrule"]
    for key, name in q:
        sub = stab[stab.quantity == key].set_index("component")
        cells = " & ".join(f"{sub.loc[c, 'median_rho']:.2f} ({sub.loc[c, 'min_rho']:.2f})" for c in CNAME)
        L.append(name + " & " + cells + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (MAN / "generated_rev_stability_table.tex").write_text("\n".join(L), encoding="utf-8")
    for name in ["split", "calibration", "sensitivity", "stability", "boundcheck", "channelmap"]:
        fit_width(MAN / f"generated_rev_{name}_table.tex")


if __name__ == "__main__":
    main()
