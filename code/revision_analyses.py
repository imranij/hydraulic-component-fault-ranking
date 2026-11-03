"""
Revision analyses for paper 09 (IEEE TIM major revision, reviewer comments R1-R16).

Stage 1 (``--stage fit``) fits every model the revision needs and caches the
out-of-fold (OOF) outputs under results/revision/cache/.  Stage 2
(``--stage analyse``) reads only that cache and writes every CSV, table and
figure.  Nothing in the original pipeline is modified; the original blocked
split is reproduced exactly as a control before any new split is used.

Splits
------
A  original blocked split: 10 contiguous equal-count blocks (idx // 220),
   GroupKFold(5) -> each test fold is two non-adjacent blocks.  No purge.
B  run-aligned purged blocked split: the 10 block boundaries are moved to the
   nearest condition-run boundary so no run is split, and the condition-runs
   immediately before and after every test block are removed from training.
C  condition-profile holdout: GroupKFold(5) over the 144 unique
   (cooler, valve, pump, accumulator) condition profiles, so every test
   profile is absent from training.
T  chronological 70/30 (as in the paper).

Reviewer items addressed: R2 (label mapping), R3 (B), R4 (C), R5 (support /
undefined F1), R6 (order preservation, tie handling, paired error-minus-bound),
R8 (channel->family map, data-driven graph), R9 (explanation stability),
R10 (OOF calibration), R11 (blocked weight sensitivity, nested selection),
R12 (label-free example selection), R13 (Table VIII vs XIII reconciliation),
R14 (HistGradientBoosting ranking and explanations).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, f1_score, log_loss, average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ceiling_analysis as ca  # noqa: E402
import paper3_diagnostics_experiment as p3  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "revision"
CACHE = OUT / "cache"
FIG = ROOT / "figures"
MAN = ROOT / "manuscript"
for d in (OUT, CACHE, FIG):
    d.mkdir(parents=True, exist_ok=True)

COMP = p3.COMPONENTS
SEV, EFF = p3.SEVERITY, p3.EFFORT
SEEDS10 = [17, 2026, 101, 202, 303, 404, 505, 606, 707, 808]   # degradation-walk seeds
SEEDS3 = [17, 2026, 4242]                                      # comparator seeds
N_BOOT = 2000
PRIO = "Priority score (prob + evidence + severity/effort)"

FAMILIES = {
    "Pressure": ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"],
    "Volume flow": ["FS1", "FS2"],
    "Temperature": ["TS1", "TS2", "TS3", "TS4"],
    "Motor power": ["EPS1"],
    "Vibration": ["VS1"],
    "Cooling (CE, CP)": ["CE", "CP"],
    "Efficiency factor": ["SE"],
}
CHANNEL_INFO = {  # from the UCI documentation.txt
    "PS1": ("Pressure", "bar", 100), "PS2": ("Pressure", "bar", 100), "PS3": ("Pressure", "bar", 100),
    "PS4": ("Pressure", "bar", 100), "PS5": ("Pressure", "bar", 100), "PS6": ("Pressure", "bar", 100),
    "EPS1": ("Motor power", "W", 100), "FS1": ("Volume flow", "l/min", 10), "FS2": ("Volume flow", "l/min", 10),
    "TS1": ("Temperature", "degC", 1), "TS2": ("Temperature", "degC", 1), "TS3": ("Temperature", "degC", 1),
    "TS4": ("Temperature", "degC", 1), "VS1": ("Vibration", "mm/s", 1), "CE": ("Cooling efficiency (virtual)", "%", 1),
    "CP": ("Cooling power (virtual)", "kW", 1), "SE": ("Efficiency factor", "%", 1),
}


# ---------------------------------------------------------------------------
# data and splits
# ---------------------------------------------------------------------------

def load():
    f, t, g = p3.build_feature_table()
    y = t[[f"{c}_fault" for c in COMP]].to_numpy(int)
    rows = [tuple(l.rstrip("\n").split("\t")) for l in open(ROOT / "data/uci_hydraulic_raw/profile.txt")]
    run = np.concatenate([[i] * len(list(gr)) for i, (_, gr) in enumerate(itertools.groupby(rows))])
    prof = t[["cooler_condition", "valve_condition", "pump_leakage", "accumulator_pressure"]].astype(int)
    prof_id = prof.astype(str).agg("-".join, axis=1).astype("category").cat.codes.to_numpy()
    return f, t, g, y, run, prof_id


def split_A(n):
    idx = np.arange(n)
    blocks = (idx // (n // 10)).clip(max=9)
    return [(tr, te) for tr, te in GroupKFold(5).split(idx, groups=blocks)], blocks


def split_B(n, run):
    """Run-aligned blocks + purge of the adjacent condition-run on each side of every test block."""
    idx = np.arange(n)
    starts = np.flatnonzero(np.r_[True, run[1:] != run[:-1]])          # first cycle of each run
    bounds = [0]
    for k in range(1, 10):
        target = k * n / 10
        b = int(starts[np.argmin(np.abs(starts - target))])
        bounds.append(max(b, bounds[-1] + 1))
    bounds.append(n)
    blocks = np.zeros(n, int)
    for k in range(10):
        blocks[bounds[k]:bounds[k + 1]] = k
    folds, purged = [], []
    for tr, te in GroupKFold(5).split(idx, groups=blocks):
        te_runs = set(run[te])
        drop = set()
        for b in set(blocks[te]):
            lo, hi = bounds[b], bounds[b + 1]
            if lo > 0:
                drop.add(run[lo - 1])
            if hi < n:
                drop.add(run[hi])
        drop -= te_runs
        keep = ~np.isin(run[tr], list(drop))
        purged.append(int((~keep).sum()))
        folds.append((tr[keep], te))
    return folds, blocks, purged


def split_C(n, prof_id):
    idx = np.arange(n)
    return [(tr, te) for tr, te in GroupKFold(5).split(idx, groups=prof_id)]


# ---------------------------------------------------------------------------
# learners with a common interface (estimators_ list, as p3.occlusion_scores expects)
# ---------------------------------------------------------------------------

class HGBMulti:
    def __init__(self, seed):
        self.seed = seed

    def fit(self, x, y):
        self.estimators_ = []
        for j in range(y.shape[1]):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, l2_regularization=0.02,
                                               random_state=self.seed)   # identical to comparators.py
            m.fit(x, y[:, j])
            self.estimators_.append(m)
        return self


def fit_fold(f, y, tr, te, seed, learner="rf"):
    if learner == "rf":
        prob, model, _, xtr, xte = ca.fit_predict(f, y, tr, te, seed)
        return prob, model, xtr, xte
    sc = StandardScaler()
    xtr = sc.fit_transform(f.iloc[tr].to_numpy(float))
    xte = sc.transform(f.iloc[te].to_numpy(float))
    model = HGBMulti(seed).fit(xtr, y[tr])
    prob = np.vstack([p3.positive_class_proba(e, xte) for e in model.estimators_]).T
    return prob, model, xtr, xte


def family_gidx(f, g):
    gi = p3.group_indices(f, g)
    return {fam: sum((gi[ch] for ch in chs), []) for fam, chs in FAMILIES.items()}


def run_oof(f, y, g, folds, seed, learner="rf", want_groups=True):
    n = len(y)
    gidx = p3.group_indices(f, g)
    fidx = family_gidx(f, g)
    P = np.full((n, 4), np.nan)
    O = np.full((n, 4), np.nan)
    fold_of = np.full(n, -1)
    occ_ch, occ_fam, perm_ch, perm_fam, f1s = [], [], [], [], []
    for k, (tr, te) in enumerate(folds):
        prob, model, xtr, xte = fit_fold(f, y, tr, te, seed, learner)
        base = np.mean(xtr, axis=0)
        occ = p3.occlusion_scores(model, xte, base, gidx)          # (n_te, 17, 4)
        P[te], O[te], fold_of[te] = prob, occ.mean(axis=1), k
        pred = (prob >= 0.5).astype(int)
        f1s.append([f1_score(y[te, j], pred[:, j], zero_division=0) for j in range(4)])
        if want_groups:
            occ_ch.append(occ.mean(axis=0))
            occ_fam.append(p3.occlusion_scores(model, xte, base, fidx).mean(axis=0))
            pc = p3.permutation_importance(model, xte, y[te], gidx, seed=seed)
            pf = p3.permutation_importance(model, xte, y[te], fidx, seed=seed)
            perm_ch.append(pc.pivot_table(index="sensor", columns="component", values="importance")
                           .reindex(index=list(gidx), columns=COMP).to_numpy())
            perm_fam.append(pf.pivot_table(index="sensor", columns="component", values="importance")
                            .reindex(index=list(fidx), columns=COMP).to_numpy())
    out = dict(P=P, O=O, fold_of=fold_of, fold_f1=np.array(f1s))
    if want_groups:
        out.update(occ_ch=np.array(occ_ch), occ_fam=np.array(occ_fam),
                   perm_ch=np.array(perm_ch), perm_fam=np.array(perm_fam))
    return out


def save(name, d):
    np.savez_compressed(CACHE / f"{name}.npz", **d)


def cached(name):
    p = CACHE / f"{name}.npz"
    return dict(np.load(p, allow_pickle=True)) if p.exists() else None


# ---------------------------------------------------------------------------
# nested priority-weight selection (R11)
# ---------------------------------------------------------------------------

GRID = [(gam, a, b) for gam in (0.0, 0.5, 1.0, 2.0) for a in (0.0, 0.5, 1.0, 1.5) for b in (0.0, 0.5, 1.0)]


def weighted_score(prob, conf, gam, a, b):
    return prob * (1.0 + gam * np.maximum(conf, 1e-6)) * (SEV ** a / EFF ** b)


def nested_selection(f, y, g, folds, blocks, seed, P_outer, O_outer):
    """For each outer test fold, choose (gamma, S-exp, E-exp) by inner GroupKFold over the
    outer-training blocks only, then score the chosen weights on the outer test fold."""
    n = len(y)
    chosen_score = np.full((n, 4), np.nan)
    choices = []
    for k, (tr, te) in enumerate(folds):
        tr_blocks = blocks[tr]
        n_inner = min(4, len(np.unique(tr_blocks)))
        inner = list(GroupKFold(n_inner).split(tr, groups=tr_blocks))
        res = run_oof(f.iloc[tr].reset_index(drop=True), y[tr], g,
                      [(a_, b_) for a_, b_ in inner], seed, want_groups=False)
        errs = [ca.top1_error(y[tr], weighted_score(res["P"], res["O"], *w)) for w in GRID]
        best = int(np.argmin(errs))          # first minimum: grid is ordered simplest-first
        w = GRID[best]
        chosen_score[te] = weighted_score(P_outer[te], O_outer[te], *w)
        choices.append({"outer_fold": k, "gamma": w[0], "s_exp": w[1], "e_exp": w[2],
                        "inner_top1_error": errs[best],
                        "inner_top1_error_prob_only": errs[GRID.index((0.0, 0.0, 0.0))]})
        print(f"  nested fold {k}: chose {w} inner err {errs[best]:.4f}", flush=True)
    return chosen_score, pd.DataFrame(choices)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def stage_fit(only=None):
    f, t, g, y, run, prof = load()
    n = len(y)
    A, blocksA = split_A(n)
    B, blocksB, purged = split_B(n, run)
    C = split_C(n, prof)
    json.dump({"purged_cycles_per_fold": purged,
               "B_block_bounds": [int(np.flatnonzero(blocksB == k)[0]) for k in range(10)]},
              open(OUT / "split_B_purge.json", "w"), indent=2)

    jobs = []
    for s in SEEDS10:
        jobs.append((f"A_rf_{s}", lambda s=s: run_oof(f, y, g, A, s, want_groups=s in SEEDS3 or s == 17)))
    for s in SEEDS3:
        jobs.append((f"B_rf_{s}", lambda s=s: run_oof(f, y, g, B, s, want_groups=(s == 2026))))
        jobs.append((f"C_rf_{s}", lambda s=s: run_oof(f, y, g, C, s, want_groups=(s == 2026))))
        jobs.append((f"A_hgb_{s}", lambda s=s: run_oof(f, y, g, A, s, "hgb", want_groups=(s == 2026))))
    cut = int(0.70 * n)
    idx = np.arange(n)
    jobs.append(("T_rf_2026", lambda: run_oof(f, y, g, [(idx[:cut], idx[cut:])], 2026, want_groups=False)))
    for name, fn in jobs:
        if only and not name.startswith(only):
            continue
        if (CACHE / f"{name}.npz").exists():
            print(f"[fit] {name}: cached", flush=True)
            continue
        t0 = time.time()
        save(name, fn())
        print(f"[fit] {name}: {time.time() - t0:.0f}s", flush=True)

    if (not only or only == "nested") and not (CACHE / "nested_A_2026.npz").exists():
        t0 = time.time()
        a26 = cached("A_rf_2026")
        sc, ch = nested_selection(f, y, g, A, blocksA, 2026, a26["P"], a26["O"])
        save("nested_A_2026", {"score": sc})
        ch.to_csv(OUT / "nested_weight_choices.csv", index=False)
        print(f"[fit] nested: {time.time() - t0:.0f}s", flush=True)

    if (not only or only == "shap") and not (CACHE / "shap_A_2026.npz").exists():
        import shap
        t0 = time.time()
        gidx = p3.group_indices(f, g)
        rng = np.random.default_rng(7)
        out = []
        for k, (tr, te) in enumerate(A):
            prob, model, xtr, xte = fit_fold(f, y, tr, te, 2026)
            sub = rng.choice(len(te), size=min(120, len(te)), replace=False)
            m = np.zeros((len(gidx), 4))
            for j in range(4):
                sv = shap.TreeExplainer(model.estimators_[j]).shap_values(xte[sub])
                sv1 = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
                ma = np.abs(sv1).mean(axis=0)
                for si, ch in enumerate(gidx):
                    m[si, j] = ma[gidx[ch]].sum()
            out.append(m)
            print(f"  shap fold {k}: {time.time() - t0:.0f}s", flush=True)
        save("shap_A_2026", {"shap_ch": np.array(out)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "analyse"], required=True)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.stage == "fit":
        stage_fit(a.only)
    else:
        import revision_analyse
        revision_analyse.main()
