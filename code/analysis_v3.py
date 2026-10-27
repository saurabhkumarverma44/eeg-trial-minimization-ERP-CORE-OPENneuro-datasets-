#!/usr/bin/env python3
"""
analysis_v3.py  (run from D:\\erp_work; needs no training)

Analysis-only fixes for the revision. Reads every per-fold accuracy file and writes tables to analysis_v3\\:
  1. fold_accounting.csv      folds expected / valid / undefined / skipped per model and paradigm
                              (replaces the "5,297 of 7,160" ambiguity with an exact count)
  2. learning_curves.csv      mean balanced accuracy per model x paradigm x N, with a SUBJECT-bootstrap 95% CI
  3. first_N_above_chance.csv smallest N whose CI lower bound is above 0.5 AND stays above for all larger N
                              (replaces the single N* from a regression on a saturating curve)
  4. slope_ci.csv             slope of mean BA on ln(N) per model, subject-cluster bootstrap CI, over paradigms
                              where the model learns at all (spread of mean BA > 0.02); collapsed ones listed separately
  5. tests_N200_vs_N20.csv    paired Wilcoxon ACROSS SUBJECTS, BA at N=200 vs N=20, Holm-corrected within model
  6. tests_model_pairs.csv    paired Wilcoxon across subjects between models at N=200, Holm-corrected
Subjects, not grid steps, are the independent units everywhere. Interrupted runs leave duplicate fold rows;
the LAST row per (paradigm, model, step, subject) is kept.
"""
import itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUT = Path("analysis_v3"); OUT.mkdir(exist_ok=True)
RNG = np.random.default_rng(0)
NB = 2000


def load():
    frames = []
    f = Path("results_cls_v2/fold_accuracy_v2.csv")
    if f.exists():
        d = pd.read_csv(f); d = d.rename(columns={"bal_acc": "ba"}); frames.append(d[["paradigm", "model", "step", "subject", "ba", "note"]])
    f = Path("results_gat_v3/fold_accuracy_gat_v3.csv")
    if f.exists():
        d = pd.read_csv(f); d["model"] = "GAT"; d = d.rename(columns={"bal_acc": "ba"}); d["note"] = ""; frames.append(d[["paradigm", "model", "step", "subject", "ba", "note"]])
    for f in sorted(Path(".").glob("results_dl_v3_*/fold_accuracy_dl_v3.csv")):
        if f.exists():
            d = pd.read_csv(f); d = d.rename(columns={"balanced_accuracy": "ba"}); d["note"] = d["note"].fillna(""); frames.append(d[["paradigm", "model", "step", "subject", "ba", "note"]])
    d = pd.concat(frames, ignore_index=True)
    d = d[d.step.astype(str) != "full"].copy(); d["step"] = d.step.astype(int)
    d["model"] = d.model.replace({"cnn1d": "CNN-1D", "eegnet": "EEGNet"})
    return d.drop_duplicates(["paradigm", "model", "step", "subject"], keep="last")


def boot_mean_ci(vals, nb=NB):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, np.nan, np.nan
    idx = RNG.integers(0, len(vals), size=(nb, len(vals)))
    m = vals[idx].mean(axis=1)
    return vals.mean(), *np.percentile(m, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def main():
    d = load()
    print(f"{len(d)} fold rows, models: {sorted(d.model.unique())}, paradigms: {d.paradigm.nunique()}")

    # 1. fold accounting
    rows = []
    for (m, p), g in d.groupby(["model", "paradigm"]):
        nsub = g.subject.nunique(); steps = g.step.nunique()
        expected = nsub * steps
        rows.append(dict(model=m, paradigm=p, subjects=nsub, steps=steps, expected_folds=expected, recorded=len(g),
                         valid=int(g.ba.notna().sum()), undefined=int(g.ba.isna().sum()),
                         missing=expected - len(g)))
    acc = pd.DataFrame(rows); acc.to_csv(OUT / "fold_accounting.csv", index=False)
    tot = acc.groupby("model")[["expected_folds", "recorded", "valid", "undefined", "missing"]].sum()
    print("\nFOLD ACCOUNTING per model (expected = subjects x steps)"); print(tot.to_string())

    # 2. learning curves with subject bootstrap
    lc = []
    for (m, p, s), g in d.groupby(["model", "paradigm", "step"]):
        mean, lo, hi = boot_mean_ci(g.ba.values)
        lc.append(dict(model=m, paradigm=p, step=s, n_subjects=int(g.ba.notna().sum()), mean_ba=mean, ci_low=lo, ci_high=hi))
    lc = pd.DataFrame(lc); lc.to_csv(OUT / "learning_curves.csv", index=False)

    # 3. first N reliably above chance
    fa = []
    for (m, p), g in lc.groupby(["model", "paradigm"]):
        g = g.sort_values("step"); ok = (g.ci_low > 0.5).values; first = None
        for i in range(len(g)):
            if ok[i:].all():
                first = int(g.step.iloc[i]); break
        fa.append(dict(model=m, paradigm=p, first_N_above_chance=first if first else "not reached",
                       best_mean_ba=round(g.mean_ba.max(), 4)))
    fa = pd.DataFrame(fa); fa.to_csv(OUT / "first_N_above_chance.csv", index=False)
    print("\nFIRST N WITH CI LOWER BOUND > 0.5 (and for all larger N)"); print(fa.pivot(index="paradigm", columns="model", values="first_N_above_chance").to_string())

    # 4. slope of BA on ln N, cluster bootstrap over subjects within paradigm
    sl = []
    for m, gm in d.groupby("model"):
        learn = [p for p, gp in lc[lc.model == m].groupby("paradigm") if gp.mean_ba.max() - gp.mean_ba.min() > 0.02]
        collapsed = sorted(set(gm.paradigm.unique()) - set(learn))
        if not learn:
            sl.append(dict(model=m, learning_paradigms="none", collapsed=",".join(collapsed), slope=np.nan, lo=np.nan, hi=np.nan)); continue
        steps = sorted(gm.step.unique()); x = np.log(steps)
        piv = {p: gm[gm.paradigm == p].pivot_table(index="subject", columns="step", values="ba") for p in learn}
        def slope(pv):
            y = pv.mean(axis=0).reindex(steps).values
            return np.polyfit(x, y, 1)[0]
        s0 = np.mean([slope(pv) for pv in piv.values()])
        bs = []
        for _ in range(500):
            vals = []
            for pv in piv.values():
                idx = RNG.integers(0, len(pv), len(pv)); vals.append(slope(pv.iloc[idx]))
            bs.append(np.mean(vals))
        sl.append(dict(model=m, learning_paradigms=",".join(learn), collapsed=",".join(collapsed), slope=s0,
                       lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
    sl = pd.DataFrame(sl); sl.to_csv(OUT / "slope_ci.csv", index=False)
    print("\nSLOPE OF MEAN BA PER ln(N) (mean over learning paradigms, subject-cluster bootstrap 95% CI)"); print(sl.to_string(index=False))

    # 5. N=200 vs N=20 across subjects
    t1 = []
    for (m, p), g in d.groupby(["model", "paradigm"]):
        pv = g.pivot_table(index="subject", columns="step", values="ba")
        if 20 in pv.columns and 200 in pv.columns:
            pr = pv[[20, 200]].dropna()
            if len(pr) >= 6 and (pr[200] - pr[20]).abs().sum() > 0:
                t1.append(dict(model=m, paradigm=p, n=len(pr), mean_diff=(pr[200] - pr[20]).mean(), p=wilcoxon(pr[200], pr[20]).pvalue))
    t1 = pd.DataFrame(t1)
    if len(t1):
        t1["p_holm"] = t1.groupby("model")["p"].transform(lambda s: holm(s.values)); t1.to_csv(OUT / "tests_N200_vs_N20.csv", index=False)
        print("\nN=200 vs N=20, paired across subjects, Holm within model"); print(t1.round(4).to_string(index=False))

    # 6. model pairs at N=200
    t2 = []
    g200 = d[d.step == 200]
    for p, gp in g200.groupby("paradigm"):
        pv = gp.pivot_table(index="subject", columns="model", values="ba")
        for a, b in itertools.combinations(pv.columns, 2):
            pr = pv[[a, b]].dropna()
            if len(pr) >= 6 and (pr[a] - pr[b]).abs().sum() > 0:
                t2.append(dict(paradigm=p, a=a, b=b, n=len(pr), mean_diff_a_minus_b=(pr[a] - pr[b]).mean(), p=wilcoxon(pr[a], pr[b]).pvalue))
    t2 = pd.DataFrame(t2)
    if len(t2):
        t2["p_holm"] = holm(t2.p.values); t2.to_csv(OUT / "tests_model_pairs.csv", index=False)
        print(f"\nMODEL PAIRS at N=200 across subjects: {len(t2)} tests, {int((t2.p_holm < 0.05).sum())} survive Holm at 0.05")
        print(t2[t2.p_holm < 0.05].round(4).to_string(index=False))
    print(f"\nTables written to {OUT}\\")


if __name__ == "__main__":
    main()
