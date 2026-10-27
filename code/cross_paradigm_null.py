#!/usr/bin/env python3
"""
cross_paradigm_null.py  (run from D:\\erp_work; needs no training, only saved maps)

Question: is a model's NFS with its OWN paradigm's ERP reference any higher than its correlation with
the references of the OTHER paradigms? If a model map correlates about equally with every reference,
the NFS reflects a generic channel pattern (e.g. which electrodes have most band power), not paradigm-specific physiology.

For every model (RF, XGB, GAT) and paradigm with a reference (ERN excluded: stimulus vs response locked):
  * mean fold map at each step, correlated with own reference and with each other paradigm's reference
  * reported per step, and for the step-averaged map (each step's map z-scored first)
  * rank of own correlation among all references (1 = own is the highest)
Output: nfs_null_v2.csv and a printed summary.
"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage4_classical_ml.nfs_fixed import pearson, align_reference  # noqa: E402
import run_gat_nfs as G  # noqa: E402

PARADIGMS = ["ERP_CORE_P3", "ERP_CORE_MMN", "ERP_CORE_N2pc", "ERP_CORE_N400", "ERP_CORE_N170"]
STEPS = [5, 10, 20, 30, 50, 75, 100, 150, 200]
CLS = Path("results_cls_v2"); GAT = Path("results_gat_v3"); FEAT = Path("features_v2")


def z(a):
    a = np.asarray(a, float)
    return (a - a.mean()) / (a.std() + 1e-12)


def get_maps(model, par):
    out = {}
    if model in ("RF", "XGB"):
        ch = json.loads((FEAT / par / "meta.json").read_text())["ch_names"]
        for s in STEPS:
            f = CLS / "maps" / par / f"{model}_step_{s}_foldmaps.npy"
            if f.exists():
                out[s] = np.load(f).mean(axis=0)
    else:
        ch = None
        for s in STEPS:
            f = GAT / "maps" / par / f"step_{s}_foldmaps.npy"
            if f.exists():
                out[s] = np.load(f).mean(axis=0)
                ch = json.loads((GAT / "maps" / par / f"step_{s}_subjects.json").read_text())["ch_names"]
    return ch, out


def refs_aligned(ch):
    r = {}
    for p in PARADIGMS:
        ref = G.load_reference(G.PARADIGM_TO_NFS_NAME[p])
        r[p] = align_reference(ref[0], ref[1], ch)
    return r


def main():
    rows = []
    for model in ("RF", "XGB", "GAT"):
        for par in PARADIGMS:
            ch, maps = get_maps(model, par)
            if not maps:
                continue
            refs = refs_aligned(ch)
            def one(m, tag):
                own = pearson(m, refs[par])
                oth = {p: pearson(m, refs[p]) for p in PARADIGMS if p != par}
                allv = [own] + list(oth.values())
                rank = 1 + sum(v > own for v in oth.values() if np.isfinite(v))
                rows.append(dict(model=model, paradigm=par, step=tag, own=f"{own:.3f}",
                                 other_mean=f"{np.nanmean(list(oth.values())):.3f}",
                                 other_max=f"{np.nanmax(list(oth.values())):.3f}", rank_of_own=rank,
                                 n_refs=len(allv)))
            for s, m in sorted(maps.items()):
                if np.std(m) > 1e-12:
                    one(m, s)
            zs = [z(m) for m in maps.values() if np.std(m) > 1e-12]
            if zs:
                one(np.mean(zs, axis=0), "all_steps")
    import csv
    with open("nfs_null_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("step-averaged maps (rank 1 of 5 means own reference is the best match):")
    for r in rows:
        if r["step"] == "all_steps":
            print(f"  {r['model']:3s} {r['paradigm']:14s} own={r['own']:>6s} others mean={r['other_mean']:>6s} "
                  f"max={r['other_max']:>6s} rank={r['rank_of_own']}/5")
    for model in ("RF", "XGB", "GAT"):
        rr = [r for r in rows if r["model"] == model and r["step"] != "all_steps"]
        if rr:
            top = sum(r["rank_of_own"] == 1 for r in rr)
            print(f"{model}: own reference ranked first in {top}/{len(rr)} paradigm-step maps "
                  f"(chance = {len(rr) / 5:.1f})")


if __name__ == "__main__":
    main()
