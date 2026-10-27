# Stage 3 — Trial Subsampling Grid & Feature Extraction

Completely separate from `preprocessing_batch/` (Phase 2, frozen/submitted).
Nothing here imports from or writes to `preprocessing_batch/` or any
`results_batch/<DATASET>/` Phase 2 output — Stage 3 only **reads** the
`*-epo.fif` files Phase 2 produced. ALL Stage 3 output goes into its own
top-level folder, `stage3_results/`, sitting next to (not inside)
`results_batch/`.

## Install location

```
/lp-dev/23BRS1328/erp_trial_min/
    preprocessing_batch/        <- Phase 2 (untouched)
    results_batch/               <- Phase 2 output (read-only)
        ERP_CORE/ DS006018/ DS002680/
    stage3_feature_extraction/  <- Stage 3 scripts (this folder)
    stage3_results/               <- ALL Stage 3 output (new, separate folder)
```

## Run order

```bash
cd /lp-dev/23BRS1328/erp_trial_min
python stage3_feature_extraction/run_stage3_all.py     # 3.1 -> 3.7
python stage3_feature_extraction/extract_features.py   # full-trial ML baseline X/y
```

or run each sub-step individually (each is independently resumable —
already-completed files are skipped unless you pass `overwrite=True`):

| Spec ID | Script                      | What it does |
|---------|------------------------------|--------------|
| 3.1 | `subsample_trials.py`        | Builds the subsampling grid (5,10,20,30,50,75,100,150,200,full) per subject/paradigm |
| 3.2 | `bootstrap_grid.py`          | 1000-iteration bootstrap (with replacement) per step per subject, mean ± 95% CI |
| 3.3 | `features_band_power.py`     | Relative band power (delta/theta/alpha/beta/gamma) per electrode, z-scored |
| 3.4 | `features_plv.py`            | PLV connectivity matrix per epoch, per band (alpha, beta) |
| 3.5 | `features_wpli.py`           | wPLI connectivity matrix per epoch, per band + PLV-vs-wPLI comparison for 3 subjects |
| 3.6 | `features_entropy.py`        | Spectral entropy per channel per epoch (+ per-dataset aggregate) |
| 3.7 | `assemble_features.py`       | Combines 3.3–3.6 into one feature matrix per paradigm per step |
| —   | `extract_features.py`        | Full-trial (non-subsampled) X/y/metadata, ready for SVM/LDA/RF baseline |

`config3.py` holds every tunable parameter. `stage3_utils.py` and
`connectivity_core.py` are shared helper modules — they don't produce output
on their own.

## Output layout (all under `stage3_results/`, separate from `results_batch/`)

```
stage3_results/
    subsample_indices/<DATASET>/<subject>_<task>_indices.json
    bootstrap/<DATASET>_<task>_bootstrap.csv
    features/
        band_power/<DATASET>/<subject>_<task>_bandpower.npy (+ .json sidecar)
        plv/<DATASET>/<subject>_<task>_<band>_plv.npy (+ _flat.npy)
        wpli/<DATASET>/<subject>_<task>_<band>_wpli.npy (+ _flat.npy)
        entropy/<DATASET>/<subject>_<task>_entropy.npy (+ per-task aggregate)
    feature_matrices/<paradigm>/<paradigm>_step-<step>.npy (+ _meta.csv, _shape.json)
    ml/<paradigm>/{X.npy,y.npy,y_codes.npy,label_map.json,metadata.csv}
    ml/ALL_PARADIGMS_COMBINED/{...}
    audit/plv_vs_wpli_comparison.csv
    logs/phase3*.log
```

## Standard conventions used (finalized — these are the defaults now baked into the code)

1. **PLV / wPLI "per epoch."** Computed within each individual epoch (band-pass
   filter -> Hilbert transform -> phase/cross-spectrum across the epoch's own
   time samples), giving one matrix per epoch with shape `(n_epochs, n_ch, n_ch)`
   as required. This is a single-trial connectivity estimate — a different
   quantity from the textbook across-trials PLV, but it's the standard this
   pipeline now uses consistently everywhere (3.4, 3.5, 3.7, and
   `extract_features.py` all use the same definition).
2. **Bootstrap metric.** Mean Global Field Power (GFP) of the bootstrap-resampled
   evoked response — paradigm-agnostic, works for every task. This is the
   fixed default (`METRIC_FN` in `bootstrap_grid.py`).
3. **Step > available trials.** 3.1 (without replacement) falls back to "all
   available trials" and flags it `False` in `step_integrity`. 3.2 (bootstrap,
   with replacement) still runs at the requested step size regardless — this
   is standard bootstrap practice. `feature_matrices/` (3.7) only uses
   trials where `step_integrity` is `True`, so under-powered steps are
   excluded automatically rather than padded or duplicated.
4. **Labels.** `extract_features.py` reads the label for every epoch straight
   from that epoch file's own `event_id` mapping — i.e. whatever condition
   Phase 2's epoching already assigned per `event_registry.py` (target/standard,
   correct/error, etc.). No relabeling logic is added.
5. **Feature vector composition** (used identically by `assemble_features.py`
   and `extract_features.py`):
   `[ band_power.flatten(), PLV_alpha_flat, PLV_beta_flat, wPLI_alpha_flat, wPLI_beta_flat, entropy ]`

These are now the project's standing conventions for Stage 3 — no further
sign-off needed to run the pipeline end-to-end. If your supervisor later
wants a different connectivity definition or bootstrap metric, only
`connectivity_core.py` / `bootstrap_grid.py` need to change — every
downstream script picks it up automatically.

