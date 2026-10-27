Trial minimisation in EEG: code and results

Code and results for "Trial Minimisation in EEG: Subject Independent Learning Curves for Six ERP Components under a Matched Protocol" (Saurabh Kumar Verma, Aditi Vinodkumar Nair, Chirag Srivastava, Shridevi S and Harish Gunasekaran), submitted to Brain Informatics.

The study asks how many pooled training trials from other subjects a model needs before it classifies an unseen subject above chance. Five models (random forest, XGBoost, a graph attention network, a 1D CNN and EEGNet) are compared on six ERP CORE components (N400, ERN, N170, P3, N2pc, MMN) under one leave one subject out protocol.

Data

The ERP CORE data are not included. Download them from the Open Science Framework (Kappenman et al., NeuroImage 2021, doi 10.1016/j.neuroimage.2020.117465). Prepared epoch files and feature matrices are not included either; they are produced by the preprocessing and feature scripts below.

Layout
code/
  preprocessing_batch/        filtering, ICA, montage and epoching scripts
  stage3_feature_extraction/  band power, PLV, wPLI and entropy features, feature assembly, trial subsampling
  stage4_classical_ml/        random forest and XGBoost runs, NFS code (nfs_fixed.py)
  dl_models/                  1D CNN, EEGNet, GAT, trainer, leave one subject out evaluation, graph building
  dl_prep/                    epoch loading, scaling, data splitting and subsampling for the deep models
  configs/                    deep learning configuration
  build_features.py           feature building
  run_classical_nfs.py        random forest and XGBoost, pooled N protocol, with NFS maps
  run_dl_v3.py, run_dl_v4.py  1D CNN and EEGNet runs
  run_gat_v3.py, run_gat_v4.py, run_gat_nfs.py   graph attention network runs and attention maps
  cross_paradigm_null.py      cross component null for the Neuro Fidelity Score (NFS)
  analysis_v3.py              first N above chance, slopes, paired tests, model pair tests, fold accounting
  verify_cache.py, verify_loader.py   output checks used in the pipeline verification
results/
  *.csv                       per cell results and test outputs behind the paper's tables and figures
  per_fold/                   fold level accuracy files for each model
  nfs_maps_classical/         NFS importance maps per fold for random forest and XGBoost
  nfs_maps_gat/               NFS attention maps per fold and subject lists for the GAT
Results files
File	Content	Paper
learning_curves.csv	Mean balanced accuracy with 95% subject bootstrap interval per model, component and N	Fig. 4, Table 6
first_N_above_chance.csv	First N whose interval stays above 0.5	Table 7, Fig. 5
slope_ci.csv	Learning slope per ln N with cluster bootstrap interval	Table 8, Fig. 6a
tests_N200_vs_N20.csv	Paired change from N = 20 to N = 200, raw and Holm adjusted p	Table 9, Fig. 6b
tests_model_pairs.csv	Paired model comparisons at N = 200	Section 6.3
fold_accounting.csv	Fold ledger: expected, recorded, valid, undefined, missing	Table 5
classical_v2.csv, gat_nfs_v2.csv, nfs_null_v2.csv	NFS per cell and the cross component null	Table 10, Fig. 7
cnn_dl_v3.csv, eeg_dl_v3.csv, eeg_mmn_dl_v3.csv	CNN and EEGNet per cell accuracy, balanced accuracy and AUC	Fig. 4, Section 6.4
Notes
Scripts were written for our working directory (erp_work). Paths at the top of each script point to our local copies and need editing before you run them.
ERN in this study is a stimulus locked correct versus error contrast, not the response locked ERN.
NFS is exploratory and failed its cross component null test. It is not a measure of fidelity to a component.
A positive control decoder, class weighted deep networks and an N400 stimulus property check were not completed for the reported results.
Main experiments ran on an NVIDIA A100 GPU; the NFS reruns and later analyses ran on a CPU with seven threads.
Authors

Saurabh Kumar Verma, Aditi Vinodkumar Nair, Chirag Srivastava, Shridevi S (corresponding author, shridevi.s@vit.ac.in) and Harish Gunasekaran. Centre for Neuroinformatics, Vellore Institute of Technology, Chennai, India. H. Gunasekaran is also at Specs SAS (BCI), Snap Inc., Paris, France.
