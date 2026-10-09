# results/

Small outputs tracked in Git. Large artefacts (`*.pkl`) are not.

- **`paper_tables/`** – the tables behind the paper's reported numbers. Produced by
  `scripts/make_paper_outputs.py` from the saved out-of-fold predictions
  (`predictions_*.pkl`): main metrics, DeLong and Wilcoxon tests, operating-point
  metrics, and 5-fold decision curves. If a number here differs from another file,
  this folder is the reference.
- **`results_MIMIC_IV.csv`, `results_eICU.csv`, `results_JUCO_XGB_MIMIC_IV.csv`,
  `aggregate_results.csv`** – the pipeline's own per-run summaries. They agree with
  `paper_tables/` to about the third decimal (for example JUCO MIMIC-IV AUROC 0.8143
  here against 0.8141 there) because they come from separate evaluation passes.
- **`stress_results_incremental.csv`** – MNAR stress test (2 folds, 10-70%); input to
  the stress-test figure.
- **`ablation_MIMIC_IV.csv`, `dca_*.csv`, `missingness_stress_MIMIC_IV.csv`,
  `statistical_tests.csv`, `bootstrap_ci_auroc.csv`, `supplementary_metrics.csv`** –
  intermediate outputs of Parts 1-3. The `dca_*.csv` files use fewer folds than
  `paper_tables/dca_*_5fold.csv`; `statistical_tests.csv` holds fold-level Wilcoxon
  results only.
- **`metric_comparison`, `ablation_study`, `fig_missingness`, `fig_dca` (pdf/png)** –
  quick-look plots written by `JUCO_Part3_Aggregate.py` from the CSVs above. The
  figures for the paper are in `figures/`.
