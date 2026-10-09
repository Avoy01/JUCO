# JUCO: Joint Uncertainty-Calibrated Optimisation for ICU Mortality Prediction

Code accompanying the manuscript *"JUCO: A Joint Uncertainty-Calibrated Optimisation Framework for ICU Mortality Prediction with Safe Abstention Under Informative Missingness"* (Chowdhury, Kumari, Tripathi; under review).

JUCO predicts in-hospital mortality from the first 24 hours of an ICU stay and is built for the case where missingness itself is informative (MNAR). It has four parts, all implemented in `juco_core.py`:

1. **Adaptive quantile imputation**: LightGBM quantile regressors give a lower and an upper bound for each missing value instead of a point estimate.
2. **Proportional fuzzy transformation**: each interval becomes a trapezoidal fuzzy number, so wider intervals carry more uncertainty.
3. **Fuzzy decision forest with joint optimisation**: a 180-tree soft-routing forest, with the uncertainty and forest hyper-parameters θ = (α_L, α_U, η, d_max, n_min) tuned jointly by two-stage Differential Evolution.
4. **Safe abstention**: cases whose predictive entropy exceeds a per-fold threshold τ are deferred to a clinician. Quality is reported with AURC and Risk@90 on the full entropy-ranked risk-coverage curve.

Frozen θ\* from the MIMIC-IV Fold 1 search: α_L = 0.0555, α_U = 0.9384, η = 0.2869, d_max = 6, n_min = 33. `scripts/extract_theta_star.py` reads it from the checkpoint.

## Headline results (5-fold patient-level GroupKFold, mean over folds)

| Dataset | Model | AUROC | ECE | AURC |
|---|---|---|---|---|
| MIMIC-IV (82,785 stays, 10.58% mortality) | JUCO | 0.8141 | 0.0057 | 0.0337 |
| | XGBoost | 0.8421 | 0.0072 | 0.0285 |
| | MissForest+XGB | 0.8431 | 0.0062 | 0.0282 |
| eICU (162,136 stays, 5.32% mortality; θ\* frozen) | JUCO | 0.8191 | 0.0029 | 0.0161 |
| | XGBoost | 0.8343 | 0.0030 | 0.0152 |

JUCO does not win on raw discrimination. Its case rests on calibration and on robustness: when the training-time MNAR masking rate rises from 10% to 70%, JUCO's AUROC falls by 0.023, against 0.054 for XGBoost and 0.056 for MissForest+XGB. The full tables are in `results/paper_tables/`.

## Repository layout

```text
juco_core.py                 data loaders, imputer, fuzzy forest, DE optimiser, abstention, baselines, metrics
JUCO_Part1_MIMIC.py          MIMIC-IV primary experiment (DE on fold 1, 5-fold evaluation) and ablations
JUCO_Part1B_Robustness.py    JUCO-XGB, MNAR stress test, decision-curve inputs
JUCO_Part2_eICU.py           eICU external validation with frozen θ*
JUCO_Part3_Aggregate.py      post-processing: aggregate tables, Wilcoxon / Friedman tests, summary plots
scripts/
  make_paper_outputs.py      regenerates manuscript Figures 3, 4, 5, D1 and the paper tables from saved predictions
  cohort_summary.py          cohort characteristics tables (appendix)
  extract_theta_star.py      prints the frozen θ* from the MIMIC-IV checkpoint
  debug_cohort.py            loads both datasets and prints cohort statistics (no training)
  inspect_files.py           checks that the raw PhysioNet files are in place
figures/                     publication figures (Figure_1, Figure_2: architecture diagrams; Figures 3, 4, 5, D1: results)
results/                     small result CSVs from the pipeline; paper_tables/ holds the tables behind the manuscript
```

Large artefacts (`*.pkl` checkpoints and predictions) and raw data are not tracked.

## Data

Both datasets need credentialed PhysioNet access and are not redistributed here:

- MIMIC-IV v2.2: https://physionet.org/content/mimiciv/2.2/
- eICU Collaborative Research Database v2.0: https://physionet.org/content/eicu-crd/2.0/

Layout expected under the repository root:

```text
mimic-iv/hosp/{admissions,patients,labevents}.csv.gz
mimic-iv/icu/{icustays,chartevents}.csv.gz
eicu/{patient,lab,vitalPeriodic}.csv.gz
```

Run `python scripts/inspect_files.py` to confirm the files are found. The PhysioNet data use agreement forbids redistributing patient-level data, so the saved prediction files (`predictions_*.pkl`) are also kept out of the repository.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

TabNet (`pytorch-tabnet`, `torch`) is optional; without it the TabNet baseline is skipped with a warning. The full pipeline was run on a 32-vCPU, 128 GB machine. For a smoke test, lower `cfg.n_trees`, the `cfg.de_*` population and iteration settings, the worker counts, and set `cfg.run_folds = 1` in the script you run.

## Reproducing the paper

Run from the repository root, in this order. Each step reads the checkpoints written by the earlier ones.

```bash
python scripts/debug_cohort.py        # optional: cohort sizes, mortality, feature counts
python JUCO_Part1_MIMIC.py            # MIMIC-IV: DE search on fold 1, 5-fold evaluation, ablations
python JUCO_Part1B_Robustness.py      # JUCO-XGB, MNAR stress test 10-70%, DCA (reuses frozen θ*)
python JUCO_Part2_eICU.py             # eICU: θ* frozen, every model refit per fold (resumable)
python JUCO_Part3_Aggregate.py        # aggregate tables and tests
python scripts/make_paper_outputs.py  # manuscript figures and tables from out-of-fold predictions
python scripts/cohort_summary.py      # appendix cohort tables
```

Protocol, as in the paper: patient-level non-stratified GroupKFold with 5 folds; within each training fold, 60/20/20 train/validation/calibration; 35 features chosen by mutual information per fold; imputers, feature selection, isotonic calibration and τ are refit in every fold; on eICU only θ\* is carried over from MIMIC-IV. The extra MNAR mask is applied to training and validation splits only, never to test data. Pooled-prediction DeLong tests use Bonferroni correction (α = 0.01 on MIMIC-IV, 0.0083 on eICU); fold-level Wilcoxon tests with five folds cannot go below p = 0.0625.

`scripts/make_paper_outputs.py` takes optional arguments `[results_dir] [figures_dir]`. It needs `results/predictions_MIMIC_IV.pkl`, `results/predictions_eICU.pkl` and `results/stress_results_incremental.csv`.

## Notes and known limits

- The seeds are fixed, but LightGBM, XGBoost and the parallel DE search can differ slightly across hardware and library versions, so re-runs may not reproduce the last decimal.
- On eICU the τ rule never binds, so no cases are deferred there; the abstention curves are still reported.
- The MIMIC-IV deferral rate at τ\* is not saved in the released outputs.

## Citation

If you use this code, please cite the manuscript (see `CITATION.cff`). The paper's status will be updated here once a DOI exists.

## Licence

MIT, see `LICENSE`. The licence covers the code only; the datasets keep their own PhysioNet terms.
