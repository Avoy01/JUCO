# JUCO Mini Project

JUCO is a Joint Uncertainty-Calibrated Optimization framework for ICU mortality
prediction under informative missingness. The project combines a LaTeX
manuscript, clinical data loaders, modelling code, robustness experiments, and
publication figure generation.

The pipeline is designed for MIMIC-IV and eICU data. Raw clinical datasets are
not included in this repository and should not be pushed to GitHub.

## What JUCO Does

The core implementation is in `juco_core.py`. It implements a four-phase
clinical prediction pipeline:

1. Adaptive Quantile Imputation
   Missing values are imputed as lower/upper quantile intervals rather than
   single point estimates.

2. Proportional Fuzzy Transformation
   Imputed intervals are converted into trapezoidal fuzzy numbers so wider
   intervals carry more uncertainty.

3. Fuzzy Decision Forest with Differential Evolution
   A fuzzy forest is trained on the transformed features. Important
   hyperparameters are optimized with a two-stage Differential Evolution search.

4. Safe Abstention Protocol
   High-uncertainty cases can be deferred using entropy-based selective
   prediction. Results include risk-coverage metrics such as AURC and Risk@90.

The main task is in-hospital ICU mortality prediction using the first 24 hours
of each ICU stay.

## Repository Structure

```text
.
|-- juco_core.py                  # Shared data loading, models, metrics, checkpoints
|-- JUCO_Part1_MIMIC.py           # MIMIC-IV primary experiment and ablation study
|-- JUCO_Part1B_Robustness.py     # MIMIC-IV robustness, JUCO-XGB, and DCA experiments
|-- JUCO_Part2_eICU.py            # eICU external validation using frozen MIMIC-IV params
|-- JUCO_Part3_Aggregate.py       # Post-processing tables, statistics, and figures
|-- generate_figures.py           # Standalone paper figure generator
|-- inspect_files.py              # Checks whether required CSV/CSV.GZ files are present
|-- debug_cohort.py               # Loads both datasets and prints cohort statistics
|-- juco_main.tex                 # Main manuscript
|-- references.bib                # Bibliography
|-- cas-*.cls/.sty/.bst           # Elsevier CAS LaTeX template files
|-- Figure*.pdf, fig.pdf          # Manuscript figures
|-- *.drawio                      # Editable diagrams
|-- results/                      # Small CSV/PDF/PNG result summaries tracked in Git
|-- mimic-iv/                     # Local raw MIMIC-IV data, ignored by Git
|-- eicu/                         # Local raw eICU data, ignored by Git
`-- .venv/                        # Local Python environment, ignored by Git
```

## Data Requirements

You need credentialed access to the following datasets:

- MIMIC-IV v2.2 from PhysioNet
- eICU Collaborative Research Database v2.0 from PhysioNet

Place the files in this layout:

```text
mimic-iv/
|-- hosp/
|   |-- admissions.csv.gz
|   |-- patients.csv.gz
|   `-- labevents.csv.gz
`-- icu/
    |-- icustays.csv.gz
    `-- chartevents.csv.gz

eicu/
|-- patient.csv.gz
|-- lab.csv.gz
`-- vitalPeriodic.csv.gz
```

The loaders expect compressed `.csv.gz` files. MIMIC-IV is read partly with
DuckDB streaming, so the very large `chartevents` and `labevents` files do not
need to be manually extracted.

To verify that the files are in the right place, run:

```powershell
python inspect_files.py
```

This prints the detected path, file size, columns, and one sample row for each
required table.

## Installation

Use Python 3.10 or newer. The code was prepared on Windows/PowerShell, but the
same commands work on Linux/macOS with the activation command adjusted.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The TabNet baseline is optional in the code. If `pytorch_tabnet` is missing, the
script prints a warning and disables that baseline. Installing from
`requirements.txt` includes it so the full baseline set can run.

## Hardware Notes

The default settings are intended for a large machine. The comments in
`JUCO_Part1_MIMIC.py` mention a 32-vCPU / 128 GB RAM GCP instance.

Important knobs:

- `cfg.forest_workers`: parallel workers for fuzzy forest training.
- `cfg.de_workers`: parallel Differential Evolution evaluations.
- `cfg.n_trees`: final forest size.
- `cfg.run_folds`: set to `0` for all folds, or a small number for testing.
- `cfg.max_features`: mutual-information feature cap per fold.

For a quick local smoke test, temporarily reduce these values inside the script
you are running:

```python
cfg.n_trees = 20
cfg.de_s1_maxiter = 1
cfg.de_s1_popsize = 2
cfg.de_s2_maxiter = 1
cfg.de_s2_popsize = 2
cfg.forest_workers = 2
cfg.de_workers = 2
cfg.run_folds = 1
```

Use the default settings again for final reported experiments.

## Recommended Run Order

Run the scripts from the repository root.

### 1. Check Dataset Cohorts

This step loads and preprocesses both datasets, then prints cohort size,
mortality rate, feature count, and fold sizes. It does not train models.

```powershell
python debug_cohort.py
```

Expected paths:

- MIMIC-IV: `./mimic-iv/`
- eICU: `./eicu/`
- Outputs: printed to terminal only

### 2. Run MIMIC-IV Primary Experiment

```powershell
python JUCO_Part1_MIMIC.py
```

This script performs:

- MIMIC-IV loading and first-24-hour feature engineering
- patient-level `GroupKFold` preprocessing
- two-stage Differential Evolution on fold 1
- 5-fold evaluation with frozen optimized parameters
- JUCO, JUCO-XGB, XGBoost, LightGBM, MissForest+XGB, MeanImp+XGB, and optional
  TabNet baselines
- a two-fold ablation study

Main outputs in `results/`:

- `checkpoint_mimic.pkl`
- `results_MIMIC_IV.csv`
- `ablation_MIMIC_IV.csv`
- `predictions_MIMIC_IV.pkl`
- `checkpoint_temp_MIMIC-IV.pkl`

The checkpoint is required by Part 1B, Part 2, and Part 3.

### 3. Run MIMIC-IV Robustness and Clinical Utility

Run this after Part 1:

```powershell
python JUCO_Part1B_Robustness.py
```

This script reuses the frozen MIMIC-IV parameters from
`results/checkpoint_mimic.pkl`; it does not repeat the DE search.

It performs:

- JUCO-XGB full 5-fold evaluation
- MNAR missingness stress test from 10% to 70%
- Decision Curve Analysis using saved fold predictions

Main outputs in `results/`:

- `results_JUCO_XGB_MIMIC_IV.csv`
- `missingness_stress_MIMIC_IV.csv`
- `dca_MIMIC_IV.csv`
- `checkpoint_mimic_robustness.pkl`

### 4. Run eICU External Validation

Run this after Part 1:

```powershell
python JUCO_Part2_eICU.py
```

This script loads the frozen parameters from MIMIC-IV and applies them to eICU.
No Differential Evolution search is performed on eICU when
`checkpoint_mimic.pkl` is available. This avoids information leakage and keeps
the external validation protocol clean.

The eICU experiment saves progress after every fold, so a stopped run can resume
from the last completed fold.

Main outputs in `results/`:

- `checkpoint_eicu.pkl`
- `results_eICU.csv`
- `predictions_eICU.pkl`
- `dca_eICU.csv`
- `eicu_exp_progress.pkl` while the run is in progress

After successful completion, the progress file is removed automatically.

### 5. Aggregate Tables, Statistics, and Figures

Run after Parts 1, 1B, and 2:

```powershell
python JUCO_Part3_Aggregate.py
```

This is pure post-processing. It does not train models.

Required inputs in `results/`:

- `results_MIMIC_IV.csv`
- `results_eICU.csv`

Optional inputs used when available:

- `results_JUCO_XGB_MIMIC_IV.csv`
- `ablation_MIMIC_IV.csv`
- `missingness_stress_MIMIC_IV.csv`
- `dca_MIMIC_IV.csv`
- `dca_eICU.csv`
- `checkpoint_mimic.pkl`
- `checkpoint_eicu.pkl`

Main outputs in `results/`:

- `aggregate_results.csv`
- `statistical_tests.csv`
- `metric_comparison.png` and `metric_comparison.pdf`
- `ablation_study.png` and `ablation_study.pdf`
- `fig_missingness.png` and `fig_missingness.pdf`
- `fig_dca.png` and `fig_dca.pdf`

## Standalone Figure Generation

To regenerate the paper-style architecture and summary figures:

```powershell
python generate_figures.py
```

The script reads CSV files from `results/` and writes figures in the repository
root, including `Figure_1.pdf` through `Figure_6.pdf` and `Figure_A1.pdf`.

If Part 3 already produced `results/fig_calibration.pdf`, use that calibration
figure instead of the fallback `Figure_A1.pdf`.

## Manuscript Compilation

Compile the Elsevier CAS manuscript with:

```powershell
pdflatex juco_main.tex
bibtex juco_main
pdflatex juco_main.tex
pdflatex juco_main.tex
```

Important manuscript files:

- `juco_main.tex`
- `references.bib`
- `cas-sc.cls`
- `cas-common.sty`
- `cas-model2-names.bst`
- `Figure*.pdf`

Generated LaTeX files such as `.aux`, `.log`, `.bbl`, `.out`, and
`.synctex.gz` are ignored by Git.

## Output and Git Policy

Tracked files should include source code, manuscript files, small result CSVs,
and publication-ready figures.

Ignored files include:

- raw MIMIC-IV and eICU data
- `.venv/`
- Python caches
- LaTeX build artifacts
- large checkpoints and prediction pickles
- temporary logs
- local scratch folder `New folder/`

Before pushing, check the repo state:

```powershell
git status --short --ignored
```

Only source files and small outputs should appear as tracked or staged files.

## Troubleshooting

`FileNotFoundError` for MIMIC-IV or eICU:
Check that the files match the exact layout in the Data Requirements section.
Then run `python inspect_files.py`.

`WARNING: pytorch_tabnet not installed`:
Install dependencies with `pip install -r requirements.txt`. If you do not need
the TabNet baseline, the warning is safe and the rest of the pipeline can run.

Out-of-memory or very slow runs:
Reduce `cfg.forest_workers`, `cfg.de_workers`, `cfg.n_trees`, and
`cfg.run_folds` for local testing. Use the original values for final results.

Interrupted eICU run:
Run `python JUCO_Part2_eICU.py` again. It should resume from
`results/eicu_exp_progress.pkl`.

Part 3 skips a figure:
That figure depends on an optional CSV. Run the corresponding earlier script,
for example Part 1B for robustness and DCA files.

## GitHub Push

After creating a GitHub repository, connect it and push:

```powershell
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
