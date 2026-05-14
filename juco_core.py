"""
JUCO Framework v2.0 — Core Module
==================================

Joint Uncertainty-Calibrated Optimization (JUCO) for ICU Mortality
Prediction Under Informative Missingness.

This module implements a four-phase clinical prediction pipeline
designed to handle Missing-Not-At-Random (MNAR) patterns that arise
naturally in electronic health records when clinicians order tests
selectively (e.g., lactate measured primarily for sicker patients).

**Four-Phase Pipeline**

1. **Adaptive Quantile Imputation**  — LightGBM quantile regression
   produces interval-valued imputations [q_L, q_U] rather than point
   estimates, preserving epistemic uncertainty from missing data.

2. **Proportional Fuzzy Transformation**  — Imputed intervals are
   mapped to trapezoidal fuzzy numbers (a, b, c, d) whose support
   width encodes feature-level uncertainty.

3. **Fuzzy Decision Forest + Differential Evolution**  — A forest of
   soft-routing decision trees operates directly on fuzzy inputs.
   Tree hyperparameters (alpha_L, alpha_U, eta) are jointly optimised
   via two-stage Differential Evolution (coarse → fine).

4. **Safe Abstention Protocol**  — Entropy-based selective prediction
   defers high-uncertainty cases, producing clinically interpretable
   risk–coverage trade-offs measured by AURC (Area Under the
   Risk-Coverage curve).

**Datasets**: MIMIC-IV v2.2 (Beth Israel Deaconess Medical Center) and
eICU Collaborative Research Database v2.0 (208 US hospitals).

**Shared module** — imported by ``JUCO_Part1_MIMIC.py``,
``JUCO_Part1B_Robustness.py``, ``JUCO_Part2_eICU.py``, and
``JUCO_Part3_Aggregate.py``.

Reference
---------
See accompanying ``JUCO_Framework.ipynb`` for derivations and figures.
"""

# ============================================================================
# 1. Imports
# ============================================================================

# --- JIT compilation (Numba) ---
from numba import jit

# --- Standard library ---
import os, sys, warnings, copy, hashlib, gc, subprocess, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

# --- Numerical computing ---
import numpy as np
import pandas as pd

# --- Visualisation (non-interactive backend for headless servers) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# --- Scikit-learn: cross-validation, preprocessing, metrics, imputation ---
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, brier_score_loss,
    log_loss, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer   # required before IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# --- Gradient-boosted trees ---
import lightgbm as lgb
import xgboost as xgb_lib

# --- Optimisation and statistical testing ---
from scipy.optimize import differential_evolution
from scipy.stats import wilcoxon, friedmanchisquare, norm

# --- Miscellaneous ---
from tqdm.auto import tqdm
import duckdb  # in-process SQL for efficient EHR loading

# --- Deep-learning baseline (optional) ---
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    TabNetClassifier = None
    print("WARNING: pytorch_tabnet not installed. TabNet baseline will be disabled.")

# --- Global settings ---
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

SEED = 42
np.random.seed(SEED)

# ============================================================================
# 2. Global Configuration
# ============================================================================

@dataclass
class JUCOConfig:
    """Central hyperparameter configuration for the JUCO framework.

    All default values match those reported in the paper.  Modify only
    ``mimic_iv_path``, ``eicu_path``, and ``output_dir`` for your
    local environment.

    Parameters are organised by pipeline phase:
      Phase 1 — Quantile Imputation  (alpha_L, alpha_U, LightGBM knobs)
      Phase 2 — Fuzzy Transformation (eta)
      Phase 3 — Fuzzy Decision Forest + DE optimisation
      Phase 4 — Safe Abstention (max_error_rate, coverage_target)
    """

    # -- Paths (update to your local dataset locations) --
    mimic_iv_path: str  = "./mimic-iv/"
    eicu_path: str      = "./eicu/"
    output_dir: str     = "./results/"

    # -- Clinical task --
    target: str              = "mortality"    # prediction target label
    observation_window_h: int = 24           # first-24 h ICU window

    # -- Cross-validation --
    n_folds: int    = 5       # GroupKFold splits (patient-level)
    run_folds: int  = 0       # 0 = run all folds; >0 = limit to first N
    seed: int       = 42

    # -- MNAR masking (applied to train / val only, never test) --
    mnar_beta: float  = -1.0   # logistic slope  (negative → higher values drop more)
    mnar_gamma: float =  0.0   # logistic intercept
    mnar_rate: float  =  0.30  # target fraction of additionally masked entries

    # -- Phase 1: Adaptive Quantile Imputation (LightGBM quantile regression) --
    alpha_L_init: float        = 0.10   # lower quantile (optimised by DE)
    alpha_U_init: float        = 0.90   # upper quantile (optimised by DE)
    lgbm_n_estimators: int     = 300
    lgbm_learning_rate: float  = 0.05
    lgbm_max_depth: int        = 6
    lgbm_min_child_samples: int = 20

    # -- Phase 2: Proportional Fuzzy Transformation --
    eta_init: float = 0.25    # proportion factor mapping intervals → trapezoids

    # -- Phase 3: Fuzzy Decision Forest --
    n_trees: int                   = 180
    max_tree_depth: int            = 6
    min_leaf_samples: int          = 15
    feature_subsample_ratio: float = 0.7   # column subsampling per tree

    # -- Phase 3: Two-Stage Differential Evolution --
    #   Stage 1 — coarse global search (small proxy model)
    de_s1_maxiter: int  = 8
    de_s1_popsize: int  = 8
    de_s1_trees: int    = 20      # proxy forest size
    de_s1_samples: int  = 2000    # training subsample size
    #   Stage 2 — fine local refinement near best region
    de_s2_maxiter: int  = 12
    de_s2_popsize: int  = 12
    de_s2_trees: int    = 40
    de_s2_samples: int  = 7000
    #   Shared DE settings
    de_tol: float                        = 1e-4
    de_patience: int                     = 3        # early-stop if no improvement for N iters
    de_mutation: Tuple[float, float]     = (0.5, 1.0)
    de_recombination: float              = 0.7
    lambda_1: float                      = 0.2      # MPIW penalty weight in fitness
    lambda_2: float                      = 2.0      # coverage-gap penalty weight

    # -- Phase 4: Safe Abstention Protocol --
    max_error_rate: float  = 0.05   # maximum tolerable error on deferred set
    coverage_target: float = 0.90   # desired coverage (fraction of patients retained)

    # -- Three-way split ratios (within each training fold, patient-level) --
    val_ratio: float = 0.20   # validation set  (for DE fitness evaluation)
    cal_ratio: float = 0.20   # calibration set (for post-hoc calibration + abstention)

    # -- Computational budget --
    max_features: int = 35    # mutual-information feature cap per fold

    # -- HPC parallelism --
    de_workers: int     = 8    # Differential Evolution parallel workers
    forest_workers: int = 16   # Joblib workers for forest fitting / prediction

    # -- Feature lists (populated during preprocessing, not set manually) --
    numeric_features: List[str]     = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)


# ============================================================================
# 3. MIMIC-IV Data Loader
# ============================================================================

# MIMIC-IV itemid → feature mapping for chartevents (ICU vitals).
# Each vital sign may have multiple itemids due to different measurement
# devices (e.g., invasive vs. non-invasive blood pressure).
MIMIC_VITALS_ITEMIDS = {
    "heart_rate":   [220045],
    "sbp":          [220050, 220179],          # systolic BP (arterial line / cuff)
    "dbp":          [220051, 220180],          # diastolic BP
    "mbp":          [220052, 220181],          # mean arterial pressure
    "resp_rate":    [220210, 224690],
    "spo2":         [220277],                  # peripheral O₂ saturation
    "temperature":  [223761, 223762],          # °F and °C sources
}

# MIMIC-IV itemid → feature mapping for labevents (hospital lab results).
MIMIC_LABS_ITEMIDS = {
    "lactate":      [50813],       # marker of tissue hypoperfusion
    "creatinine":   [50912],       # renal function
    "bun":          [51006],       # blood urea nitrogen
    "potassium":    [50971],       # electrolyte
    "sodium":       [50983],       # electrolyte
    "glucose":      [50931, 50809],
    "wbc":          [51301],       # white blood cell count
    "hemoglobin":   [51222],
    "platelets":    [51265],       # coagulation indicator
    "bilirubin":    [50885],       # hepatic function (total bilirubin)
}


def load_mimic_iv(cfg: JUCOConfig) -> pd.DataFrame:
    """Load MIMIC-IV data for the first 24 hours of each ICU stay.

    Uses DuckDB streaming to process compressed CSV files without loading
    the full tables into memory.  For each vital sign and lab analyte, nine
    temporal aggregates are computed: min, max, mean, std, first, last,
    delta (last − first), slope (delta / 24 h), and count.

    Parameters
    ----------
    cfg : JUCOConfig
        Configuration object with ``mimic_iv_path`` and
        ``observation_window_h`` attributes.

    Returns
    -------
    pd.DataFrame
        One row per ICU stay with columns: subject_id, stay_id, age,
        gender (binary), mortality (binary label), and ~153 temporal
        features.  Stays with > 50 % missing features are dropped.
    """
    base = cfg.mimic_iv_path
    con = duckdb.connect()

    print("Loading MIMIC-IV core tables...")
    patients = pd.read_csv(os.path.join(base, "hosp", "patients.csv.gz"),
                           usecols=["subject_id", "gender", "anchor_age"])
    admissions = pd.read_csv(os.path.join(base, "hosp", "admissions.csv.gz"),
                             usecols=["subject_id", "hadm_id", "hospital_expire_flag"],
                             dtype={"hospital_expire_flag": int})
    icustays = pd.read_csv(os.path.join(base, "icu", "icustays.csv.gz"),
                           usecols=["subject_id", "hadm_id", "stay_id", "intime"],
                           parse_dates=["intime"])

    stays = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    stays = stays.merge(patients, on="subject_id", how="inner")
    con.register("stays_df", stays[["stay_id", "hadm_id", "intime"]])

    # ---- Vitals from chartevents ----
    chart_path = os.path.join(base, "icu", "chartevents.csv.gz").replace("\\", "/")
    all_vital_ids = [v for ids in MIMIC_VITALS_ITEMIDS.values() for v in ids]
    id_list = ",".join(str(i) for i in all_vital_ids)

    case_parts = []
    for feat, ids in MIMIC_VITALS_ITEMIDS.items():
        id_csv = ",".join(str(i) for i in ids)
        case_parts.append(f"WHEN itemid IN ({id_csv}) THEN '{feat}'")
    case_expr = "CASE " + " ".join(case_parts) + " END"

    if os.path.exists(chart_path.replace("/", os.sep)):
        print("Loading chartevents (vitals) via DuckDB — streaming...")
        vitals_sql = f"""
        SELECT stay_id, feat,
               MIN(valuenum) AS vmin, MAX(valuenum) AS vmax,
               AVG(valuenum) AS vmean, STDDEV_SAMP(valuenum) AS vstd,
               ARG_MIN(valuenum, dt_sec) AS vfirst,
               ARG_MAX(valuenum, dt_sec) AS vlast,
               COUNT(valuenum) AS vcount
        FROM (
            SELECT c.stay_id, c.valuenum,
                   {case_expr} AS feat,
                   EPOCH(c.charttime::TIMESTAMP - s.intime::TIMESTAMP) AS dt_sec
            FROM read_csv_auto('{chart_path}',
                               header=true, delim=',', quote='"',
                               ignore_errors=true, null_padding=true) c
            JOIN stays_df s ON c.stay_id = s.stay_id
            WHERE c.itemid IN ({id_list})
              AND c.valuenum IS NOT NULL
        ) sub
        WHERE dt_sec >= 0 AND dt_sec <= {cfg.observation_window_h * 3600}
        GROUP BY stay_id, feat
        """
        vitals_long = con.execute(vitals_sql).fetchdf()
        print(f"  chartevents done: {len(vitals_long)} (stay, feat) groups")
    else:
        print("  Warning: chartevents not found.")
        vitals_long = pd.DataFrame(columns=["stay_id", "feat", "vmin", "vmax", "vmean", "vstd"])

    vitals_agg = pd.DataFrame(index=vitals_long["stay_id"].unique())
    vitals_agg.index.name = "stay_id"
    for feat in MIMIC_VITALS_ITEMIDS:
        sub = vitals_long[vitals_long["feat"] == feat].set_index("stay_id")
        if len(sub) > 0:
            vitals_agg[f"{feat}_min"] = sub["vmin"]
            vitals_agg[f"{feat}_max"] = sub["vmax"]
            vitals_agg[f"{feat}_mean"] = sub["vmean"]
            vitals_agg[f"{feat}_std"] = sub["vstd"]
            vitals_agg[f"{feat}_first"] = sub["vfirst"]
            vitals_agg[f"{feat}_last"] = sub["vlast"]
            vitals_agg[f"{feat}_delta"] = sub["vlast"] - sub["vfirst"]
            vitals_agg[f"{feat}_slope"] = (sub["vlast"] - sub["vfirst"]) / cfg.observation_window_h
            vitals_agg[f"{feat}_count"] = sub["vcount"]
    del vitals_long; gc.collect()

    # ---- Labs from labevents ----
    lab_path = os.path.join(base, "hosp", "labevents.csv.gz").replace("\\", "/")
    all_lab_ids = [v for ids in MIMIC_LABS_ITEMIDS.values() for v in ids]
    lab_id_list = ",".join(str(i) for i in all_lab_ids)

    lab_case_parts = []
    for feat, ids in MIMIC_LABS_ITEMIDS.items():
        id_csv = ",".join(str(i) for i in ids)
        lab_case_parts.append(f"WHEN itemid IN ({id_csv}) THEN '{feat}'")
    lab_case_expr = "CASE " + " ".join(lab_case_parts) + " END"

    hadm_map = stays[["hadm_id", "stay_id", "intime"]].drop_duplicates("hadm_id")
    con.register("hadm_df", hadm_map)

    if os.path.exists(lab_path.replace("/", os.sep)):
        print("Loading labevents (labs) via DuckDB — streaming...")
        labs_sql = f"""
        SELECT h.stay_id, feat,
               MIN(valuenum) AS vmin, MAX(valuenum) AS vmax,
               AVG(valuenum) AS vmean, STDDEV_SAMP(valuenum) AS vstd,
               ARG_MIN(valuenum, EPOCH(sub.charttime::TIMESTAMP - h.intime::TIMESTAMP)) AS vfirst,
               ARG_MAX(valuenum, EPOCH(sub.charttime::TIMESTAMP - h.intime::TIMESTAMP)) AS vlast,
               COUNT(valuenum) AS vcount
        FROM (
            SELECT l.hadm_id, l.valuenum, l.charttime,
                   {lab_case_expr} AS feat
            FROM read_csv_auto('{lab_path}',
                               header=true, delim=',', quote='"',
                               ignore_errors=true, null_padding=true) l
            WHERE l.itemid IN ({lab_id_list})
              AND l.valuenum IS NOT NULL
        ) sub
        JOIN hadm_df h ON sub.hadm_id = h.hadm_id
        WHERE EPOCH(sub.charttime::TIMESTAMP - h.intime::TIMESTAMP) >= 0
          AND EPOCH(sub.charttime::TIMESTAMP - h.intime::TIMESTAMP) <= {cfg.observation_window_h * 3600}
        GROUP BY h.stay_id, feat
        """
        labs_long = con.execute(labs_sql).fetchdf()
        print(f"  labevents done: {len(labs_long)} (stay, feat) groups")
    else:
        print("  Warning: labevents not found.")
        labs_long = pd.DataFrame(columns=["stay_id", "feat", "vmin", "vmax", "vmean", "vstd"])

    labs_agg = pd.DataFrame(index=labs_long["stay_id"].unique())
    labs_agg.index.name = "stay_id"
    for feat in MIMIC_LABS_ITEMIDS:
        sub = labs_long[labs_long["feat"] == feat].set_index("stay_id")
        if len(sub) > 0:
            labs_agg[f"{feat}_min"] = sub["vmin"]
            labs_agg[f"{feat}_max"] = sub["vmax"]
            labs_agg[f"{feat}_mean"] = sub["vmean"]
            labs_agg[f"{feat}_std"] = sub["vstd"]
            labs_agg[f"{feat}_first"] = sub["vfirst"]
            labs_agg[f"{feat}_last"] = sub["vlast"]
            labs_agg[f"{feat}_delta"] = sub["vlast"] - sub["vfirst"]
            labs_agg[f"{feat}_slope"] = (sub["vlast"] - sub["vfirst"]) / cfg.observation_window_h
            labs_agg[f"{feat}_count"] = sub["vcount"]
    del labs_long; gc.collect()
    con.close()

    # ---- Merge ----
    result = stays[["subject_id", "stay_id", "anchor_age", "gender",
                     "hospital_expire_flag"]].copy()
    result = result.rename(columns={"anchor_age": "age", "hospital_expire_flag": "mortality"})
    result["gender"] = (result["gender"] == "M").astype(int)

    if len(vitals_agg) > 0:
        result = result.merge(vitals_agg, left_on="stay_id", right_index=True, how="left")
    if len(labs_agg) > 0:
        result = result.merge(labs_agg, left_on="stay_id", right_index=True, how="left")

    feat_cols = [c for c in result.columns if c not in ["subject_id", "stay_id", "mortality", "gender"]]
    result = result.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5))
    result = result.reset_index(drop=True)
    gc.collect()

    print(f"MIMIC-IV loaded: {len(result)} ICU stays, "
          f"{result['mortality'].sum()} deaths "
          f"({result['mortality'].mean():.1%} mortality)")
    return result


# ============================================================================
# 4. eICU Data Loader
# ============================================================================

# eICU vital sign column names from vitalPeriodic table.
EICU_VITALS = ["heartrate", "systemicsystolic", "systemicdiastolic",
               "systemicmean", "respiration", "sao2", "temperature"]

# eICU lab analyte names → standardised feature names (aligned with MIMIC-IV).
EICU_LABS_MAP = {
    "lactate":     ["lactate"],
    "creatinine":  ["creatinine"],
    "bun":         ["BUN"],
    "potassium":   ["potassium"],
    "sodium":      ["sodium"],
    "glucose":     ["glucose"],
    "wbc":         ["WBC x 1000"],
    "hemoglobin":  ["Hgb"],
    "platelets":   ["platelets x 1000"],
    "bilirubin":   ["total bilirubin"],
}

# Rename map to align eICU vital names with MIMIC-IV feature names.
EICU_VITAL_RENAME = {
    "heartrate": "heart_rate",
    "systemicsystolic": "sbp",
    "systemicdiastolic": "dbp",
    "systemicmean": "mbp",
    "respiration": "resp_rate",
    "sao2": "spo2",
}


def load_eicu(cfg: JUCOConfig) -> pd.DataFrame:
    """Load eICU Collaborative Research Database (first 24 h per stay).

    Mirrors the feature engineering in ``load_mimic_iv`` to produce
    harmonised feature columns across both datasets.  Column names are
    renamed to match the MIMIC-IV convention (e.g., ``heartrate`` →
    ``heart_rate``) so that a model trained on MIMIC-IV can be applied
    to eICU without schema mismatches.

    Parameters
    ----------
    cfg : JUCOConfig
        Configuration with ``eicu_path`` and ``observation_window_h``.

    Returns
    -------
    pd.DataFrame
        One row per ICU stay with the same column schema as
        ``load_mimic_iv``.
    """
    base = cfg.eicu_path
    window_min = cfg.observation_window_h * 60
    con = duckdb.connect()

    print("Loading eICU patient table...")
    pat = pd.read_csv(os.path.join(base, "patient.csv.gz"),
                       usecols=["patientunitstayid", "uniquepid", "age",
                                "gender", "unitdischargestatus"],
                       low_memory=False)
    pat["age"] = pd.to_numeric(pat["age"].replace("> 89", "90"), errors="coerce")
    pat = pat.dropna(subset=["age"])
    pat["gender"] = (pat["gender"].str.upper() == "MALE").astype(int)
    pat["mortality"] = (pat["unitdischargestatus"] == "Expired").astype(int)
    pat = pat.rename(columns={"uniquepid": "subject_id", "patientunitstayid": "stay_id"})

    # ---- Vitals ----
    vp_path = os.path.join(base, "vitalPeriodic.csv.gz").replace("\\", "/")
    vitals_cols = ", ".join(
        f"MIN({v}) AS {v}_min, MAX({v}) AS {v}_max, "
        f"AVG({v}) AS {v}_mean, STDDEV_SAMP({v}) AS {v}_std, "
        f"ARG_MIN({v}, observationoffset) AS {v}_first, "
        f"ARG_MAX({v}, observationoffset) AS {v}_last, "
        f"COUNT({v}) AS {v}_count"
        for v in EICU_VITALS
    )

    if os.path.exists(vp_path.replace("/", os.sep)):
        print("Loading eICU vitals via DuckDB — streaming...")
        vp_sql = f"""
        SELECT patientunitstayid AS stay_id, {vitals_cols}
        FROM read_csv_auto('{vp_path}', header=true, ignore_errors=true)
        WHERE observationoffset >= 0
          AND observationoffset <= {window_min}
        GROUP BY patientunitstayid
        """
        vitals_df = con.execute(vp_sql).fetchdf().set_index("stay_id")
        print(f"  vitalPeriodic done: {len(vitals_df)} stays")
        # Compute trend features from first/last values
        for v in EICU_VITALS:
            if f"{v}_first" in vitals_df.columns and f"{v}_last" in vitals_df.columns:
                vitals_df[f"{v}_delta"] = vitals_df[f"{v}_last"] - vitals_df[f"{v}_first"]
                vitals_df[f"{v}_slope"] = (vitals_df[f"{v}_last"] - vitals_df[f"{v}_first"]) / cfg.observation_window_h
    else:
        print("  Warning: vitalPeriodic not found.")
        vitals_df = pd.DataFrame()

    # ---- Labs ----
    lab_path = os.path.join(base, "lab.csv.gz").replace("\\", "/")
    lab_case_parts = []
    all_lab_names = []
    for feat, names in EICU_LABS_MAP.items():
        for name in names:
            lab_case_parts.append(f"WHEN labname = '{name}' THEN '{feat}'")
            all_lab_names.append(name)
    lab_case_expr = "CASE " + " ".join(lab_case_parts) + " END"
    lab_name_filter = ", ".join(f"'{n}'" for n in all_lab_names)

    if os.path.exists(lab_path.replace("/", os.sep)):
        print("Loading eICU labs via DuckDB — streaming...")
        lab_sql = f"""
        SELECT patientunitstayid AS stay_id, feat,
               MIN(val) AS vmin, MAX(val) AS vmax,
               AVG(val) AS vmean, STDDEV_SAMP(val) AS vstd,
               ARG_MIN(val, labresultoffset) AS vfirst,
               ARG_MAX(val, labresultoffset) AS vlast,
               COUNT(val) AS vcount
        FROM (
            SELECT patientunitstayid, labresultoffset,
                   TRY_CAST(labresult AS DOUBLE) AS val,
                   {lab_case_expr} AS feat
            FROM read_csv_auto('{lab_path}', header=true, ignore_errors=true)
            WHERE labname IN ({lab_name_filter})
              AND labresultoffset >= 0
              AND labresultoffset <= {window_min}
        ) sub
        WHERE val IS NOT NULL AND feat IS NOT NULL
        GROUP BY patientunitstayid, feat
        """
        labs_long = con.execute(lab_sql).fetchdf()
        print(f"  lab done: {len(labs_long)} (stay, feat) groups")

        labs_df = pd.DataFrame(index=labs_long["stay_id"].unique())
        labs_df.index.name = "stay_id"
        for feat in EICU_LABS_MAP:
            sub = labs_long[labs_long["feat"] == feat].set_index("stay_id")
            if len(sub) > 0:
                labs_df[f"{feat}_min"] = sub["vmin"]
                labs_df[f"{feat}_max"] = sub["vmax"]
                labs_df[f"{feat}_mean"] = sub["vmean"]
                labs_df[f"{feat}_std"] = sub["vstd"]
                labs_df[f"{feat}_first"] = sub["vfirst"]
                labs_df[f"{feat}_last"] = sub["vlast"]
                labs_df[f"{feat}_delta"] = sub["vlast"] - sub["vfirst"]
                labs_df[f"{feat}_slope"] = (sub["vlast"] - sub["vfirst"]) / cfg.observation_window_h
                labs_df[f"{feat}_count"] = sub["vcount"]
        del labs_long
    else:
        print("  Warning: lab table not found.")
        labs_df = pd.DataFrame()

    con.close()
    gc.collect()

    # ---- Merge ----
    result = pat[["subject_id", "stay_id", "age", "gender", "mortality"]].copy()
    if len(vitals_df) > 0:
        result = result.merge(vitals_df, left_on="stay_id", right_index=True, how="left")
    if len(labs_df) > 0:
        result = result.merge(labs_df, left_on="stay_id", right_index=True, how="left")

    rename_map = {}
    for old_pfx, new_pfx in EICU_VITAL_RENAME.items():
        for stat in ["min", "max", "mean", "std", "first", "last", "delta", "slope", "count"]:
            old_col = f"{old_pfx}_{stat}"
            new_col = f"{new_pfx}_{stat}"
            if old_col in result.columns:
                rename_map[old_col] = new_col
    result = result.rename(columns=rename_map)

    feat_cols = [c for c in result.columns if c not in ["subject_id", "stay_id", "mortality", "gender"]]
    result = result.dropna(subset=feat_cols, thresh=int(len(feat_cols) * 0.5))
    result = result.reset_index(drop=True)
    gc.collect()

    print(f"eICU loaded: {len(result)} ICU stays, "
          f"{result['mortality'].sum()} deaths "
          f"({result['mortality'].mean():.1%} mortality)")
    return result


# ============================================================================
# 5. Preprocessing Utilities
# ============================================================================

def apply_mnar_mask(X, beta=-1.0, gamma=0.0, target_rate=0.30, seed=42):
    """Simulate informative (MNAR) missingness via logistic dropout.

    In real ICU data, missingness is informative: sicker patients tend
    to have certain labs ordered more frequently while healthier ones
    are missing those values.  This function artificially amplifies
    that pattern by masking observed values with probability:

        P(mask | x) = sigma(beta * x_norm + gamma)

    where x_norm is min–max normalised.  With beta < 0, higher-valued
    observations are *less* likely to be masked, mimicking the clinical
    pattern where abnormal (high) values trigger more tests.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix (may already contain NaNs from natural missingness).
    beta : float
        Logistic slope controlling missingness direction.
    gamma : float
        Logistic intercept.
    target_rate : float
        Desired fraction of *additionally* masked entries.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    X_masked : np.ndarray
        Copy of X with additional NaN entries.
    mask : np.ndarray[bool]
        Boolean mask indicating which entries were artificially dropped.
    """
    rng = np.random.RandomState(seed)
    X_masked = X.copy()
    mask = np.zeros_like(X, dtype=bool)

    for j in range(X.shape[1]):
        col = X[:, j]
        observed = ~np.isnan(col)
        if observed.sum() == 0:
            continue
        obs_vals = col[observed]
        col_min, col_max = obs_vals.min(), obs_vals.max()
        if col_max - col_min < 1e-8:
            continue
        normalized = (obs_vals - col_min) / (col_max - col_min)
        logit = beta * normalized + gamma
        prob = 1.0 / (1.0 + np.exp(-logit))
        if prob.mean() > 0:
            prob = prob * (target_rate / prob.mean())
        prob = np.clip(prob, 0.0, 0.95)
        drop = rng.binomial(1, prob).astype(bool)
        obs_indices = np.where(observed)[0]
        mask[obs_indices[drop], j] = True
        X_masked[obs_indices[drop], j] = np.nan

    actual_rate = mask.sum() / max(1, (~np.isnan(X)).sum())
    print(f"  MNAR mask: target={target_rate:.2f}, actual={actual_rate:.3f}")
    return X_masked, mask


def compute_sample_weights(y, verbose=True):
    """Compute inverse-frequency sample weights for class imbalance.

    ICU mortality is a rare event (~10–12 % prevalence).  Assigning
    weight w_k = N / (K * n_k) to each class k ensures that the
    minority class (deaths) contributes proportionally more to the
    training loss.

    Returns
    -------
    sample_weights : np.ndarray
        Per-sample weight vector.
    weight_dict : dict
        {class_label: weight} mapping for tree-based learners.
    """
    classes, counts = np.unique(y.astype(int), return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weights = n_samples / (n_classes * counts)
    weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    sample_weights = np.array([weight_dict[int(yi)] for yi in y])
    if verbose:
        print(f"  Class weights: {weight_dict}")
    return sample_weights, weight_dict


def three_way_split(X, y, groups, train_idx, val_ratio=0.20, cal_ratio=0.20, seed=42):
    """Split a training fold into train / validation / calibration sets.

    Splitting is performed at the **patient level** (using ``groups``)
    to prevent data leakage across different ICU stays of the same
    patient.

    - **Validation set**: used by DE fitness evaluation (Phase 3).
    - **Calibration set**: used for post-hoc isotonic calibration and
      abstention threshold tuning (Phase 4).
    """
    rng = np.random.RandomState(seed)
    fold_groups = groups[train_idx]
    unique_subjects = np.unique(fold_groups)
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)
    n_val = max(1, int(n * val_ratio))
    n_cal = max(1, int(n * cal_ratio))

    val_subjects = set(unique_subjects[:n_val])
    cal_subjects = set(unique_subjects[n_val:n_val + n_cal])
    tr_subjects = set(unique_subjects[n_val + n_cal:])

    fold_X = X[train_idx]
    fold_y = y[train_idx]

    tr_mask = np.array([g in tr_subjects for g in fold_groups])
    val_mask = np.array([g in val_subjects for g in fold_groups])
    cal_mask = np.array([g in cal_subjects for g in fold_groups])

    return {
        "X_tr": fold_X[tr_mask], "y_tr": fold_y[tr_mask],
        "X_val": fold_X[val_mask], "y_val": fold_y[val_mask],
        "X_cal": fold_X[cal_mask], "y_cal": fold_y[cal_mask],
    }


def select_top_features(X, y, feature_names, max_features=25, seed=42):
    """Rank features by mutual information and keep the top K.

    Feature selection is performed per fold using only training data to
    avoid information leakage.  Mutual information is estimated via
    k-nearest-neighbours (k = 5); NaNs are temporarily median-imputed
    for the ranking step only.
    """
    from sklearn.feature_selection import mutual_info_classif
    X_temp = X.copy()
    for j in range(X_temp.shape[1]):
        m = np.isnan(X_temp[:, j])
        if m.any():
            med = np.nanmedian(X_temp[:, j])
            X_temp[m, j] = med if not np.isnan(med) else 0.0
    mi_scores = mutual_info_classif(X_temp, y, random_state=seed, n_neighbors=5)
    top_idx = np.argsort(mi_scores)[-max_features:]
    top_idx = np.sort(top_idx)
    selected_names = [feature_names[i] for i in top_idx]
    print(f"  Feature selection: {len(feature_names)} -> {len(selected_names)} (top {max_features} by MI)")
    for rank, idx in enumerate(np.argsort(mi_scores)[-5:][::-1]):
        print(f"    Top {rank+1}: {feature_names[idx]} (MI={mi_scores[idx]:.4f})")
    return top_idx, selected_names


def preprocess_dataset(df, cfg):
    """Convert a loaded DataFrame into feature arrays and fold indices.

    Steps
    -----
    1. Separate metadata (subject_id, stay_id, mortality) from features.
    2. Build binary missingness indicator features for columns with
       > 5 % missing (informative in EHR data — carries clinical signal).
    3. Create patient-level GroupKFold splits.

    Feature selection is deferred to ``run_experiment`` to avoid leakage.
    """
    meta_cols = {"subject_id", "stay_id", "mortality"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    categorical_cols = ["gender"] if "gender" in feature_cols else []
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    X = df[feature_cols].to_numpy(dtype=np.float64, na_value=np.nan)
    y = df["mortality"].to_numpy(dtype=int, na_value=0)
    groups = df["subject_id"].values

    # --- Missingness indicators (informative in EHR data) ---
    miss_mask = np.isnan(X).astype(np.float64)
    miss_rate = np.mean(miss_mask, axis=0)
    informative = miss_rate > 0.05  # only features with >5% missing
    if informative.any():
        miss_features = miss_mask[:, informative]
        miss_names = [f"{feature_cols[j]}_missing" for j in range(len(feature_cols)) if informative[j]]
        X = np.hstack([X, miss_features])
        feature_cols = feature_cols + miss_names
        numeric_cols = numeric_cols + miss_names
        print(f"  Added {len(miss_names)} missingness indicator features")

    # NOTE: Feature selection is done INSIDE each fold (no leakage).
    # preprocess_dataset returns ALL features; run_experiment selects per fold.

    gkf = GroupKFold(n_splits=cfg.n_folds)
    folds = list(gkf.split(X, y, groups=groups))

    print(f"  Features       : {len(feature_cols)} ({len(numeric_cols)} num, {len(categorical_cols)} cat)")
    print(f"  Samples        : {len(y)}")
    print(f"  Mortality rate : {y.mean():.3f}")
    print(f"  Unique patients: {len(np.unique(groups))}")
    print(f"  Folds          : {cfg.n_folds} (patient-level GroupKFold)")

    return {
        "X_raw": X,
        "y": y,
        "groups": groups,
        "feature_names": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "folds": folds,
    }


# ============================================================================
# 6. Phase 1 — Adaptive Quantile Imputer
# ============================================================================

class AdaptiveQuantileImputer:
    """LightGBM-based quantile regression imputer (Phase 1).

    For each feature column *j* with missing values, three separate
    LightGBM models are fitted to predict quantiles of the conditional
    distribution  P(x_j | X_{-j}):

        - q_L  = alpha_L   quantile  (lower bound)
        - q_M  = 0.50      quantile  (median / point estimate)
        - q_U  = alpha_U   quantile  (upper bound)

    The interval [q_L, q_U] captures imputation uncertainty: wider
    intervals indicate features where the model is less certain about
    the true value.  This uncertainty is propagated downstream through
    the fuzzy transformation (Phase 2).

    Cost-sensitive sample weighting ensures that the rare-event class
    (mortality) receives proportionally higher influence during
    imputer training, which is critical for calibrated risk estimation.

    Parameters
    ----------
    alpha_L : float
        Lower quantile level (optimised by DE in Phase 3).
    alpha_U : float
        Upper quantile level.
    n_estimators, learning_rate, max_depth, min_child_samples : int/float
        LightGBM hyperparameters for the per-feature quantile models.
    seed : int
        Random state for reproducibility.
    """

    def __init__(self, alpha_L=0.10, alpha_U=0.90, n_estimators=300,
                 learning_rate=0.05, max_depth=6, min_child_samples=20, seed=42):
        self.alpha_L = alpha_L
        self.alpha_U = alpha_U
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.seed = seed
        self.models = {}
        self._col_medians = None

    def fit(self, X, y_labels=None, class_weight_dict=None, verbose=False):
        n_samples, n_features = X.shape
        self._col_medians = np.nanmedian(X, axis=0)
        self.models = {}

        X_filled = X.copy()
        for j in range(n_features):
            m = np.isnan(X_filled[:, j])
            if m.any():
                X_filled[m, j] = self._col_medians[j]

        for j in range(n_features):
            nan_mask = np.isnan(X[:, j])
            if nan_mask.sum() == 0:
                continue
            obs_mask = ~nan_mask
            if obs_mask.sum() < 30:
                continue

            pred_idx = [k for k in range(n_features) if k != j]
            X_train = X_filled[obs_mask][:, pred_idx]
            y_train = X[obs_mask, j]

            sw = None
            if y_labels is not None and class_weight_dict is not None:
                sw = np.array([class_weight_dict.get(int(yi), 1.0)
                               for yi in y_labels[obs_mask]])

            self.models[j] = {}
            for quantile in [self.alpha_L, 0.5, self.alpha_U]:
                params = {
                    "objective": "quantile",
                    "alpha": quantile,
                    "n_estimators": self.n_estimators,
                    "learning_rate": self.learning_rate,
                    "max_depth": self.max_depth,
                    "min_child_samples": self.min_child_samples,
                    "random_state": self.seed,
                    "verbosity": -1,
                    "n_jobs": 1,
                }
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, sample_weight=sw)
                self.models[j][quantile] = model

            if verbose:
                print(f"  Feature {j}: {nan_mask.sum()} missing, {obs_mask.sum()} train samples")
        return self

    def predict_intervals(self, X):
        n_samples, n_features = X.shape
        L = np.full_like(X, np.nan)
        M = np.full_like(X, np.nan)
        U = np.full_like(X, np.nan)

        X_filled = X.copy()
        for j in range(n_features):
            m = np.isnan(X_filled[:, j])
            if m.any() and self._col_medians is not None:
                X_filled[m, j] = self._col_medians[j]

        for j, qmodels in self.models.items():
            nan_mask = np.isnan(X[:, j])
            if nan_mask.sum() == 0:
                continue
            pred_idx = [k for k in range(n_features) if k != j]
            X_pred = X_filled[nan_mask][:, pred_idx]
            L[nan_mask, j] = qmodels[self.alpha_L].predict(X_pred)
            M[nan_mask, j] = qmodels[0.5].predict(X_pred)
            U[nan_mask, j] = qmodels[self.alpha_U].predict(X_pred)
            L[nan_mask, j] = np.minimum(L[nan_mask, j], M[nan_mask, j])
            U[nan_mask, j] = np.maximum(U[nan_mask, j], M[nan_mask, j])
        return L, M, U

    def impute_median(self, X):
        _, M, _ = self.predict_intervals(X)
        X_imp = X.copy()
        for j in self.models:
            m = np.isnan(X[:, j])
            if m.any():
                X_imp[m, j] = M[m, j]
        for j in range(X_imp.shape[1]):
            m = np.isnan(X_imp[:, j])
            if m.any() and self._col_medians is not None:
                X_imp[m, j] = self._col_medians[j]
        return X_imp


# ============================================================================
# 7. Phase 2 — Proportional Fuzzy Transformation
# ============================================================================

class TrapezoidalFuzzyNumber:
    """Trapezoidal fuzzy number T = (a, b, c, d) with  a <= b <= c <= d.

    Membership function:
        mu(x) = 0                              if x < a or x > d
        mu(x) = (x - a) / (b - a)             if a <= x < b  (left ramp)
        mu(x) = 1                              if b <= x <= c (core)
        mu(x) = (d - x) / (d - c)             if c < x <= d  (right ramp)

    Properties:
        centroid    = (a + b + c + d) / 4
        uncertainty = d - a   (support width — encodes imputation confidence)
    """
    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d

    def membership(self, x):
        if x < self.a or x > self.d: return 0.0
        if self.a <= x < self.b: return (x - self.a) / (self.b - self.a + 1e-12)
        if self.b <= x <= self.c: return 1.0
        if self.c < x <= self.d: return (self.d - x) / (self.d - self.c + 1e-12)
        return 0.0

    @property
    def centroid(self): return (self.a + self.b + self.c + self.d) / 4.0

    @property
    def uncertainty(self): return self.d - self.a


class TriangularFuzzySet:
    """Triangular fuzzy set used for split thresholds in the fuzzy tree."""
    __slots__ = ("left", "center", "right")

    def __init__(self, left, center, right):
        self.left, self.center, self.right = left, center, right


class ProportionalFuzzyTransformer:
    """Map quantile-imputed intervals [L, M, U] to trapezoidal fuzzy numbers.

    For each missing entry, the transformer constructs a trapezoid
    (a, b, c, d) where:
        a = q_L              (left foot  — lower quantile)
        b = max(a, M − eta * interval_width)
        c = min(d, M + eta * interval_width)
        d = q_U              (right foot — upper quantile)

    The parameter ``eta`` controls how much of the interval is allocated
    to the flat core [b, c] versus the sloped shoulders.  Smaller eta
    produces larger flat cores (closer to interval-valued imputation);
    larger eta produces more peaked trapezoids (closer to point imputation).

    For observed (non-missing) entries, the trapezoid degenerates to a
    crisp value: a = b = c = d = x_observed.

    Parameters
    ----------
    eta : float
        Proportion factor, clipped to [0, 0.5].
    """

    def __init__(self, eta=0.25):
        self.eta = np.clip(eta, 0.0, 0.5)

    def transform_to_arrays(self, X, L, M, U):
        """Vectorised transformation returning four arrays (a, b, c, d)."""
        n, p = X.shape
        a_arr = np.copy(X)
        b_arr = np.copy(X)
        c_arr = np.copy(X)
        d_arr = np.copy(X)
        nan_mask = np.isnan(X)

        # Safe fallback: where interval bounds are NaN, collapse to available values
        l_safe = np.where(np.isnan(L), 0.0, L)
        m_safe = np.where(np.isnan(M), l_safe, M)
        u_safe = np.where(np.isnan(U), m_safe, U)

        sigma = self.eta * (u_safe - l_safe)   # shoulder width

        a_arr[nan_mask] = l_safe[nan_mask]
        d_arr[nan_mask] = u_safe[nan_mask]
        b_arr[nan_mask] = np.maximum(l_safe[nan_mask], m_safe[nan_mask] - sigma[nan_mask])
        c_arr[nan_mask] = np.minimum(u_safe[nan_mask], m_safe[nan_mask] + sigma[nan_mask])

        # Enforce a <= b <= c <= d
        b_arr = np.maximum(a_arr, np.minimum(b_arr, d_arr))
        c_arr = np.maximum(b_arr, np.minimum(c_arr, d_arr))
        return a_arr, b_arr, c_arr, d_arr


# ============================================================================
# 8. Phase 3 — Fuzzy Decision Tree / Forest
# ============================================================================

def _compute_fuzzy_membership_fast(a, b, c, d, t_center, t_width):
    """Centroid-based soft routing for fuzzy split nodes.

    Instead of hard left/right partitioning, each sample receives a
    continuous membership degree in [0, 1] for each child.  The
    membership is computed via a sigmoid centred on the split threshold,
    with sharpness inversely proportional to the combined uncertainty
    of the fuzzy number and the threshold:

        sharpness = 5 / (uncertainty + t_width + eps)
        mu_left   = sigmoid(sharpness * (t_center - centroid))
        mu_right  = 1 - mu_left

    **Interpretation**: samples with crisp (observed) features route
    near-deterministically, while uncertain (imputed) samples split
    their weight across both children — naturally down-weighting them
    in the final vote.
    """
    centroid = (a + b + c + d) / 4.0
    uncertainty = d - a                               # support width of the trapezoid
    sharpness = 5.0 / (uncertainty + t_width + 1e-12) # inverse uncertainty → sharpness
    z = sharpness * (t_center - centroid)
    mu_left = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    mu_right = 1.0 - mu_left
    return mu_left, mu_right


class FuzzyTreeNode:
    """Internal/leaf node for the fuzzy decision tree."""

    def __init__(self):
        self.feature_idx        = None   # split feature index (None at leaves)
        self.threshold          = None   # TriangularFuzzySet split threshold
        self.left               = None   # left child node
        self.right              = None   # right child node
        self.class_distribution = None   # normalised class probabilities at leaf
        self.is_leaf            = False
        self.depth              = 0
        self.n_samples          = 0.0    # effective (weighted) sample count


class FuzzyDecisionTree:
    """Single soft-routing decision tree operating on trapezoidal fuzzy inputs.

    Each internal node splits samples using soft membership degrees
    rather than hard binary partitions.  At prediction time the tree
    traverses both branches simultaneously, weighting leaf contributions
    by the cumulative membership product along the path.
    """

    def __init__(self, max_depth=6, min_leaf_samples=10, n_classes=2,
                 feature_subset=None, seed=42):
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.n_classes = n_classes
        self.feature_subset = feature_subset
        self.seed = seed
        self.root = None
        self._rng = np.random.RandomState(seed)

    def _fuzzy_entropy(self, y, weights):
        total_w = weights.sum()
        if total_w < 1e-12: return 0.0
        entropy = 0.0
        for c in range(self.n_classes):
            p = weights[y == c].sum() / (total_w + 1e-12)
            if p > 1e-12:
                entropy -= p * np.log2(p + 1e-12)
        return entropy

    def _fuzzy_information_gain(self, y, weights, mu_left, mu_right):
        parent_entropy = self._fuzzy_entropy(y, weights)
        w_left = weights * mu_left
        w_right = weights * mu_right
        total_w = weights.sum()
        child_entropy = (
            (w_left.sum() / (total_w + 1e-12)) * self._fuzzy_entropy(y, w_left)
            + (w_right.sum() / (total_w + 1e-12)) * self._fuzzy_entropy(y, w_right)
        )
        return parent_entropy - child_entropy

    def _find_best_split(self, a, b, c, d, y, weights, feature_indices):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        for feat_idx in feature_indices:
            centroids = (a[:, feat_idx] + b[:, feat_idx] + c[:, feat_idx] + d[:, feat_idx]) / 4.0
            candidates = np.unique(np.percentile(centroids, [15, 30, 45, 50, 55, 70, 85]))
            global_std = np.std(centroids) + 1e-12
            for split_val in candidates:
                local_std = np.std(centroids[np.abs(centroids - split_val) < global_std])
                width = max(local_std * 0.5, 1e-6)
                mu_left, mu_right = _compute_fuzzy_membership_fast(
                    a[:, feat_idx], b[:, feat_idx], c[:, feat_idx], d[:, feat_idx],
                    split_val, width,
                )
                if (weights * mu_left).sum() < self.min_leaf_samples: continue
                if (weights * mu_right).sum() < self.min_leaf_samples: continue
                gain = self._fuzzy_information_gain(y, weights, mu_left, mu_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = (split_val, width)  # store as tuple
        return best_feature, best_threshold, best_gain

    def _build_tree(self, a, b, c, d, y, weights, depth):
        node = FuzzyTreeNode()
        node.depth = depth
        node.n_samples = weights.sum()
        dist = np.zeros(self.n_classes)
        for cls in range(self.n_classes):
            dist[cls] = weights[y == cls].sum()
        node.class_distribution = dist / (dist.sum() + 1e-12)

        if (depth >= self.max_depth or node.n_samples < self.min_leaf_samples * 2
                or np.max(node.class_distribution) > 0.99):
            node.is_leaf = True
            return node

        feat_indices = self.feature_subset if self.feature_subset else list(range(a.shape[1]))
        best_feat, best_thresh, best_gain = self._find_best_split(a, b, c, d, y, weights, feat_indices)

        if best_feat is None or best_gain <= 0:
            node.is_leaf = True
            return node

        node.feature_idx = best_feat
        node.threshold = best_thresh  # now a (center, width) tuple
        mu_left, mu_right = _compute_fuzzy_membership_fast(
            a[:, best_feat], b[:, best_feat], c[:, best_feat], d[:, best_feat],
            best_thresh[0], best_thresh[1],
        )
        node.left = self._build_tree(a, b, c, d, y, weights * mu_left, depth + 1)
        node.right = self._build_tree(a, b, c, d, y, weights * mu_right, depth + 1)
        return node

    def fit(self, a, b, c, d, y, sample_weights=None):
        if sample_weights is None:
            sample_weights = np.ones(len(y))
        self.root = self._build_tree(a, b, c, d, y, sample_weights, depth=0)
        return self

    def _predict_batch(self, node, a, b, c, d, memberships):
        n = len(memberships)
        if node.is_leaf or n == 0:
            return memberships[:, None] * node.class_distribution[None, :]
        feat = node.feature_idx
        th = node.threshold  # (center, width) tuple
        mu_l, mu_r = _compute_fuzzy_membership_fast(
            a[:, feat], b[:, feat], c[:, feat], d[:, feat],
            th[0], th[1],
        )
        proba = np.zeros((n, self.n_classes))
        m_l = memberships * mu_l
        m_r = memberships * mu_r
        active_l = m_l > 1e-6
        active_r = m_r > 1e-6
        if active_l.any():
            idx = np.where(active_l)[0]
            proba[idx] += self._predict_batch(node.left, a[idx], b[idx], c[idx], d[idx], m_l[idx])
        if active_r.any():
            idx = np.where(active_r)[0]
            proba[idx] += self._predict_batch(node.right, a[idx], b[idx], c[idx], d[idx], m_r[idx])
        return proba

    def predict_proba(self, a, b, c, d):
        n = a.shape[0]
        memberships = np.ones(n)
        proba = self._predict_batch(self.root, a, b, c, d, memberships)
        row_sums = proba.sum(axis=1, keepdims=True) + 1e-12
        return proba / row_sums


class FuzzyDecisionForest:
    """Ensemble of fuzzy decision trees with balanced bootstrap.

    Aggregation follows the standard random-forest averaging rule.
    Column subsampling (``feature_subsample_ratio``) decorrelates trees.
    Balanced bootstrap ensures equal class representation per tree to
    counteract the ~10 % mortality prevalence.

    For small forests (n_trees <= 3) used during DE proxy evaluation,
    trees are built sequentially to avoid joblib overhead.  For the
    final full forest, parallel fitting is used.
    """

    def __init__(self, n_trees=50, max_depth=6, min_leaf_samples=10,
                 feature_subsample_ratio=0.7, n_classes=2, seed=42,
                 forest_workers=16, balanced_bootstrap=True):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples
        self.feature_subsample_ratio = feature_subsample_ratio
        self.n_classes = n_classes
        self.seed = seed
        self.forest_workers = forest_workers
        self.balanced_bootstrap = balanced_bootstrap
        self.trees = []
        self.feature_subsets = []

    def fit(self, a, b, c, d, y, sample_weights=None, verbose=False):
        from joblib import Parallel, delayed

        n_samples, n_features = a.shape
        n_sub = max(1, int(n_features * self.feature_subsample_ratio))
        rng = np.random.RandomState(self.seed)

        tree_args = []
        self.feature_subsets = []

        for t in range(self.n_trees):
            feat_sub = sorted(rng.choice(n_features, n_sub, replace=False))
            self.feature_subsets.append(feat_sub)
            # Balanced bootstrap: equal samples from each class
            if self.balanced_bootstrap:
                pos_idx = np.where(y == 1)[0]
                neg_idx = np.where(y == 0)[0]
                n_per_class = max(len(pos_idx), len(neg_idx))
                boot_pos = rng.choice(pos_idx, n_per_class, replace=True)
                boot_neg = rng.choice(neg_idx, n_per_class, replace=True)
                boot_idx = np.concatenate([boot_pos, boot_neg])
                rng.shuffle(boot_idx)
            else:
                boot_idx = rng.choice(n_samples, n_samples, replace=True)
            sw_boot = sample_weights[boot_idx] if sample_weights is not None else None
            tree_args.append((t, feat_sub, boot_idx, sw_boot))

        def _fit_single_tree(args):
            t, feat_sub, boot_idx, sw_boot = args
            tree = FuzzyDecisionTree(
                max_depth=self.max_depth,
                min_leaf_samples=self.min_leaf_samples,
                n_classes=self.n_classes,
                feature_subset=list(range(n_sub)),
                seed=self.seed + t,
            )
            tree.fit(
                a[boot_idx][:, feat_sub],
                b[boot_idx][:, feat_sub],
                c[boot_idx][:, feat_sub],
                d[boot_idx][:, feat_sub],
                y[boot_idx],
                sample_weights=sw_boot,
            )
            return tree

        # Sequential mode for DE safety (n_trees <= 3)
        # Parallel mode for final training (n_trees > 3)
        if self.n_trees <= 3:
            self.trees = [_fit_single_tree(args) for args in tree_args]
        else:
            if verbose:
                print(f"  -> Building {self.n_trees} trees in parallel (n_jobs={self.forest_workers})...")
            self.trees = Parallel(n_jobs=self.forest_workers, backend="loky")(
                delayed(_fit_single_tree)(args) for args in tree_args
            )

        return self

    def predict_proba(self, a, b, c, d):
        proba = np.zeros((a.shape[0], self.n_classes))
        for tree, fs in zip(self.trees, self.feature_subsets):
            proba += tree.predict_proba(a[:, fs], b[:, fs], c[:, fs], d[:, fs])
        return proba / len(self.trees)

    def predict(self, a, b, c, d):
        return np.argmax(self.predict_proba(a, b, c, d), axis=1)


# ============================================================================
# 9. Phase 3 — Two-Stage Differential Evolution Optimiser
# ============================================================================

class JUCOOptimizer:
    """Joint optimisation of quantile, fuzzy, and tree hyperparameters.

    Optimises five parameters simultaneously via ``scipy.optimize.
    differential_evolution``:

        alpha_L          — lower quantile level   (Phase 1)
        alpha_U          — upper quantile level   (Phase 1)
        eta              — fuzzy proportion factor (Phase 2)
        max_depth        — tree depth             (Phase 3)
        min_leaf_samples — leaf size guard        (Phase 3)

    **Two-stage schedule** (coarse → fine):
      - Stage 1 uses a small proxy forest and subsampled data for a
        fast global search across the full parameter space.
      - Stage 2 seeds the population around the Stage-1 optimum and
        evaluates with a larger forest / more data for accurate local
        refinement.

    **Composite fitness function**:
        L_total = L_class + lambda_1 * MPIW_norm + lambda_2 * coverage_gap

    where L_class is log-loss on the validation set, MPIW_norm is the
    normalised mean prediction interval width (penalises overly wide
    intervals), and coverage_gap = max(0, 0.95 − PICP) penalises
    intervals that fail to cover the true values.

    DE is run **once on Fold 1** and the resulting parameters are frozen
    for all subsequent folds to avoid per-fold overfitting.
    """

    BOUNDS = [
        (0.05, 0.35),   # alpha_L: reasonable lower quantile
        (0.65, 0.95),   # alpha_U: reasonable upper quantile
        (0.1, 0.5),     # eta
        (4, 6),          # max_depth: fuzzy trees overfit beyond 6
        (15, 50),        # min_leaf_samples: prevent overfitting
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.history = []
        self.best_params = None

    def _composite_fitness(self, params, X_train, y_train, sw_train,
                           X_val, y_val, X_val_masked, val_mask, X_val_true):
        alpha_L, alpha_U, eta, max_depth_f, min_leaf_f = params
        max_depth = int(round(max_depth_f))
        min_leaf = int(round(min_leaf_f))
        n_trees = self._current_n_trees
        max_n = self._current_max_samples

        try:
            n_tr = len(y_train)
            if n_tr > max_n:
                rng_sub = np.random.RandomState(self.cfg.seed)
                sub_idx = rng_sub.choice(n_tr, max_n, replace=False)
                X_train_s, y_train_s, sw_train_s = X_train[sub_idx], y_train[sub_idx], sw_train[sub_idx]
            else:
                X_train_s, y_train_s, sw_train_s = X_train, y_train, sw_train

            imputer = AdaptiveQuantileImputer(
                alpha_L=alpha_L, alpha_U=alpha_U,
                n_estimators=50, learning_rate=0.1, max_depth=4,
                min_child_samples=self.cfg.lgbm_min_child_samples, seed=self.cfg.seed,
            )
            _, cw = compute_sample_weights(y_train_s, verbose=False)
            imputer.fit(X_train_s, y_labels=y_train_s, class_weight_dict=cw)

            n_val = len(y_val)
            max_val = min(2000, max_n)
            if n_val > max_val:
                rng_val = np.random.RandomState(self.cfg.seed + 99)
                vi = rng_val.choice(n_val, max_val, replace=False)
                X_val_s, y_val_s = X_val[vi], y_val[vi]
                X_val_masked_s, val_mask_s = X_val_masked[vi], val_mask[vi]
                X_val_true_s = X_val_true[vi]
            else:
                X_val_s, y_val_s = X_val, y_val
                X_val_masked_s, val_mask_s = X_val_masked, val_mask
                X_val_true_s = X_val_true

            L, M, U = imputer.predict_intervals(X_val_masked_s)
            transformer = ProportionalFuzzyTransformer(eta=eta)
            a_v, b_v, c_v, d_v = transformer.transform_to_arrays(X_val_masked_s, L, M, U)
            L_tr, M_tr, U_tr = imputer.predict_intervals(X_train_s)
            a_t, b_t, c_t, d_t = transformer.transform_to_arrays(X_train_s, L_tr, M_tr, U_tr)

            forest = FuzzyDecisionForest(
                n_trees=n_trees, max_depth=max_depth,
                min_leaf_samples=min_leaf,
                feature_subsample_ratio=self.cfg.feature_subsample_ratio,
                n_classes=2, seed=self.cfg.seed,
                forest_workers=1,
            )
            forest.fit(a_t, b_t, c_t, d_t, y_train_s, sample_weights=sw_train_s)
            proba = np.clip(forest.predict_proba(a_v, b_v, c_v, d_v), 1e-7, 1 - 1e-7)

            L_class = log_loss(y_val_s, proba)

            feature_ranges = np.nanmax(X_train, axis=0) - np.nanmin(X_train, axis=0)
            feature_ranges[feature_ranges < 1e-8] = 1.0
            widths = np.where(np.isnan(np.abs(U - L)), 0.0, np.abs(U - L))
            MPIW_norm = np.nanmean(np.nanmean(widths, axis=0) / feature_ranges)

            if val_mask_s.sum() > 0:
                covered, total = 0, 0
                for j in range(X_val_masked_s.shape[1]):
                    mj = val_mask_s[:, j]
                    if mj.sum() == 0: continue
                    tv = X_val_true_s[mj, j]
                    lv, uv = L[mj, j], U[mj, j]
                    valid = ~(np.isnan(lv) | np.isnan(uv) | np.isnan(tv))
                    if valid.sum() > 0:
                        covered += ((tv[valid] >= lv[valid]) & (tv[valid] <= uv[valid])).sum()
                        total += valid.sum()
                PICP = covered / (total + 1e-12)
            else:
                PICP = 1.0

            coverage_gap = max(0, 0.95 - PICP)

            L_total = (
                L_class
                + self.cfg.lambda_1 * MPIW_norm
                + self.cfg.lambda_2 * coverage_gap
            )
            if np.isnan(L_total) or np.isinf(L_total):
                return 10.0
            self.history.append({"alpha_L": alpha_L, "alpha_U": alpha_U, "eta": eta,
                                 "L_total": L_total, "PICP": PICP})
            return L_total
        except Exception:
            return 10.0

    def _seed_population_around(self, best_x, popsize, rng, radius=0.15):
        """Create initial population centered on best_x with ±radius perturbation."""
        dim = len(best_x)
        pop = np.empty((popsize * dim, dim))
        for i in range(popsize * dim):
            candidate = best_x.copy()
            for j in range(dim):
                lo, hi = self.BOUNDS[j]
                span = hi - lo
                perturb = rng.uniform(-radius * span, radius * span)
                candidate[j] = np.clip(best_x[j] + perturb, lo, hi)
            pop[i] = candidate
        return pop

    def optimize(self, X_train, y_train, sw_train, X_val, y_val,
                 X_val_masked, val_mask, X_val_true, verbose=True):
        self.history = []
        de_args = (X_train, y_train, sw_train, X_val, y_val,
                   X_val_masked, val_mask, X_val_true)

        # Early-stopping callback for DE
        patience = self.cfg.de_patience
        class _EarlyStop:
            def __init__(self):
                self.best = np.inf
                self.wait = 0
                self.iter = 0
            def __call__(self, xk, convergence):
                self.iter += 1
                # convergence is fractional std of population energies
                # xk is the current best solution vector
                # We track actual best fitness via convergence metric
                if convergence < self.best - 1e-6:
                    self.best = convergence
                    self.wait = 0
                else:
                    self.wait += 1
                if self.wait >= patience:
                    if verbose:
                        print(f"    Early stop at iter {self.iter} (no improvement for {patience} iters)")
                    return True
                return False

        # ── Stage 1: Fast coarse search ──────────────────────────────
        if verbose:
            print(f"DE Stage 1 — coarse search: popsize={self.cfg.de_s1_popsize}, "
                  f"maxiter={self.cfg.de_s1_maxiter}, trees={self.cfg.de_s1_trees}, "
                  f"samples={self.cfg.de_s1_samples}, patience={patience}")

        self._current_n_trees = self.cfg.de_s1_trees
        self._current_max_samples = self.cfg.de_s1_samples

        s1_result = differential_evolution(
            func=self._composite_fitness,
            bounds=self.BOUNDS,
            args=de_args,
            maxiter=self.cfg.de_s1_maxiter,
            popsize=self.cfg.de_s1_popsize,
            tol=self.cfg.de_tol,
            mutation=self.cfg.de_mutation,
            recombination=self.cfg.de_recombination,
            seed=self.cfg.seed,
            polish=False,
            disp=verbose,
            workers=self.cfg.de_workers,
            updating="deferred",
            callback=_EarlyStop(),
        )
        if verbose:
            print(f"  Stage 1 best fitness: {s1_result.fun:.4f}")
            print(f"  Stage 1 best params: {np.round(s1_result.x, 4)}")

        # ── Stage 2: Fine refinement near Stage 1 best ──────────────
        if verbose:
            print(f"\nDE Stage 2 — fine refinement: popsize={self.cfg.de_s2_popsize}, "
                  f"maxiter={self.cfg.de_s2_maxiter}, trees={self.cfg.de_s2_trees}, "
                  f"samples={self.cfg.de_s2_samples}")

        rng = np.random.RandomState(self.cfg.seed + 1)
        init_pop = self._seed_population_around(
            s1_result.x, self.cfg.de_s2_popsize, rng, radius=0.15)

        self._current_n_trees = self.cfg.de_s2_trees
        self._current_max_samples = self.cfg.de_s2_samples

        s2_result = differential_evolution(
            func=self._composite_fitness,
            bounds=self.BOUNDS,
            args=de_args,
            maxiter=self.cfg.de_s2_maxiter,
            popsize=self.cfg.de_s2_popsize,
            tol=self.cfg.de_tol,
            mutation=self.cfg.de_mutation,
            recombination=self.cfg.de_recombination,
            seed=self.cfg.seed + 1,
            polish=False,
            disp=verbose,
            workers=self.cfg.de_workers,
            updating="deferred",
            init=init_pop,
            callback=_EarlyStop(),
        )

        # Use the best result across both stages
        if s2_result.fun <= s1_result.fun:
            best = s2_result
        else:
            best = s1_result
            if verbose:
                print("  (Stage 1 was better — keeping Stage 1 params)")

        self.best_params = {
            "alpha_L": best.x[0], "alpha_U": best.x[1], "eta": best.x[2],
            "max_depth": int(round(best.x[3])), "min_leaf_samples": int(round(best.x[4])),
            "fitness": best.fun,
        }

        if verbose:
            print(f"\nDE complete (two-stage). Best fitness: {best.fun:.4f}")
            for k, v in self.best_params.items():
                print(f"  {k}: {v}")
        return self.best_params


# ============================================================================
# 10. Phase 4 — Safe Abstention Protocol
# ============================================================================

class SafeAbstentionProtocol:
    """Entropy-based selective prediction with risk–coverage guarantees.

    The protocol identifies high-uncertainty predictions (measured by
    Shannon entropy of the output probability vector) and defers them
    to human clinicians rather than issuing unreliable predictions.

    **Calibration step** (on the held-out calibration set):
      1. Compute entropy H(p) for each calibration sample.
      2. Sort samples by increasing entropy.
      3. Walk along the sorted sequence and find the largest prefix
         whose cumulative error rate <= ``max_error_rate``.
      4. Set the entropy threshold tau at the boundary.

    **Inference step**: any test sample with H(p) > tau is flagged as
    "abstained" (deferred to clinician review).

    The risk–coverage curve and AURC (Area Under the Risk-Coverage
    curve) are used to summarise performance: lower AURC indicates
    better selective prediction quality.
    """

    def __init__(self, max_error_rate=0.05):
        self.max_error_rate = max_error_rate
        self.threshold = None

    @staticmethod
    def entropy(proba):
        proba = np.clip(proba, 1e-12, 1.0)
        return -np.sum(proba * np.log(proba), axis=1)

    def calibrate_threshold(self, proba, y_true):
        H = self.entropy(proba)
        y_pred = np.argmax(proba, axis=1)
        correct = (y_pred == y_true).astype(float)
        sorted_idx = np.argsort(H)
        sorted_correct = correct[sorted_idx]
        sorted_H = H[sorted_idx]

        best_tau = np.max(H) + 0.01
        cumsum_correct = np.cumsum(sorted_correct)
        for k in range(1, len(sorted_correct) + 1):
            error_rate = 1.0 - (cumsum_correct[k - 1] / k)
            if error_rate > self.max_error_rate:
                best_tau = sorted_H[k - 2] if k > 1 else 0.0
                break
        self.threshold = best_tau
        return best_tau

    def predict_with_abstention(self, proba):
        H = self.entropy(proba)
        abstain_mask = H > self.threshold
        predictions = np.argmax(proba, axis=1)
        return predictions, abstain_mask

    @staticmethod
    def compute_risk_coverage_curve(proba, y_true):
        H = SafeAbstentionProtocol.entropy(proba)
        errors = (np.argmax(proba, axis=1) != y_true).astype(float)
        sorted_idx = np.argsort(H)
        sorted_errors = errors[sorted_idx]
        n = len(sorted_errors)
        coverages = np.arange(1, n + 1) / n
        risks = np.cumsum(sorted_errors) / np.arange(1, n + 1)
        return coverages, risks

    @staticmethod
    def compute_aurc(coverages, risks):
        return np.trapezoid(risks, coverages)

    @staticmethod
    def compute_risk_at_coverage(coverages, risks, target=0.90):
        return risks[np.argmin(np.abs(coverages - target))]


# ============================================================================
# 11. JUCO Pipeline (Fuzzy Decision Forest backbone)
# ============================================================================

class JUCOPipeline:
    """End-to-end JUCO pipeline: Scaler → Imputer → Fuzzy → Forest.

    Chains the four phases into a single fit/predict interface:
      1. StandardScaler (preserves NaN mask)
      2. AdaptiveQuantileImputer → interval [L, M, U]
      3. ProportionalFuzzyTransformer → trapezoid (a, b, c, d)
      4. FuzzyDecisionForest → P(mortality | fuzzy features)

    Parameters
    ----------
    cfg : JUCOConfig
        Global configuration.
    params : dict or None
        If provided, overrides config values for alpha_L, alpha_U, eta,
        max_depth, min_leaf_samples (typically from DE optimisation).
    """

    def __init__(self, cfg, params=None):
        self.cfg = cfg
        self.params = params
        self.imputer = None
        self.transformer = None
        self.forest = None
        self.scaler = None

    def fit(self, X_train, y_train, sample_weights=None, verbose=True):
        p = self.params or {}
        alpha_L = p.get("alpha_L", self.cfg.alpha_L_init)
        alpha_U = p.get("alpha_U", self.cfg.alpha_U_init)
        eta = p.get("eta", self.cfg.eta_init)
        max_depth = p.get("max_depth", self.cfg.max_tree_depth)
        min_leaf = p.get("min_leaf_samples", self.cfg.min_leaf_samples)

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
        X_sc[np.isnan(X_train)] = np.nan

        sw, cw_dict = compute_sample_weights(y_train, verbose=verbose)
        if sample_weights is None:
            sample_weights = sw

        if verbose: print("Phase 1: Fitting Adaptive Quantile Imputer...")
        self.imputer = AdaptiveQuantileImputer(
            alpha_L=alpha_L, alpha_U=alpha_U,
            n_estimators=self.cfg.lgbm_n_estimators,
            learning_rate=self.cfg.lgbm_learning_rate,
            max_depth=self.cfg.lgbm_max_depth,
            min_child_samples=self.cfg.lgbm_min_child_samples, seed=self.cfg.seed,
        )
        self.imputer.fit(X_sc, y_labels=y_train, class_weight_dict=cw_dict, verbose=verbose)

        if verbose: print("Phase 2: Fuzzy Transformation...")
        self.transformer = ProportionalFuzzyTransformer(eta=eta)
        L, M, U = self.imputer.predict_intervals(X_sc)
        a, b, c, d = self.transformer.transform_to_arrays(X_sc, L, M, U)

        if verbose: print(f"Phase 3: Training Fuzzy Forest ({self.cfg.n_trees} trees)...")
        self.forest = FuzzyDecisionForest(
            n_trees=self.cfg.n_trees, max_depth=max_depth,
            min_leaf_samples=min_leaf,
            feature_subsample_ratio=self.cfg.feature_subsample_ratio,
            n_classes=2, seed=self.cfg.seed,
            forest_workers=self.cfg.forest_workers,
        )
        self.forest.fit(a, b, c, d, y_train, sample_weights=sample_weights, verbose=verbose)
        return self

    def predict_proba(self, X):
        X_sc = self.scaler.transform(np.nan_to_num(X, nan=0.0))
        X_sc[np.isnan(X)] = np.nan
        L, M, U = self.imputer.predict_intervals(X_sc)
        a, b, c, d = self.transformer.transform_to_arrays(X_sc, L, M, U)
        return self.forest.predict_proba(a, b, c, d)


# ============================================================================
# 11b. JUCO-XGB Pipeline (XGBoost backbone with uncertainty features)
# ============================================================================

class JUCOXGBPipeline:
    """JUCO framework with XGBoost as the downstream classifier.

    Instead of a fuzzy decision forest, this variant expands each of
    d features into three columns [q_L, center_with_NaN, q_U] and
    feeds the 3d-dimensional matrix to XGBoost.

    **Design rationale**: XGBoost natively handles NaN via learned
    split routing.  In clinical data, missingness itself is highly
    predictive (clinicians don't order labs for stable patients).
    By restoring NaN in the centre column, XGBoost retains that
    structural signal while also receiving uncertainty bounds (L, U)
    from the JUCO imputer.

    - Present values → centre = observed value; L ≈ U (tight interval)
    - Missing values → centre = NaN (native routing); L < U (wide)
    """

    def __init__(self, cfg, params=None):
        self.cfg = cfg
        self.params = params
        self.imputer = None
        self.scaler = None
        self.model = None

    def _expand_features(self, X_sc):
        """Expand scaled features using imputer intervals.

        For each of d features, produces 3 columns:
          [lower_bound (L), center_with_NaN, upper_bound (U)]
        Result shape: (n_samples, 3 * d)

        The center column uses the imputed median for present values
        but restores NaN for originally-missing values, so XGBoost
        can use its native missing-value split routing.  L and U
        always have valid numbers (imputer always produces bounds),
        giving XGBoost the uncertainty width signal even for NaN rows.
        """
        miss_mask = np.isnan(X_sc)
        L, M, U = self.imputer.predict_intervals(X_sc)

        # Center = imputed median, but restore NaN for missing values
        center = M.copy()
        center[miss_mask] = np.nan

        return np.hstack([L, center, U])

    def fit(self, X_train, y_train, sample_weights=None, verbose=True):
        p = self.params or {}
        alpha_L = p.get("alpha_L", self.cfg.alpha_L_init)
        alpha_U = p.get("alpha_U", self.cfg.alpha_U_init)

        self.scaler = StandardScaler()
        X_sc = self.scaler.fit_transform(np.nan_to_num(X_train, nan=0.0))
        X_sc[np.isnan(X_train)] = np.nan

        sw, cw_dict = compute_sample_weights(y_train, verbose=verbose)
        if sample_weights is None:
            sample_weights = sw

        if verbose:
            print("JUCO-XGB Phase 1: Fitting Adaptive Quantile Imputer...")
        self.imputer = AdaptiveQuantileImputer(
            alpha_L=alpha_L, alpha_U=alpha_U,
            n_estimators=self.cfg.lgbm_n_estimators,
            learning_rate=self.cfg.lgbm_learning_rate,
            max_depth=self.cfg.lgbm_max_depth,
            min_child_samples=self.cfg.lgbm_min_child_samples,
            seed=self.cfg.seed,
        )
        self.imputer.fit(X_sc, y_labels=y_train, class_weight_dict=cw_dict, verbose=verbose)

        if verbose:
            print("JUCO-XGB Phase 2: Feature Expansion (3× features: L, center+NaN, U)...")
        X_expanded = self._expand_features(X_sc)
        if verbose:
            print(f"  Expanded: {X_train.shape[1]} → {X_expanded.shape[1]} features")

        if verbose:
            print("JUCO-XGB Phase 3: Training XGBoost on expanded features...")
        pw = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-9)
        self.model = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=pw,
            subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=self.cfg.seed,
            verbosity=0, n_jobs=-1,
            missing=np.nan,   # explicit: route NaN via learned splits
        )
        self.model.fit(X_expanded, y_train, sample_weight=sample_weights)
        return self

    def predict_proba(self, X):
        X_sc = self.scaler.transform(np.nan_to_num(X, nan=0.0))
        X_sc[np.isnan(X)] = np.nan
        X_expanded = self._expand_features(X_sc)
        return self.model.predict_proba(X_expanded)


# ============================================================================
# 12. Baseline Models
# ============================================================================
# Five baselines for comparative evaluation:
#
#   BaselineXGBoost    — XGBoost with native NaN handling (no imputation)
#   BaselineLGBM       — LightGBM with native NaN handling
#   BaselineMissForestXGB  — IterativeImputer (MissForest-style) + XGBoost
#   BaselineMeanXGB    — Mean imputation + XGBoost
#   BaselineTabNet     — Deep-learning baseline (Arik & Pfister, 2021)
#
# All baselines use inverse-frequency class weighting (scale_pos_weight)
# to handle the ~10 % mortality prevalence, mirroring JUCO's approach.
# ============================================================================

class BaselineXGBoost:
    """XGBoost baseline — relies on native NaN split routing."""
    def __init__(self, seed=42):
        self.model = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            tree_method="hist", random_state=seed, verbosity=0, n_jobs=-1,
        )
    def fit(self, X, y, **kw):
        pw = np.sum(y == 0) / (np.sum(y == 1) + 1e-9)
        self.model.set_params(scale_pos_weight=pw)
        self.model.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class BaselineLGBM:
    """LightGBM baseline — bin-based NaN handling."""
    def __init__(self, seed=42):
        self.model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            random_state=seed, verbosity=-1, n_jobs=-1,
        )
    def fit(self, X, y, **kw):
        pw = np.sum(y == 0) / (np.sum(y == 1) + 1e-9)
        self.model.set_params(scale_pos_weight=pw)
        self.model.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class BaselineMissForestXGB:
    """MissForest-style iterative imputation (Stekhoven & Bühlmann, 2012) + XGBoost."""
    def __init__(self, seed=42):
        self.imputer = IterativeImputer(estimator=None, max_iter=10, random_state=seed)
        self.model = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            tree_method="hist", random_state=seed, verbosity=0, n_jobs=-1,
        )
    def fit(self, X, y, **kw):
        X_imp = self.imputer.fit_transform(X)
        pw = np.sum(y == 0) / (np.sum(y == 1) + 1e-9)
        self.model.set_params(scale_pos_weight=pw)
        self.model.fit(X_imp, y)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(self.imputer.transform(X))


class BaselineMeanXGB:
    """Simple mean imputation + XGBoost (naïve baseline)."""
    def __init__(self, seed=42):
        self.imputer = SimpleImputer(strategy="mean")
        self.model = xgb_lib.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            tree_method="hist", random_state=seed, verbosity=0, n_jobs=-1,
        )
    def fit(self, X, y, **kw):
        X_imp = self.imputer.fit_transform(X)
        pw = np.sum(y == 0) / (np.sum(y == 1) + 1e-9)
        self.model.set_params(scale_pos_weight=pw)
        self.model.fit(X_imp, y)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(self.imputer.transform(X))


class BaselineTabNet:
    """TabNet deep-learning baseline (Arik & Pfister, AAAI 2021).

    Features are mean-imputed and augmented with binary missingness
    indicators, mirroring the recommended TabNet preprocessing.
    Automatic class balancing is enabled via ``weights=1``.
    Falls back to zero-probability output if pytorch_tabnet is not
    installed.
    """
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None

    def _prepare_data(self, X):
        X_imp = np.copy(X)
        col_means = np.nanmean(X_imp, axis=0)
        inds = np.where(np.isnan(X_imp))
        X_imp[inds] = np.take(col_means, inds[1])
        X_imp = np.nan_to_num(X_imp, nan=0.0)
        return np.hstack((X_imp, np.isnan(X).astype(np.float32)))

    def fit(self, X, y, sample_weight=None, **kw):
        if TabNetClassifier is None:
            return self

        # Force y to int64 (TabNet requirement)
        y = y.astype(np.int64)

        X_dl = self._prepare_data(X)

        # ---- Train/Val Split (CRUCIAL) ----
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_dl, y, test_size=0.2, stratify=y, random_state=self.seed
        )

        self.model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=1e-3),
            seed=self.seed,
            verbose=0,
            device_name="cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=["auc"],
            max_epochs=200,
            patience=20,
            batch_size=4096,
            virtual_batch_size=256,
            weights=1  # 1 = automatic class balancing by TabNet
        )

        return self

    def predict_proba(self, X):
        if self.model is None:
            return np.zeros((len(X), 2))
        return self.model.predict_proba(self._prepare_data(X))


# Registry of baseline models (used by run_experiment and stress test).
# TabNet is included only when pytorch_tabnet is available.
BASELINES = {
    "XGBoost":        BaselineXGBoost,
    "LightGBM":       BaselineLGBM,
    "MissForest+XGB": BaselineMissForestXGB,
    "MeanImp+XGB":    BaselineMeanXGB,
}
if TabNetClassifier is not None:
    BASELINES["TabNet (DL)"] = BaselineTabNet


# ============================================================================
# 13. Evaluation Metrics
# ============================================================================

# Metrics reported in the paper (Tables 1–2).
METRIC_NAMES = ["Accuracy", "F1_macro", "AUROC", "ECE", "Brier", "AURC", "Risk@90"]


def compute_ece(proba, y_true, n_bins=15):
    """Expected Calibration Error (Naeini et al., AAAI 2015).

    ECE measures the gap between predicted confidence and observed
    accuracy, averaged across equal-width confidence bins:

        ECE = sum_{b=1}^{B} (n_b / N) * |acc_b - conf_b|

    Lower ECE indicates better calibration.  A perfectly calibrated
    model has ECE = 0.
    """
    confidences = np.max(proba, axis=1)
    predictions = np.argmax(proba, axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - confidences[mask].mean())
    return ece


def platt_calibrate(proba_cal, y_cal, proba_target):
    """Platt scaling: fit logistic regression on calibration probabilities.

    Maps uncalibrated P(class=1) to calibrated probabilities via a
    one-dimensional logistic regression on the calibration set.
    """
    pc = proba_cal[:, 1] if proba_cal.ndim == 2 else proba_cal
    pt = proba_target[:, 1] if proba_target.ndim == 2 else proba_target
    try:
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(pc.reshape(-1, 1), y_cal)
        calibrated = lr.predict_proba(pt.reshape(-1, 1))
        return calibrated
    except Exception:
        return np.column_stack([1 - pt, pt])


def isotonic_calibrate(proba_cal, y_cal, proba_target):
    """Isotonic regression calibration (Zadrozny & Elkan, 2002).

    Non-parametric calibration via monotonic regression.  Preferred
    over Platt scaling for large datasets because it makes no
    distributional assumptions.
    """
    pc = proba_cal[:, 1] if proba_cal.ndim == 2 else proba_cal
    pt = proba_target[:, 1] if proba_target.ndim == 2 else proba_target
    try:
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(pc, y_cal)
        calibrated_pos = np.clip(ir.predict(pt), 1e-7, 1 - 1e-7)
        return np.column_stack([1 - calibrated_pos, calibrated_pos])
    except Exception:
        return np.column_stack([1 - pt, pt])


def evaluate_model(proba, y_true):
    """Compute all seven evaluation metrics for a single model.

    Returns a dict with keys matching ``METRIC_NAMES``:
    Accuracy, F1_macro, AUROC, ECE, Brier, AURC, Risk@90.
    """
    y_pred = np.argmax(proba, axis=1)
    proba_pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["F1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        metrics["AUROC"] = roc_auc_score(y_true, proba_pos)
    except ValueError:
        metrics["AUROC"] = 0.5
    metrics["ECE"] = compute_ece(proba, y_true)
    metrics["Brier"] = brier_score_loss(y_true, proba_pos)
    cov, risk = SafeAbstentionProtocol.compute_risk_coverage_curve(proba, y_true)
    metrics["AURC"] = SafeAbstentionProtocol.compute_aurc(cov, risk)
    metrics["Risk@90"] = SafeAbstentionProtocol.compute_risk_at_coverage(cov, risk, 0.90)
    return metrics


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    """Non-parametric bootstrap confidence intervals for any scalar metric.

    Returns (mean, lower, upper) for a (1 - alpha) interval.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return np.mean(scores), lo, hi


def delong_roc_test(y_true, y_score_a, y_score_b):
    """DeLong's test for comparing two correlated AUROCs (DeLong et al., 1988).

    Computes a z-statistic and two-sided p-value testing the null
    hypothesis AUC_A = AUC_B on the same test set.
    """
    y_true = np.asarray(y_true, dtype=int)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    m, n = len(pos_idx), len(neg_idx)
    if m < 2 or n < 2:
        return 0.0, 1.0

    sa_pos, sa_neg = y_score_a[pos_idx], y_score_a[neg_idx]
    sb_pos, sb_neg = y_score_b[pos_idx], y_score_b[neg_idx]
    V10_a = np.array([np.mean((sa_neg < s) + 0.5 * (sa_neg == s)) for s in sa_pos])
    V10_b = np.array([np.mean((sb_neg < s) + 0.5 * (sb_neg == s)) for s in sb_pos])
    V01_a = np.array([np.mean((sa_pos > s) + 0.5 * (sa_pos == s)) for s in sa_neg])
    V01_b = np.array([np.mean((sb_pos > s) + 0.5 * (sb_pos == s)) for s in sb_neg])

    S10 = np.cov(np.vstack([V10_a, V10_b]))
    S01 = np.cov(np.vstack([V01_a, V01_b]))
    S = S10 / m + S01 / n

    auc_a, auc_b = np.mean(V10_a), np.mean(V10_b)
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff < 1e-12:
        return 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    return z, p


def summarize_results(results, dataset_name):
    """Aggregate per-fold metrics into mean ± std summary table."""
    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            row[metric_name] = f"{mean_val:.4f} +/- {std_val:.4f}"
            row[f"{metric_name}_mean"] = mean_val
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


# ============================================================================
# 14. Experiment Runner (main entry point)
# ============================================================================

def run_experiment(dataset_name, data, cfg, verbose=True, frozen_params=None):
    """Run the full JUCO + baseline benchmark on one dataset.

    **Workflow per fold**:
      1. Three-way patient-level split (train / val / cal / test).
      2. Per-fold mutual-information feature selection (if max_features > 0).
      3. MNAR masking on train + val (test remains unmasked).
      4. DE optimisation (Fold 1 only — params frozen for remaining folds).
      5. Train JUCO, JUCO-XGB, and all baselines.
      6. Isotonic calibration on the calibration set.
      7. Evaluate all seven metrics on the held-out test set.
      8. Save per-fold predictions for later DCA analysis.

    Parameters
    ----------
    dataset_name : str
        Label used for logging and checkpoint filenames.
    data : dict
        Output of ``preprocess_dataset`` (X_raw, y, groups, folds, …).
    cfg : JUCOConfig
        Global configuration.
    verbose : bool
        Print progress messages.
    frozen_params : dict or None
        If provided, skip DE optimisation and use these params for all
        folds.  Used for external validation (e.g., eICU) with params
        learned on the training dataset (MIMIC-IV) to avoid data leakage.

    Returns
    -------
    all_results : dict
        {model_name: {metric: [fold_values]}} across all folds.
    frozen_params : dict
        DE-optimised (or pre-supplied) parameter dictionary.
    """
    X_raw = data["X_raw"]
    y = data["y"]
    groups = data["groups"]
    folds = data["folds"]
    if cfg.run_folds > 0:
        folds = folds[:cfg.run_folds]
        if verbose:
            print(f"  (Limiting to {cfg.run_folds} fold(s) for verification)")

    skip_de = frozen_params is not None
    if skip_de and verbose:
        print(f"  Using pre-frozen params (skip DE): {frozen_params}")

    model_names = ["JUCO", "JUCO-XGB"] + list(BASELINES.keys())
    all_results = {m: {metric: [] for metric in METRIC_NAMES} for m in model_names}
    # Per-fold predictions for DCA (y_true + calibrated probas per model)
    fold_predictions = []
    exp_t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_t0 = time.time()
        print(f"\n{'='*70}")
        print(f" {dataset_name} -- Fold {fold_idx + 1}/{len(folds)}")
        print(f"{'='*70}")

        split = three_way_split(X_raw, y, groups, train_idx,
                                cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx)
        X_tr, y_tr = split["X_tr"], split["y_tr"]
        X_val, y_val = split["X_val"], split["y_val"]
        X_cal, y_cal = split["X_cal"], split["y_cal"]
        X_test = X_raw[test_idx].copy()
        y_test = y[test_idx].copy()

        # Per-fold feature selection (no leakage — uses training data only)
        feature_names = data["feature_names"]
        if cfg.max_features > 0 and X_tr.shape[1] > cfg.max_features:
            top_idx, fold_features = select_top_features(
                X_tr, y_tr, feature_names, cfg.max_features, cfg.seed + fold_idx)
            X_tr = X_tr[:, top_idx]
            X_val = X_val[:, top_idx]
            X_cal = X_cal[:, top_idx]
            X_test = X_test[:, top_idx]

        print("MNAR masking (train + val only)...")
        X_tr_m, tr_mask = apply_mnar_mask(X_tr, cfg.mnar_beta, cfg.mnar_gamma,
                                           cfg.mnar_rate, cfg.seed + fold_idx)
        X_val_true = X_val.copy()
        X_val_m, val_mask = apply_mnar_mask(X_val, cfg.mnar_beta, cfg.mnar_gamma,
                                             cfg.mnar_rate, cfg.seed + fold_idx + 50)

        sample_weights, cw_dict = compute_sample_weights(y_tr)

        if skip_de:
            print(f"Using pre-frozen params (external validation): {frozen_params}")
        elif fold_idx == 0:
            print("Running two-stage DE optimization (Fold 1 -- will freeze params)...")
            optimizer = JUCOOptimizer(cfg)
            frozen_params = optimizer.optimize(
                X_tr_m, y_tr, sample_weights, X_val_m, y_val,
                X_val_m, val_mask, X_val_true, verbose=verbose,
            )
            print(f"Frozen params: {frozen_params}")
        else:
            print(f"Using frozen DE params from Fold 1: {frozen_params}")

        print(f"\nTraining JUCO (frozen DE params)...")
        juco = JUCOPipeline(cfg, params=frozen_params)
        juco.fit(X_tr_m, y_tr, sample_weights=sample_weights, verbose=verbose)

        proba_cal_juco = np.clip(juco.predict_proba(X_cal), 1e-7, 1 - 1e-7)
        proba_test_juco_raw = np.clip(juco.predict_proba(X_test), 1e-7, 1 - 1e-7)
        proba_test_juco = isotonic_calibrate(proba_cal_juco, y_cal, proba_test_juco_raw)

        proba_cal_juco_calibrated = isotonic_calibrate(proba_cal_juco, y_cal, proba_cal_juco)
        abstention = SafeAbstentionProtocol(cfg.max_error_rate)
        tau = abstention.calibrate_threshold(proba_cal_juco_calibrated, y_cal)
        print(f"  Abstention tau = {tau:.4f}")

        metrics_juco = evaluate_model(proba_test_juco, y_test)
        for m, v in metrics_juco.items():
            all_results["JUCO"][m].append(v)
        print(f"  JUCO -- AUROC: {metrics_juco['AUROC']:.4f}, ECE: {metrics_juco['ECE']:.4f}")

        # --- JUCO-XGB ---
        print(f"\nTraining JUCO-XGB (frozen DE params)...")
        jxgb = JUCOXGBPipeline(cfg, params=frozen_params)
        jxgb.fit(X_tr_m, y_tr, sample_weights=sample_weights, verbose=(fold_idx == 0))
        proba_cal_jxgb = np.clip(jxgb.predict_proba(X_cal), 1e-7, 1 - 1e-7)
        proba_test_jxgb_raw = np.clip(jxgb.predict_proba(X_test), 1e-7, 1 - 1e-7)
        proba_test_jxgb = isotonic_calibrate(proba_cal_jxgb, y_cal, proba_test_jxgb_raw)
        metrics_jxgb = evaluate_model(proba_test_jxgb, y_test)
        for m, v in metrics_jxgb.items():
            all_results["JUCO-XGB"][m].append(v)
        print(f"  JUCO-XGB -- AUROC: {metrics_jxgb['AUROC']:.4f}, ECE: {metrics_jxgb['ECE']:.4f}")

        bl_probas = {}
        for bl_name, bl_class in BASELINES.items():
            print(f"\nTraining {bl_name}...")
            bl = bl_class(seed=cfg.seed)
            bl.fit(X_tr_m, y_tr)
            proba_cal_bl = np.clip(bl.predict_proba(X_cal), 1e-7, 1 - 1e-7)
            proba_test_bl_raw = np.clip(bl.predict_proba(X_test), 1e-7, 1 - 1e-7)
            proba_test_bl = isotonic_calibrate(proba_cal_bl, y_cal, proba_test_bl_raw)
            bl_probas[bl_name] = proba_test_bl
            metrics_bl = evaluate_model(proba_test_bl, y_test)
            for m, v in metrics_bl.items():
                all_results[bl_name][m].append(v)
            print(f"  {bl_name} -- AUROC: {metrics_bl['AUROC']:.4f}, ECE: {metrics_bl['ECE']:.4f}")

        # --- Save per-fold predictions for DCA ---
        fold_pred = {"fold": fold_idx, "y_true": y_test,
                     "JUCO": proba_test_juco, "JUCO-XGB": proba_test_jxgb}
        for bl_name in BASELINES:
            fold_pred[bl_name] = bl_probas[bl_name]
        fold_predictions.append(fold_pred)

        # --- SPOT INSTANCE SAFEGUARD ---
        save_checkpoint(
            os.path.join(cfg.output_dir, f"checkpoint_temp_{dataset_name}.pkl"),
            experiment_results={dataset_name: all_results},
            frozen_params={dataset_name: frozen_params}
        )
        # ----------------------------------------

        fold_elapsed = time.time() - fold_t0
        total_elapsed = time.time() - exp_t0
        print(f"\n  [Timer] Fold {fold_idx+1} took {fold_elapsed/60:.1f} min  |  Total elapsed: {total_elapsed/60:.1f} min")

    # Save predictions for DCA reuse
    pred_path = os.path.join(cfg.output_dir, f"predictions_{dataset_name.replace('-', '_')}.pkl")
    save_checkpoint(pred_path, fold_predictions=fold_predictions)
    print(f"  Predictions saved: {pred_path}")

    return all_results, frozen_params


# ============================================================================
# 15. Ablation Studies
# ============================================================================

def run_ablation_studies(dataset_name, data, cfg):
    """Run five ablation experiments to quantify each pipeline component.

    Each ablation removes or replaces one component while keeping the
    rest intact, isolating its contribution:

      A0: JUCO (Full)       — complete pipeline (reference)
      A1: Sequential        — impute first, then standard Random Forest
                              (no joint fuzzy-forest reasoning)
      A2: Crisp Forest      — standard Random Forest on median-imputed
                              data (no fuzzy soft routing)
      A3: Point Imputation  — fuzzy forest on point (median) imputations
                              (no interval uncertainty propagation)
      A4: No Abstention     — full JUCO without selective prediction
                              (all patients receive a prediction)
      A5: No MNAR           — JUCO trained on raw data without
                              additional MNAR masking (no robustness
                              augmentation)

    Runs on first 2 folds for computational efficiency.
    """
    X_raw = data["X_raw"]
    y = data["y"]
    groups = data["groups"]
    folds = data["folds"][:2]

    ablation_names = ["JUCO (Full)", "A1: Sequential", "A2: Crisp Forest",
                      "A3: Point Imputation", "A4: No Abstention", "A5: No MNAR"]
    abl_metrics = ["AUROC", "ECE", "Brier", "AURC"]
    abl_results = {n: {m: [] for m in abl_metrics} for n in ablation_names}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n  Ablation Fold {fold_idx + 1}/{len(folds)}...")
        split = three_way_split(X_raw, y, groups, train_idx,
                                cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx)
        X_tr, y_tr = split["X_tr"], split["y_tr"]
        X_val, y_val = split["X_val"], split["y_val"]
        X_cal, y_cal = split["X_cal"], split["y_cal"]
        X_test = X_raw[test_idx].copy()
        y_test = y[test_idx].copy()

        # Per-fold feature selection (no leakage)
        feature_names = data["feature_names"]
        if cfg.max_features > 0 and X_tr.shape[1] > cfg.max_features:
            top_idx, _ = select_top_features(
                X_tr, y_tr, feature_names, cfg.max_features, cfg.seed + fold_idx)
            X_tr = X_tr[:, top_idx]
            X_val = X_val[:, top_idx]
            X_cal = X_cal[:, top_idx]
            X_test = X_test[:, top_idx]

        X_tr_m, _ = apply_mnar_mask(X_tr, cfg.mnar_beta, cfg.mnar_gamma,
                                     cfg.mnar_rate, cfg.seed + fold_idx)
        X_val_m, _ = apply_mnar_mask(X_val, cfg.mnar_beta, cfg.mnar_gamma,
                                      cfg.mnar_rate, cfg.seed + fold_idx + 50)
        sw, cw = compute_sample_weights(y_tr)

        # A0: Full JUCO (no DE for speed)
        juco = JUCOPipeline(cfg, params=None)
        juco.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
        pc = np.clip(juco.predict_proba(X_cal), 1e-7, 1-1e-7)
        pt = np.clip(juco.predict_proba(X_test), 1e-7, 1-1e-7)
        pt = isotonic_calibrate(pc, y_cal, pt)
        m = evaluate_model(pt, y_test)
        for k in abl_metrics:
            abl_results["JUCO (Full)"][k].append(m[k])

        # A1: Sequential
        imp = AdaptiveQuantileImputer(seed=cfg.seed)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(np.nan_to_num(X_tr_m, nan=0.0))
        Xs[np.isnan(X_tr_m)] = np.nan
        imp.fit(Xs, y_labels=y_tr, class_weight_dict=cw)
        X_tr_imp = imp.impute_median(Xs)
        Xts = scaler.transform(np.nan_to_num(X_test, nan=0.0))
        Xts[np.isnan(X_test)] = np.nan
        X_te_imp = imp.impute_median(Xts)
        rf = RandomForestClassifier(n_estimators=200, max_depth=cfg.max_tree_depth,
                                     random_state=cfg.seed, n_jobs=1)
        rf.fit(X_tr_imp, y_tr)
        Xcs = scaler.transform(np.nan_to_num(X_cal, nan=0.0))
        Xcs[np.isnan(X_cal)] = np.nan
        X_cal_imp = imp.impute_median(Xcs)
        pc = np.clip(rf.predict_proba(X_cal_imp), 1e-7, 1-1e-7)
        pt = np.clip(rf.predict_proba(X_te_imp), 1e-7, 1-1e-7)
        pt = isotonic_calibrate(pc, y_cal, pt)
        m = evaluate_model(pt, y_test)
        for k in abl_metrics:
            abl_results["A1: Sequential"][k].append(m[k])

        # A2: Crisp Forest
        crisp = RandomForestClassifier(n_estimators=cfg.n_trees, max_depth=cfg.max_tree_depth,
                                        random_state=cfg.seed, n_jobs=1)
        X_tr_crisp = SimpleImputer(strategy="median").fit_transform(X_tr_m)
        X_cal_crisp = SimpleImputer(strategy="median").fit(X_tr_m).transform(X_cal)
        X_te_crisp = SimpleImputer(strategy="median").fit(X_tr_m).transform(X_test)
        crisp.fit(X_tr_crisp, y_tr)
        pc = np.clip(crisp.predict_proba(X_cal_crisp), 1e-7, 1-1e-7)
        pt = np.clip(crisp.predict_proba(X_te_crisp), 1e-7, 1-1e-7)
        pt = isotonic_calibrate(pc, y_cal, pt)
        m = evaluate_model(pt, y_test)
        for k in abl_metrics:
            abl_results["A2: Crisp Forest"][k].append(m[k])

        # A3: Point Imputation
        imp3 = AdaptiveQuantileImputer(seed=cfg.seed)
        sc3 = StandardScaler()
        X3s = sc3.fit_transform(np.nan_to_num(X_tr_m, nan=0.0))
        X3s[np.isnan(X_tr_m)] = np.nan
        imp3.fit(X3s, y_labels=y_tr, class_weight_dict=cw)
        X_tr_pt = imp3.impute_median(X3s)
        a3 = b3 = c3 = d3 = X_tr_pt
        forest3 = FuzzyDecisionForest(n_trees=cfg.n_trees, max_depth=cfg.max_tree_depth,
                                       min_leaf_samples=cfg.min_leaf_samples, n_classes=2, seed=cfg.seed,
                                       forest_workers=cfg.forest_workers)
        forest3.fit(a3, b3, c3, d3, y_tr, sample_weights=sw)
        X3ts = sc3.transform(np.nan_to_num(X_test, nan=0.0))
        X3ts[np.isnan(X_test)] = np.nan
        X_te_pt = imp3.impute_median(X3ts)
        X3cs = sc3.transform(np.nan_to_num(X_cal, nan=0.0))
        X3cs[np.isnan(X_cal)] = np.nan
        X_cal_pt = imp3.impute_median(X3cs)
        pc = np.clip(forest3.predict_proba(X_cal_pt, X_cal_pt, X_cal_pt, X_cal_pt), 1e-7, 1-1e-7)
        pt = np.clip(forest3.predict_proba(X_te_pt, X_te_pt, X_te_pt, X_te_pt), 1e-7, 1-1e-7)
        pt = isotonic_calibrate(pc, y_cal, pt)
        m = evaluate_model(pt, y_test)
        for k in abl_metrics:
            abl_results["A3: Point Imputation"][k].append(m[k])

        # A4: No Abstention (force 100% coverage — no selective prediction)
        pt_full = np.clip(juco.predict_proba(X_test), 1e-7, 1-1e-7)
        pt_full = isotonic_calibrate(np.clip(juco.predict_proba(X_cal), 1e-7, 1-1e-7), y_cal, pt_full)
        m_a4 = evaluate_model(pt_full, y_test)
        # Override AURC: with no abstention, AURC = full error rate (coverage=1.0 everywhere)
        y_pred_a4 = np.argmax(pt_full, axis=1)
        m_a4["AURC"] = float(np.mean(y_pred_a4 != y_test))
        m_a4["Risk@90"] = m_a4["AURC"]  # no selective prediction = same risk at any coverage
        for k in abl_metrics:
            abl_results["A4: No Abstention"][k].append(m_a4[k])

        # A5: No MNAR
        juco5 = JUCOPipeline(cfg, params=None)
        juco5.fit(X_tr, y_tr, sample_weights=sw, verbose=False)
        pc5 = np.clip(juco5.predict_proba(X_cal), 1e-7, 1-1e-7)
        pt5 = np.clip(juco5.predict_proba(X_test), 1e-7, 1-1e-7)
        pt5 = isotonic_calibrate(pc5, y_cal, pt5)
        m = evaluate_model(pt5, y_test)
        for k in abl_metrics:
            abl_results["A5: No MNAR"][k].append(m[k])

    rows = []
    for name, metrics in abl_results.items():
        row = {"Variant": name}
        for metric_name, values in metrics.items():
            row[metric_name] = f"{np.mean(values):.4f} +/- {np.std(values):.4f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Variant")


# ============================================================================
# 16. Missingness Stress Test
# ============================================================================

def run_missingness_stress_test(dataset_name, data, cfg, frozen_params=None,
                                 miss_rates=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
                                 n_stress_folds=3, verbose=True):
    """Evaluate JUCO and baselines under increasing MNAR missingness.

    **Central robustness experiment**: all models are retrained at each
    missingness rate r in {10 %, 20 %, …, 60 %}.  If JUCO degrades
    more gracefully than baselines at high missingness, this validates
    the core claim that uncertainty-aware imputation + fuzzy soft
    routing provides robustness to informative missingness.

    The output table reports AUROC at each rate along with absolute
    degradation from the lowest rate, making it easy to see which
    models are most affected by increasing missingness.
    """
    X_raw = data["X_raw"]
    y = data["y"]
    groups = data["groups"]
    folds = data["folds"][:n_stress_folds]

    model_names = ["JUCO", "JUCO-XGB", "XGBoost", "LightGBM", "MissForest+XGB", "MeanImp+XGB"]
    # results[rate][model] = list of AUROC values across folds
    results = {r: {m: [] for m in model_names} for r in miss_rates}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if verbose:
            print(f"\n--- Stress Test Fold {fold_idx+1}/{len(folds)} ---")

        split = three_way_split(X_raw, y, groups, train_idx,
                                cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx)
        X_tr, y_tr = split["X_tr"], split["y_tr"]
        X_cal, y_cal = split["X_cal"], split["y_cal"]
        X_test = X_raw[test_idx].copy()
        y_test = y[test_idx].copy()

        # Per-fold feature selection
        feature_names = data["feature_names"]
        if cfg.max_features > 0 and X_tr.shape[1] > cfg.max_features:
            top_idx, _ = select_top_features(
                X_tr, y_tr, feature_names, cfg.max_features, cfg.seed + fold_idx)
            X_tr = X_tr[:, top_idx]
            X_cal = X_cal[:, top_idx]
            X_test = X_test[:, top_idx]

        sw, cw = compute_sample_weights(y_tr, verbose=False)

        for rate in miss_rates:
            if verbose:
                print(f"  Missing rate = {rate:.0%}...", end=" ")

            # Apply MNAR mask at this rate
            X_tr_m, _ = apply_mnar_mask(X_tr, cfg.mnar_beta, cfg.mnar_gamma,
                                         rate, cfg.seed + fold_idx)

            # --- JUCO ---
            juco = JUCOPipeline(cfg, params=frozen_params)
            juco.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
            pc = np.clip(juco.predict_proba(X_cal), 1e-7, 1-1e-7)
            pt = np.clip(juco.predict_proba(X_test), 1e-7, 1-1e-7)
            pt = isotonic_calibrate(pc, y_cal, pt)
            proba_pos = pt[:, 1] if pt.ndim == 2 else pt
            try:
                auc = roc_auc_score(y_test, proba_pos)
            except ValueError:
                auc = 0.5
            results[rate]["JUCO"].append(auc)

            # --- JUCO-XGB ---
            jxgb = JUCOXGBPipeline(cfg, params=frozen_params)
            jxgb.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
            pc_jx = np.clip(jxgb.predict_proba(X_cal), 1e-7, 1-1e-7)
            pt_jx = np.clip(jxgb.predict_proba(X_test), 1e-7, 1-1e-7)
            pt_jx = isotonic_calibrate(pc_jx, y_cal, pt_jx)
            proba_pos_jx = pt_jx[:, 1] if pt_jx.ndim == 2 else pt_jx
            try:
                auc_jx = roc_auc_score(y_test, proba_pos_jx)
            except ValueError:
                auc_jx = 0.5
            results[rate]["JUCO-XGB"].append(auc_jx)

            # --- Baselines ---
            bl_classes = {
                "XGBoost": BaselineXGBoost,
                "LightGBM": BaselineLGBM,
                "MissForest+XGB": BaselineMissForestXGB,
                "MeanImp+XGB": BaselineMeanXGB,
            }
            for bl_name, bl_cls in bl_classes.items():
                bl = bl_cls(seed=cfg.seed)
                bl.fit(X_tr_m, y_tr)
                pc_bl = np.clip(bl.predict_proba(X_cal), 1e-7, 1-1e-7)
                pt_bl = np.clip(bl.predict_proba(X_test), 1e-7, 1-1e-7)
                pt_bl = isotonic_calibrate(pc_bl, y_cal, pt_bl)
                proba_pos_bl = pt_bl[:, 1] if pt_bl.ndim == 2 else pt_bl
                try:
                    auc_bl = roc_auc_score(y_test, proba_pos_bl)
                except ValueError:
                    auc_bl = 0.5
                results[rate][bl_name].append(auc_bl)

            if verbose:
                juco_auc = results[rate]["JUCO"][-1]
                jxgb_auc = results[rate]["JUCO-XGB"][-1]
                xgb_auc = results[rate]["XGBoost"][-1]
                print(f"JUCO={juco_auc:.4f}, JUCO-XGB={jxgb_auc:.4f}, XGB={xgb_auc:.4f}")

    # Build summary table
    rows = []
    for rate in miss_rates:
        row = {"Missing_Rate": f"{rate:.0%}"}
        for m in model_names:
            vals = results[rate][m]
            row[f"{m}_AUROC"] = f"{np.mean(vals):.4f}"
            row[f"{m}_AUROC_mean"] = np.mean(vals)
        rows.append(row)
    summary_df = pd.DataFrame(rows).set_index("Missing_Rate")

    # Compute degradation from lowest rate
    if verbose:
        base_rate = miss_rates[0]
        print(f"\n{'='*80}")
        print(f" Missingness Stress Test — AUROC Degradation from {base_rate:.0%}")
        print(f"{'='*80}")
        print(f"{'Rate':<10}", end="")
        for m in model_names:
            print(f"  {m:<16}", end="")
        print()
        for rate in miss_rates:
            print(f"{rate:<10.0%}", end="")
            for m in model_names:
                base_val = np.mean(results[base_rate][m])
                curr_val = np.mean(results[rate][m])
                delta = curr_val - base_val
                print(f"  {curr_val:.4f} ({delta:+.4f})  ", end="")
            print()

    return summary_df, results


# ============================================================================
# 17. Decision Curve Analysis (Net Benefit)
# ============================================================================

def decision_curve_analysis(y_true, model_probas, model_names, thresholds=None):
    """Compute net benefit curves for clinical decision-making.

    Decision Curve Analysis (Vickers & Elkin, Medical Decision Making,
    2006) evaluates models by the clinical utility they provide at
    different treatment threshold probabilities:

        Net Benefit = TP/N − FP/N × p_t / (1 − p_t)

    where p_t is the threshold probability at which a clinician would
    treat.  Two reference strategies are included:
      - "Treat All"  — treating every patient (upper-left baseline)
      - "Treat None"  — treating no patient (NB = 0)

    A model is clinically useful at threshold p_t if its net benefit
    exceeds both reference strategies.

    Returns
    -------
    pd.DataFrame
        Long-format table with columns: Threshold, Model, Net_Benefit.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 0.80, 0.01)

    n = len(y_true)
    rows = []

    # "Treat All" baseline
    for pt in thresholds:
        tp = np.sum(y_true == 1)
        fp = np.sum(y_true == 0)
        nb = tp / n - fp / n * (pt / (1 - pt + 1e-12))
        rows.append({"Threshold": pt, "Model": "Treat All", "Net_Benefit": nb})

    # "Treat None" baseline
    for pt in thresholds:
        rows.append({"Threshold": pt, "Model": "Treat None", "Net_Benefit": 0.0})

    # Each model
    for proba, name in zip(model_probas, model_names):
        proba_pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba
        for pt in thresholds:
            y_pred = (proba_pos >= pt).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            nb = tp / n - fp / n * (pt / (1 - pt + 1e-12))
            rows.append({"Threshold": pt, "Model": name, "Net_Benefit": nb})

    return pd.DataFrame(rows)


def run_decision_curve_experiment(dataset_name, data, cfg, frozen_params=None,
                                   n_dca_folds=3, verbose=True):
    """Run DCA across multiple folds and average net benefit curves.

    Each fold independently trains all models, calibrates predictions
    on the calibration set, and computes the per-threshold net benefit
    on the test set.  The final table reports the mean net benefit
    across folds at each threshold.
    """
    X_raw = data["X_raw"]
    y = data["y"]
    groups = data["groups"]
    folds = data["folds"][:n_dca_folds]
    thresholds = np.arange(0.01, 0.50, 0.01)

    model_names = ["JUCO", "JUCO-XGB", "XGBoost", "LightGBM", "MissForest+XGB"]
    # Accumulate net benefit per threshold per model
    nb_accum = {m: np.zeros(len(thresholds)) for m in model_names + ["Treat All", "Treat None"]}
    n_folds_done = 0

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if verbose:
            print(f"\n--- DCA Fold {fold_idx+1}/{len(folds)} ---")

        split = three_way_split(X_raw, y, groups, train_idx,
                                cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx)
        X_tr, y_tr = split["X_tr"], split["y_tr"]
        X_cal, y_cal = split["X_cal"], split["y_cal"]
        X_test = X_raw[test_idx].copy()
        y_test = y[test_idx].copy()

        feature_names = data["feature_names"]
        if cfg.max_features > 0 and X_tr.shape[1] > cfg.max_features:
            top_idx, _ = select_top_features(
                X_tr, y_tr, feature_names, cfg.max_features, cfg.seed + fold_idx)
            X_tr = X_tr[:, top_idx]
            X_cal = X_cal[:, top_idx]
            X_test = X_test[:, top_idx]

        X_tr_m, _ = apply_mnar_mask(X_tr, cfg.mnar_beta, cfg.mnar_gamma,
                                     cfg.mnar_rate, cfg.seed + fold_idx)
        sw, _ = compute_sample_weights(y_tr, verbose=False)

        # Collect calibrated probas for each model
        fold_probas = {}

        # JUCO
        juco = JUCOPipeline(cfg, params=frozen_params)
        juco.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
        pc = np.clip(juco.predict_proba(X_cal), 1e-7, 1-1e-7)
        pt = np.clip(juco.predict_proba(X_test), 1e-7, 1-1e-7)
        fold_probas["JUCO"] = isotonic_calibrate(pc, y_cal, pt)

        # JUCO-XGB
        jxgb = JUCOXGBPipeline(cfg, params=frozen_params)
        jxgb.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
        pc_jx = np.clip(jxgb.predict_proba(X_cal), 1e-7, 1-1e-7)
        pt_jx = np.clip(jxgb.predict_proba(X_test), 1e-7, 1-1e-7)
        fold_probas["JUCO-XGB"] = isotonic_calibrate(pc_jx, y_cal, pt_jx)

        # Baselines
        bl_map = {"XGBoost": BaselineXGBoost, "LightGBM": BaselineLGBM,
                   "MissForest+XGB": BaselineMissForestXGB}
        for bl_name, bl_cls in bl_map.items():
            bl = bl_cls(seed=cfg.seed)
            bl.fit(X_tr_m, y_tr)
            pc_bl = np.clip(bl.predict_proba(X_cal), 1e-7, 1-1e-7)
            pt_bl = np.clip(bl.predict_proba(X_test), 1e-7, 1-1e-7)
            fold_probas[bl_name] = isotonic_calibrate(pc_bl, y_cal, pt_bl)

        # Compute DCA for this fold
        dca_df = decision_curve_analysis(
            y_test,
            [fold_probas[m] for m in model_names],
            model_names,
            thresholds=thresholds,
        )

        # Accumulate
        for m in model_names + ["Treat All", "Treat None"]:
            sub = dca_df[dca_df["Model"] == m].sort_values("Threshold")
            nb_accum[m] += sub["Net_Benefit"].values
        n_folds_done += 1

    # Average
    for m in nb_accum:
        nb_accum[m] /= n_folds_done

    # Build final DCA table
    rows = []
    for i, pt in enumerate(thresholds):
        row = {"Threshold": pt}
        for m in model_names + ["Treat All", "Treat None"]:
            row[m] = nb_accum[m][i]
        rows.append(row)
    dca_table = pd.DataFrame(rows).set_index("Threshold")

    if verbose:
        print(f"\n{'='*80}")
        print(f" Decision Curve Analysis — Mean Net Benefit ({n_folds_done} folds)")
        print(f"{'='*80}")
        # Show a few key thresholds
        key_pts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        print(f"{'Threshold':<12}", end="")
        for m in model_names + ["Treat All"]:
            print(f"  {m:<16}", end="")
        print()
        for kp in key_pts:
            idx = np.argmin(np.abs(thresholds - kp))
            print(f"{thresholds[idx]:<12.2f}", end="")
            for m in model_names + ["Treat All"]:
                print(f"  {nb_accum[m][idx]:<16.4f}", end="")
            print()

    return dca_table


# ============================================================================
# 18. Checkpoint Save / Load Helpers
# ============================================================================
import pickle


def save_checkpoint(filepath, **kwargs):
    """Persist intermediate results to a pickle file.

    Used as a spot-instance safeguard during long GCP training runs:
    after each fold, the current results and frozen params are saved so
    that a pre-empted VM can resume from the last completed fold.
    Also saves per-fold predictions for offline DCA analysis.
    """
    with open(filepath, "wb") as f:
        pickle.dump(kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Checkpoint saved: {filepath} ({os.path.getsize(filepath)/1024:.0f} KB)")


def load_checkpoint(filepath):
    """Load previously saved checkpoint data from a pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    print(f"  Checkpoint loaded: {filepath}")
    return data
