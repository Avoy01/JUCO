"""
Debug script: load both datasets via the same pipeline as Part1 / Part2,
then print full cohort statistics (no training, no CV — stops after
preprocess_dataset so it runs in seconds once data is loaded).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np

from juco_core import (
    JUCOConfig,
    load_mimic_iv,
    load_eicu,
    preprocess_dataset,
)


def print_cohort(label, df, data):
    """Print cohort statistics from the loaded DataFrame and preprocessed data."""
    y        = data["y"]
    n_stays  = len(y)
    n_deaths = int(y.sum())
    n_feat   = data["X_raw"].shape[1]
    n_unique = len(np.unique(data["groups"]))

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total ICU stays             : {n_stays:,}")
    print(f"  Unique patients             : {n_unique:,}")
    print(f"  In-hospital deaths          : {n_deaths:,}")
    print(f"  Crude mortality rate        : {n_deaths / n_stays:.2%}")
    print(f"  Total features (after MI)   : {n_feat}")
    print(f"  CV folds                    : {len(data['folds'])}")

    # per-fold test-set size (GroupKFold, non-overlapping)
    print(f"\n  Per-fold test-set sizes:")
    for i, (_, test_idx) in enumerate(data["folds"]):
        yt = y[test_idx]
        print(f"    Fold {i+1}: {len(test_idx):7,} stays | "
              f"{int(yt.sum()):5,} deaths | {yt.mean():.2%} mortality")


# ── MIMIC-IV ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Loading MIMIC-IV...")
print("="*60)
cfg_mimic = JUCOConfig()
cfg_mimic.mimic_iv_path = "./mimic-iv/"
cfg_mimic.output_dir    = "./results/"

df_mimic = load_mimic_iv(cfg_mimic)
print(f"\n[DEBUG] Raw DataFrame shape : {df_mimic.shape}")
print(f"[DEBUG] Columns             : {list(df_mimic.columns[:8])} ...")
print(f"[DEBUG] Mortality col dtype : {df_mimic['mortality'].dtype}")

print("\nPreprocessing MIMIC-IV...")
data_mimic = preprocess_dataset(df_mimic, cfg_mimic)

print_cohort("MIMIC-IV Cohort", df_mimic, data_mimic)


# ── eICU ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Loading eICU...")
print("="*60)
cfg_eicu = JUCOConfig()
cfg_eicu.eicu_path   = "./eicu/"
cfg_eicu.output_dir  = "./results/"

df_eicu = load_eicu(cfg_eicu)
print(f"\n[DEBUG] Raw DataFrame shape : {df_eicu.shape}")
print(f"[DEBUG] Columns             : {list(df_eicu.columns[:8])} ...")
print(f"[DEBUG] Mortality col dtype : {df_eicu['mortality'].dtype}")

print("\nPreprocessing eICU...")
data_eicu = preprocess_dataset(df_eicu, cfg_eicu)

print_cohort("eICU Cohort", df_eicu, data_eicu)

