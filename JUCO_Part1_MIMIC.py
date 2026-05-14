"""
JUCO Framework v2.0 â€” Part 1: MIMIC-IV Primary Experiment
=========================================================
Entry point for the MIMIC-IV arm of the JUCO evaluation pipeline.

This script executes two stages on the MIMIC-IV critical-care cohort
(Beth Israel Deaconess Medical Center, PhysioNet credentialed access):

  Stage A â€” **Main experiment** (Section 3 below)
      Two-stage Differential Evolution (DE) optimises fuzzy-tree
      hyper-parameters on Fold 1, then the optimised parameters are
      *frozen* and reused for the remaining 4 folds.  Every fold
      trains  JUCO (FDF backbone),  JUCO-XGB (XGBoost backbone), and
      five baselines (XGBoost, LightGBM, MissForest+XGB, MeanImp+XGB,
      TabNet).  Metrics recorded per fold: AUROC, AUPRC, Brier Score,
      ECE, F1, Sensitivity, Specificity, AURC, Risk@90.

  Stage B â€” **Ablation study** (Section 4 below)
      Runs six variants (Full, A1â€“A5) on 2 folds to isolate the
      contribution of each JUCO component (see ``run_ablation_studies``
      docstring in ``juco_core.py`` for variant descriptions).

All outputs are saved under ``./results/`` and a pickle checkpoint is
written so that Part 3 (``JUCO_Part3_Aggregate.py``) can produce the
final cross-dataset comparison tables without re-training.

How to run
----------
1. Place ``juco_core.py`` alongside this file.
2. Ensure MIMIC-IV CSVs are at ``./mimic-iv/`` (or update
   ``cfg.mimic_iv_path`` below).
3. Install dependencies::

       pip install numpy pandas scikit-learn lightgbm xgboost scipy \
                   matplotlib seaborn tqdm duckdb numba pytorch-tabnet

4. Execute::

       python JUCO_Part1_MIMIC.py

Outputs
-------
``./results/checkpoint_mimic.pkl``
    Serialised dict with keys *experiment_results*, *frozen_params*,
    and (if ablation succeeded) *ablation_tables*.  Consumed by Part 3.
``./results/results_MIMIC_IV.csv``
    Fold-level metric table (models Ã— metrics).  Imported into LaTeX.
``./results/ablation_MIMIC_IV.csv``
    Ablation comparison table (variants Ã— metrics).
"""

import os
import sys
import time

import numpy as np
import pandas as pd

from juco_core import (
    JUCOConfig,
    load_mimic_iv,
    preprocess_dataset,
    run_experiment,
    run_ablation_studies,
    summarize_results,
    save_checkpoint,
    METRIC_NAMES,
)


def main():
    """Run the complete MIMIC-IV experiment and ablation study."""
    t0 = time.time()

    # ====================================================================
    # 1. Configuration
    # ====================================================================
    #   Hyper-parameters below were tuned for a 32-vCPU / 128 GB GCP
    #   e2-highcpu-32 instance.  Adjust *forest_workers* and
    #   *de_workers* to match your hardware.
    # ====================================================================
    cfg = JUCOConfig()

    # -- Fuzzy Decision Forest size --
    cfg.n_trees            = 180     # trees in the final ensemble

    # -- DE Stage 1: coarse global search --
    cfg.de_s1_maxiter      = 8
    cfg.de_s1_popsize      = 8
    cfg.de_s1_trees        = 20      # lightweight proxy forest
    cfg.de_s1_samples      = 2000    # subsample for speed

    # -- DE Stage 2: fine local refinement --
    cfg.de_s2_maxiter      = 12
    cfg.de_s2_popsize      = 12
    cfg.de_s2_trees        = 40      # higher-fidelity proxy
    cfg.de_s2_samples      = 7000

    # -- Parallelism --
    cfg.forest_workers     = 28      # tree-building threads (â‰¤ vCPUs)
    cfg.de_workers         = 12      # parallel DE fitness evaluations

    # -- Folds --
    cfg.run_folds          = 0       # 0 â†’ run all 5 folds

    # -- Paths (update if data is elsewhere) --
    cfg.mimic_iv_path      = "./mimic-iv/"
    cfg.output_dir         = "./results/"
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 70)
    print("  JUCO Part 1: MIMIC-IV Experiment")
    print("=" * 70)

    # ====================================================================
    # 2. Data Loading & Preprocessing
    # ====================================================================
    #   load_mimic_iv  â†’ DuckDB SQL joins across hosp/ and icu/ CSVs,
    #                     returns a single patient-level DataFrame.
    #   preprocess_dataset â†’ GroupKFold splits, feature selection,
    #                        sample-weight computation.
    # ====================================================================
    print("\n[1/4] Loading MIMIC-IV data...")
    df_mimic = load_mimic_iv(cfg)

    print("\n[2/4] Preprocessing MIMIC-IV...")
    data = preprocess_dataset(df_mimic, cfg)
    # ====================================================================
    # 3. Main Experiment â€” Two-Stage DE + 5-Fold Cross-Validation
    # ====================================================================
    #   Fold 1:  DE optimises eta, tree depth, abstention Ï„, etc.
    #   Folds 2â€“5:  frozen params â†’ train + evaluate all models.
    #   Output:  per-fold dict of {model_name: {metric: value}}.
    # ====================================================================
    print("\n[3/4] Running MIMIC-IV experiment (two-stage DE + 5-fold CV)...")
    print("  This is the heaviest step. DE runs on Fold 1, then params are frozen.")
    print(f"  DE Stage 1: popsize={cfg.de_s1_popsize}, maxiter={cfg.de_s1_maxiter}, trees={cfg.de_s1_trees}")
    print(f"  DE Stage 2: popsize={cfg.de_s2_popsize}, maxiter={cfg.de_s2_maxiter}, trees={cfg.de_s2_trees}")
    print(f"  Final forest: n_trees={cfg.n_trees}, n_folds={cfg.n_folds}")

    experiment_results, frozen_params = run_experiment(
        "MIMIC-IV", data, cfg, verbose=True
    )

    # Summarise and persist fold-level metric table
    summary = summarize_results(experiment_results, "MIMIC-IV")
    print(f"\n{'='*100}")
    print(f" MIMIC-IV: Mean Â± Std across {cfg.n_folds} folds")
    print(f"{'='*100}")
    print(summary[METRIC_NAMES].to_string())
    summary.to_csv(os.path.join(cfg.output_dir, "results_MIMIC_IV.csv"))

    # ====================================================================
    # 4. Ablation Study (2 folds)
    # ====================================================================
    #   Six variants: Full, A1 (Sequential imputeâ†’RF), A2 (Crisp RF),
    #   A3 (Point imputation only), A4 (No abstention), A5 (No MNAR).
    #   Wrapped in try/except so a failure here does not discard the
    #   main experiment results already obtained above.
    # ====================================================================
    print("\n[4/4] Running MIMIC-IV ablation studies (2 folds)...")
    ablation_table = None
    try:
        ablation_table = run_ablation_studies("MIMIC-IV", data, cfg)
        print(f"\n{ablation_table.to_string()}")
        ablation_table.to_csv(os.path.join(cfg.output_dir, "ablation_MIMIC_IV.csv"))
    except Exception as e:
        print(f"\nCRITICAL ERROR in Ablation Study: {e}")
        import traceback
        traceback.print_exc()
        print("Saving main experiment results anyway...")

    # ====================================================================
    # 5. Save Checkpoint for Part 3 Aggregation
    # ====================================================================
    #   The checkpoint bundles everything Part 3 needs to produce
    #   cross-dataset comparison tables and LaTeX-ready figures
    #   without re-running any training.
    # ====================================================================
    ckpt_data = {
        "experiment_results": {"MIMIC-IV": experiment_results},
        "frozen_params":      {"MIMIC-IV": frozen_params},
        "dataset_name":       "MIMIC-IV",
    }
    if ablation_table is not None:
        ckpt_data["ablation_tables"] = {"MIMIC-IV": ablation_table}

    save_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint_mimic.pkl"),
        **ckpt_data,
    )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Part 1 COMPLETE â€” Total time: {elapsed / 60:.1f} min")
    print(f"  Results saved to: {cfg.output_dir}")
    print(f"  Checkpoint: checkpoint_mimic.pkl")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
