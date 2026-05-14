#!/usr/bin/env python3
"""
JUCO Part 1B — Robustness & Clinical Utility Experiments
=========================================================
Run this script AFTER Part 1 (JUCO_Part1_MIMIC.py) has completed.
It reuses the frozen Differential Evolution (DE) hyperparameters and
the preprocessed MIMIC-IV data saved during Part 1, so
**no DE optimization is repeated**.

Three experiments are executed sequentially:

  Experiment 0 — JUCO-XGB Full Evaluation
      Train the XGBoost variant of JUCO (JUCO-XGB) under the same
      5-fold stratified group CV used for all Part 1 baselines.
      Reports AUROC, AUPRC, Brier, ECE, AURC per fold, then
      mean ± std.  Saves results_JUCO_XGB_MIMIC_IV.csv.

  Experiment 1 — Missingness Stress Test
      Systematically inject Missing-Not-At-Random (MNAR) values at
      rates 10 %–70 % and measure how each model's AUROC degrades.
      Uses 2 CV folds per rate.  Saves missingness_stress_MIMIC_IV.csv.

  Experiment 2 — Decision Curve Analysis (DCA)
      Computes Net Benefit vs. decision-threshold curves for all
      models using **saved fold predictions** from Part 1 (no
      retraining).  Requires predictions_MIMIC_IV.pkl generated
      automatically by Part 1's run_experiment().  Saves dca_MIMIC_IV.csv.

Outputs (all written to ./results/):
  - results_JUCO_XGB_MIMIC_IV.csv
  - missingness_stress_MIMIC_IV.csv
  - dca_MIMIC_IV.csv
  - checkpoint_mimic_robustness.pkl  (combined checkpoint)
"""

# ── Standard library ────────────────────────────────────────────────
import os
import sys
import time

# ── Third-party ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── JUCO framework (shared across all pipeline stages) ──────────────
from juco_core import (
    # Configuration & I/O
    JUCOConfig, load_mimic_iv, preprocess_dataset,
    load_checkpoint, save_checkpoint,
    # Model pipelines
    JUCOXGBPipeline,
    # Data helpers
    three_way_split, select_top_features, apply_mnar_mask,
    compute_sample_weights,
    # Evaluation & calibration
    evaluate_model, isotonic_calibrate,
    SafeAbstentionProtocol, summarize_results, METRIC_NAMES,
    # Experiment helpers
    run_missingness_stress_test, decision_curve_analysis,
)


# ====================================================================
#  MAIN
# ====================================================================
def main():
    wall_start = time.time()

    # ── 1. Configuration ────────────────────────────────────────────
    # Mirror the same settings used in Part 1 so that data loading,
    # preprocessing, and fold splits are identical.
    cfg = JUCOConfig()
    cfg.n_trees        = 180        # Fuzzy Decision Forest tree count
    cfg.forest_workers = 28         # parallel workers for forest fitting
    cfg.de_workers     = 12         # parallel workers for DE (unused here)
    cfg.mimic_iv_path  = "./mimic-iv/"
    cfg.output_dir     = "./results/"
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 70)
    print("  JUCO Part 1B — Robustness & Clinical Utility Experiments")
    print("=" * 70)

    # ── 2. Load frozen DE hyperparameters from Part 1 ───────────────
    # The checkpoint stores the best DE-optimised parameters found
    # during Part 1.  We freeze them here to ensure every experiment
    # uses the exact same model configuration.
    ckpt_path = os.path.join(cfg.output_dir, "checkpoint_mimic.pkl")
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path)
        frozen_params = ckpt.get("frozen_params", {}).get("MIMIC-IV", None)
        if frozen_params:
            print(f"  Loaded frozen DE params: {frozen_params}")
        else:
            print("  WARNING: No frozen params in checkpoint — using defaults.")
    else:
        print(f"  WARNING: {ckpt_path} not found — using default params.")
        frozen_params = None

    # ── 3. Load & preprocess MIMIC-IV ───────────────────────────────
    # Uses the same ETL pipeline as Part 1 (DuckDB SQL extraction →
    # feature engineering → group-aware stratified k-fold split).
    print("\n[1/3] Loading MIMIC-IV data …")
    df_mimic = load_mimic_iv(cfg)

    print("[2/3] Preprocessing (feature engineering + fold split) …")
    data = preprocess_dataset(df_mimic, cfg)

    # Unpack the preprocessed arrays used across experiments
    X_raw          = data["X_raw"]        # (N, F) raw feature matrix
    y              = data["y"]            # (N,)   binary mortality labels
    groups         = data["groups"]       # (N,)   patient group IDs
    folds          = data["folds"]        # list of (train_idx, test_idx)
    feature_names  = data["feature_names"]

    # ================================================================
    #  EXPERIMENT 0 — JUCO-XGB Full 5-Fold Cross-Validation
    # ================================================================
    # JUCO-XGB replaces the Fuzzy Decision Forest backend with XGBoost
    # while keeping the full JUCO pipeline (Quantile Imputation →
    # Fuzzy Transform → XGBoost → Isotonic Calibration → Safe
    # Abstention).  This evaluates whether a gradient-boosted tree
    # backend can improve over the default forest.
    print("\n" + "=" * 70)
    print("  Experiment 0: JUCO-XGB Full Evaluation (5-fold CV)")
    print("  Same protocol as Part 1 baselines — frozen DE params")
    print("=" * 70)

    jxgb_results = {metric: [] for metric in METRIC_NAMES}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_t0 = time.time()
        print(f"\n--- JUCO-XGB  Fold {fold_idx + 1}/{len(folds)} ---")

        # 0a. Three-way split: train / calibration / test
        #     Calibration set is used for isotonic recalibration and
        #     abstention-threshold tuning.
        split = three_way_split(
            X_raw, y, groups, train_idx,
            cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx,
        )
        X_tr,  y_tr  = split["X_tr"],  split["y_tr"]
        X_cal, y_cal = split["X_cal"], split["y_cal"]
        X_test       = X_raw[test_idx].copy()
        y_test       = y[test_idx].copy()

        # 0b. Feature selection — keep only the top-k most informative
        #     features (mutual-information ranking) to reduce noise.
        if cfg.max_features > 0 and X_tr.shape[1] > cfg.max_features:
            top_idx, _ = select_top_features(
                X_tr, y_tr, feature_names,
                cfg.max_features, cfg.seed + fold_idx,
            )
            X_tr   = X_tr[:, top_idx]
            X_cal  = X_cal[:, top_idx]
            X_test = X_test[:, top_idx]

        # 0c. Inject MNAR missingness at the configured baseline rate.
        #     This simulates realistic ICU charting patterns where
        #     abnormal values are more likely to be missing.
        X_tr_m, _ = apply_mnar_mask(
            X_tr, cfg.mnar_beta, cfg.mnar_gamma,
            cfg.mnar_rate, cfg.seed + fold_idx,
        )

        # 0d. Class-balanced sample weights (addresses ~90/10 imbalance)
        sw, _ = compute_sample_weights(y_tr, verbose=False)

        # 0e. Train JUCO-XGB with frozen hyperparameters
        jxgb = JUCOXGBPipeline(cfg, params=frozen_params)
        jxgb.fit(X_tr_m, y_tr, sample_weights=sw, verbose=(fold_idx == 0))

        # 0f. Post-hoc isotonic calibration: fit on calibration set,
        #     then transform test predictions to well-calibrated probs.
        proba_cal      = np.clip(jxgb.predict_proba(X_cal),  1e-7, 1 - 1e-7)
        proba_test_raw = np.clip(jxgb.predict_proba(X_test), 1e-7, 1 - 1e-7)
        proba_test     = isotonic_calibrate(proba_cal, y_cal, proba_test_raw)

        # 0g. Safe Abstention — find threshold tau below which the
        #     model should abstain from a prediction (defer to human).
        proba_cal_cal = isotonic_calibrate(proba_cal, y_cal, proba_cal)
        abstention    = SafeAbstentionProtocol(cfg.max_error_rate)
        tau           = abstention.calibrate_threshold(proba_cal_cal, y_cal)

        # 0h. Compute all evaluation metrics for this fold
        metrics = evaluate_model(proba_test, y_test)
        for m, v in metrics.items():
            jxgb_results[m].append(v)

        fold_elapsed = time.time() - fold_t0
        print(f"  AUROC={metrics['AUROC']:.4f}  ECE={metrics['ECE']:.4f}  "
              f"AURC={metrics['AURC']:.4f}  tau={tau:.4f}  "
              f"({fold_elapsed / 60:.1f} min)")

    # ── Experiment 0 summary ────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print(f"  JUCO-XGB: Mean ± Std across {len(folds)} folds")
    print(f"{'=' * 80}")
    jxgb_summary = summarize_results({"JUCO-XGB": jxgb_results}, "MIMIC-IV")
    print(jxgb_summary[METRIC_NAMES].to_string())

    jxgb_csv = os.path.join(cfg.output_dir, "results_JUCO_XGB_MIMIC_IV.csv")
    jxgb_summary.to_csv(jxgb_csv)
    print(f"\n  Saved → {jxgb_csv}")

    # ================================================================
    #  EXPERIMENT 1 — Missingness Stress Test
    # ================================================================
    # Progressively increase the MNAR injection rate from 10 % to 70 %
    # and record AUROC for every model (JUCO, JUCO-XGB, and baselines).
    # This reveals how robust each method is when the fraction of
    # missing lab values grows — a common real-world scenario in ICUs
    # where sicker patients have more data gaps.
    print("\n" + "=" * 70)
    print("  Experiment 1: Missingness Stress Test")
    print("  MNAR rates: 10 %, 20 %, 30 %, 40 %, 50 %, 60 %, 70 %")
    print("  2 CV folds per rate (frozen DE params)")
    print("=" * 70)

    stress_table, stress_raw = run_missingness_stress_test(
        "MIMIC-IV", data, cfg,
        frozen_params=frozen_params,
        miss_rates=(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70),
        n_stress_folds=2,
        verbose=True,
    )

    stress_csv = os.path.join(cfg.output_dir, "missingness_stress_MIMIC_IV.csv")
    stress_table.to_csv(stress_csv)
    print(f"\n  Saved → {stress_csv}")

    # ================================================================
    #  EXPERIMENT 2 — Decision Curve Analysis (from saved predictions)
    # ================================================================
    # DCA evaluates clinical utility by computing Net Benefit at every
    # decision threshold p_t ∈ [0.01, 0.49].  A model with higher Net
    # Benefit at a given threshold offers more clinical value than the
    # default "treat all" or "treat none" strategies.
    #
    # Instead of retraining models, we load the calibrated per-fold
    # predicted probabilities that Part 1's run_experiment() saves
    # automatically.  This makes the DCA step very fast (~seconds).
    N_DCA_FOLDS = 2
    DCA_MODELS  = ["JUCO", "JUCO-XGB", "XGBoost", "LightGBM", "MissForest+XGB"]
    thresholds  = np.arange(0.01, 0.50, 0.01)

    print("\n" + "=" * 70)
    print("  Experiment 2: Decision Curve Analysis")
    print(f"  Net Benefit vs Threshold ({N_DCA_FOLDS} folds, saved predictions)")
    print("=" * 70)

    # Load saved predictions (generated automatically by Part 1)
    pred_path = os.path.join(cfg.output_dir, "predictions_MIMIC_IV.pkl")
    if not os.path.exists(pred_path):
        print(f"  ERROR: {pred_path} not found.")
        print("  Please run JUCO_Part1_MIMIC.py first.")
        sys.exit(1)

    print(f"  Loading predictions from {pred_path} …")
    dca_t0 = time.time()

    pred_ckpt        = load_checkpoint(pred_path)
    fold_predictions = pred_ckpt["fold_predictions"][:N_DCA_FOLDS]

    # Compute per-fold Net Benefit curves for every model
    dca_fold_data = {}
    for fp in fold_predictions:
        fold_idx = fp["fold"]
        y_test   = fp["y_true"]

        # Check which models have saved predictions in this fold
        available = [m for m in DCA_MODELS if m in fp]
        missing   = [m for m in DCA_MODELS if m not in fp]
        if missing:
            print(f"  WARNING: fold {fold_idx} missing predictions for {missing}")

        # Run DCA — returns DataFrame with columns [Model, Threshold, Net_Benefit]
        fold_probas = [fp[m] for m in available]
        dca_df = decision_curve_analysis(
            y_test, fold_probas, available, thresholds=thresholds,
        )

        # Extract Net Benefit arrays keyed by model name
        fold_nb = {}
        for m in available + ["Treat All", "Treat None"]:
            sub = dca_df[dca_df["Model"] == m].sort_values("Threshold")
            fold_nb[m] = (sub["Net_Benefit"].values
                          if len(sub) > 0
                          else np.zeros(len(thresholds)))
        dca_fold_data[fold_idx] = fold_nb

    # Average Net Benefit across all completed folds
    all_models  = DCA_MODELS + ["Treat All", "Treat None"]
    nb_accum    = {m: np.zeros(len(thresholds)) for m in all_models}
    n_folds_done = len(dca_fold_data)

    for fidx in dca_fold_data:
        for m in all_models:
            if m in dca_fold_data[fidx]:
                nb_accum[m] += dca_fold_data[fidx][m]
    if n_folds_done > 0:
        for m in nb_accum:
            nb_accum[m] /= n_folds_done

    # Assemble into a tidy CSV-friendly DataFrame
    dca_rows = []
    for i, pt_val in enumerate(thresholds):
        row = {"Threshold": pt_val}
        for m in all_models:
            row[m] = nb_accum[m][i]
        dca_rows.append(row)
    dca_table = pd.DataFrame(dca_rows).set_index("Threshold")

    # Print a compact summary at clinically relevant thresholds
    dca_elapsed = time.time() - dca_t0
    print(f"  DCA finished in {dca_elapsed:.1f} s (no retraining)")

    print(f"\n{'=' * 80}")
    print(f"  Decision Curve Analysis — Mean Net Benefit ({n_folds_done} folds)")
    print(f"{'=' * 80}")
    key_pts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    header  = f"{'Threshold':<12}"
    for m in DCA_MODELS + ["Treat All"]:
        header += f"  {m:<16}"
    print(header)
    for kp in key_pts:
        idx  = np.argmin(np.abs(thresholds - kp))
        line = f"{thresholds[idx]:<12.2f}"
        for m in DCA_MODELS + ["Treat All"]:
            line += f"  {nb_accum[m][idx]:<16.4f}"
        print(line)

    dca_csv = os.path.join(cfg.output_dir, "dca_MIMIC_IV.csv")
    dca_table.to_csv(dca_csv)
    print(f"\n  Saved → {dca_csv}")

    # ================================================================
    #  SAVE COMBINED CHECKPOINT
    # ================================================================
    # Persist all experiment outputs in a single pickle so that
    # downstream scripts (Part 3 aggregation, notebook) can consume
    # them without re-running anything.
    save_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint_mimic_robustness.pkl"),
        juco_xgb_results=jxgb_results,
        juco_xgb_summary=jxgb_summary,
        stress_test={"table": stress_table, "raw": stress_raw},
        dca={"table": dca_table, "raw": dca_fold_data},
        frozen_params=frozen_params,
    )

    wall_elapsed = time.time() - wall_start
    print(f"\n{'=' * 70}")
    print(f"  Part 1B COMPLETE — wall time: {wall_elapsed / 60:.1f} min")
    print(f"  All results saved to: {cfg.output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
