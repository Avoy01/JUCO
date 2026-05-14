"""
JUCO Framework v2.0 — Part 2: eICU External Validation
=======================================================
External-validation arm of the JUCO evaluation pipeline.

This script evaluates on the eICU Collaborative Research Database
(208 US hospitals, Pollard et al., Scientific Data, 2018) using
hyper-parameters that were optimised **exclusively on MIMIC-IV** in
Part 1.  No DE search is performed here — the frozen parameter vector
from Part 1 is loaded from ``checkpoint_mimic.pkl`` and applied as-is.
This design prevents information leakage and establishes genuine
cross-institution generalisability.

Pipeline overview
-----------------
1. Load frozen DE parameters from ``checkpoint_mimic.pkl``.
2. Load and preprocess eICU (schema-harmonised to match MIMIC-IV).
3. Run 5-fold CV with frozen params (no optimisation).
   - Results are **checkpointed per fold** so a pre-empted GCP VM
     can resume from the last completed fold.
4. Decision Curve Analysis (DCA) — computed from saved per-fold
   predictions (no retraining).  Falls back to training if the
   prediction file is unavailable.
5. Save ``checkpoint_eicu.pkl`` for Part 3 aggregation.

How to run
----------
1. Execute ``JUCO_Part1_MIMIC.py`` first (produces
   ``checkpoint_mimic.pkl``).
2. Ensure eICU CSVs are at ``./eicu/`` (or update
   ``cfg.eicu_path``).
3. Execute::

       python JUCO_Part2_eICU.py

Outputs
-------
``./results/checkpoint_eicu.pkl``
    Experiment results + DCA tables for Part 3 aggregation.
``./results/results_eICU.csv``
    Fold-level metric table (models × metrics).
``./results/dca_eICU.csv``
    Mean net-benefit table across DCA folds.
``./results/predictions_eICU.pkl``
    Per-fold calibrated predictions for offline DCA reuse.
"""

import os
import sys
import time
import pickle

import numpy as np
import pandas as pd

from juco_core import (
    JUCOConfig,
    load_eicu,
    preprocess_dataset,
    summarize_results,
    save_checkpoint,
    load_checkpoint,
    METRIC_NAMES,
    BASELINES,
    decision_curve_analysis,
    JUCOPipeline,
    JUCOXGBPipeline,
    BaselineXGBoost,
    BaselineLGBM,
    BaselineMissForestXGB,
    BaselineMeanXGB,
    three_way_split,
    select_top_features,
    apply_mnar_mask,
    compute_sample_weights,
    evaluate_model,
    isotonic_calibrate,
    SafeAbstentionProtocol,
)


# ======================================================================
# Per-Fold Checkpoint Helpers (VM crash recovery)
# ======================================================================
#   On a pre-emptible GCP instance each fold takes ~20 min.  These
#   helpers serialise {completed_folds, all_results, fold_predictions}
#   after every fold so that a re-started script skips finished folds.
# ======================================================================


def load_progress(progress_file):
    """Load per-fold progress checkpoint, or return None if absent."""
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            return pickle.load(f)
    return None


def save_progress(progress_file, progress_data):
    """Persist per-fold progress checkpoint to *progress_file*."""
    with open(progress_file, "wb") as f:
        pickle.dump(progress_data, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """Run eICU external validation using frozen MIMIC-IV parameters."""
    t0 = time.time()

    # ====================================================================
    # 1. Configuration
    # ====================================================================
    #   Only forest / parallelism settings are specified here; the
    #   fuzzy-tree hyper-parameters come from frozen_params (Part 1).
    # ====================================================================
    cfg = JUCOConfig()
    cfg.n_trees            = 180     # must match Part 1 forest size
    cfg.forest_workers     = 28      # tree-building threads
    cfg.de_workers         = 12      # (unused — no DE on eICU)
    cfg.eicu_path          = "./eicu/"
    cfg.output_dir         = "./results/"
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 70)
    print("  JUCO Part 2: eICU External Validation")
    print("  (Using frozen DE params from MIMIC-IV — no data leakage)")
    print("=" * 70)

    # ====================================================================
    # 2. Load Frozen DE Parameters from MIMIC-IV
    # ====================================================================
    #   External-validation protocol: hyper-parameters are optimised
    #   once on MIMIC-IV (Part 1) and never touched again.  If the
    #   checkpoint is missing, a warning is printed and defaults are
    #   used — but this weakens the external-validation claim.
    # ====================================================================
    mimic_ckpt_path = os.path.join(cfg.output_dir, "checkpoint_mimic.pkl")
    if os.path.exists(mimic_ckpt_path):
        ckpt = load_checkpoint(mimic_ckpt_path)
        frozen_params = ckpt.get("frozen_params", {}).get("MIMIC-IV", None)
        if frozen_params:
            print(f"  Loaded frozen DE params from MIMIC-IV: {frozen_params}")
        else:
            print("  WARNING: No frozen params in MIMIC checkpoint. DE will run on eICU.")
    else:
        print(f"  WARNING: {mimic_ckpt_path} not found. DE will run on eICU.")
        frozen_params = None

    # ====================================================================
    # 3. Data Loading & Preprocessing
    # ====================================================================
    #   load_eicu harmonises column names to match MIMIC-IV so that
    #   the same preprocessing pipeline can be applied unchanged.
    # ====================================================================
    print("\n[1/4] Loading eICU data...")
    df_eicu = load_eicu(cfg)

    print("\n[2/4] Preprocessing eICU...")
    data = preprocess_dataset(df_eicu, cfg)

    # ====================================================================
    # 4. Main Experiment — 5-Fold CV with Frozen Parameters
    # ====================================================================
    #   No DE is run.  Each fold: train JUCO, JUCO-XGB, and 5 baselines
    #   with frozen params → isotonic calibration on cal set → evaluate
    #   on test set.  Per-fold predictions are saved for offline DCA.
    #
    #   Checkpoint-resume: after each fold, {completed_folds,
    #   all_results, fold_predictions} are serialised so a pre-empted
    #   VM can skip finished folds on restart.
    # ====================================================================
    print("\n[3/4] Running eICU experiment (5-fold CV, frozen MIMIC-IV params)...")
    if frozen_params:
        print("  DE skipped — using MIMIC-IV frozen params for external validation.")
    else:
        print("  WARNING: No frozen params. Using defaults.")
    print(f"  Final forest: n_trees={cfg.n_trees}, n_folds={cfg.n_folds}")

    # --- Per-fold checkpoint path ---
    exp_progress_file = os.path.join(cfg.output_dir, "eicu_exp_progress.pkl")

    # Resume from last completed fold if checkpoint exists
    progress = load_progress(exp_progress_file)
    if progress:
        completed_folds = progress["completed_folds"]
        all_results = progress["all_results"]
        fold_predictions = progress["fold_predictions"]
        print(f"  Resumed: {len(completed_folds)} folds already completed")
    else:
        completed_folds = set()
        model_names = ["JUCO", "JUCO-XGB"] + list(BASELINES.keys())
        all_results = {m: {metric: [] for metric in METRIC_NAMES} for m in model_names}
        fold_predictions = []

    X_raw = data["X_raw"]
    y = data["y"]
    groups = data["groups"]
    folds = data["folds"]

    model_names = ["JUCO", "JUCO-XGB"] + list(BASELINES.keys())
    total_folds = len(folds)
    print(f"  Total folds: {total_folds}, completed: {len(completed_folds)}, "
          f"remaining: {total_folds - len(completed_folds)}")

    exp_t0 = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        if fold_idx in completed_folds:
            print(f"\n--- Fold {fold_idx+1}/{total_folds} — already done, skipping ---")
            continue

        fold_t0 = time.time()
        print(f"\n{'='*70}")
        print(f" eICU -- Fold {fold_idx + 1}/{total_folds}")
        print(f"{'='*70}")

        split = three_way_split(X_raw, y, groups, train_idx,
                                cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx)
        X_tr, y_tr = split["X_tr"], split["y_tr"]
        X_val, y_val = split["X_val"], split["y_val"]
        X_cal, y_cal = split["X_cal"], split["y_cal"]
        X_test = X_raw[test_idx].copy()
        y_test = y[test_idx].copy()

        # Per-fold feature selection
        feature_names = data["feature_names"]
        if cfg.max_features > 0 and X_tr.shape[1] > cfg.max_features:
            top_idx, _ = select_top_features(
                X_tr, y_tr, feature_names, cfg.max_features, cfg.seed + fold_idx)
            X_tr = X_tr[:, top_idx]
            X_val = X_val[:, top_idx]
            X_cal = X_cal[:, top_idx]
            X_test = X_test[:, top_idx]

        # ---- MNAR masking (train + val only; test stays natural) ----
        print("MNAR masking (train + val only)...")
        X_tr_m, _ = apply_mnar_mask(X_tr, cfg.mnar_beta, cfg.mnar_gamma,
                                     cfg.mnar_rate, cfg.seed + fold_idx)
        X_val_m, _ = apply_mnar_mask(X_val, cfg.mnar_beta, cfg.mnar_gamma,
                                      cfg.mnar_rate, cfg.seed + fold_idx + 50)
        sample_weights, cw_dict = compute_sample_weights(y_tr)

        print(f"Using frozen params: {frozen_params}")

        # ---- JUCO (Fuzzy Decision Forest backbone) ----
        print(f"\nTraining JUCO (frozen DE params)...")
        juco = JUCOPipeline(cfg, params=frozen_params)
        juco.fit(X_tr_m, y_tr, sample_weights=sample_weights, verbose=(fold_idx == 0))
        proba_cal_juco = np.clip(juco.predict_proba(X_cal), 1e-7, 1 - 1e-7)
        proba_test_juco_raw = np.clip(juco.predict_proba(X_test), 1e-7, 1 - 1e-7)
        proba_test_juco = isotonic_calibrate(proba_cal_juco, y_cal, proba_test_juco_raw)

        # Calibrate abstention threshold on calibration set
        proba_cal_juco_cal = isotonic_calibrate(proba_cal_juco, y_cal, proba_cal_juco)
        abstention = SafeAbstentionProtocol(cfg.max_error_rate)
        tau = abstention.calibrate_threshold(proba_cal_juco_cal, y_cal)
        print(f"  Abstention τ = {tau:.4f}")

        metrics_juco = evaluate_model(proba_test_juco, y_test)
        for m, v in metrics_juco.items():
            all_results["JUCO"][m].append(v)
        print(f"  JUCO -- AUROC: {metrics_juco['AUROC']:.4f}, ECE: {metrics_juco['ECE']:.4f}")

        # ---- JUCO-XGB (XGBoost backbone with fuzzy features) ----
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

        # ---- Baselines (XGBoost, LightGBM, MissForest+XGB, MeanImp+XGB, TabNet) ----
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

        # ---- Save per-fold predictions for offline DCA ----
        fold_pred = {"fold": fold_idx, "y_true": y_test,
                     "JUCO": proba_test_juco, "JUCO-XGB": proba_test_jxgb}
        for bl_name in BASELINES:
            fold_pred[bl_name] = bl_probas[bl_name]
        fold_predictions.append(fold_pred)

        # ---- CHECKPOINT: serialise progress after each fold ----
        completed_folds.add(fold_idx)
        save_progress(exp_progress_file, {
            "completed_folds": completed_folds,
            "all_results": all_results,
            "fold_predictions": fold_predictions,
        })

        fold_elapsed = time.time() - fold_t0
        total_elapsed = time.time() - exp_t0
        print(f"\n  [Timer] Fold {fold_idx+1} took {fold_elapsed/60:.1f} min  |  "
              f"Total elapsed: {total_elapsed/60:.1f} min")

    # Persist all per-fold predictions for DCA reuse
    pred_path = os.path.join(cfg.output_dir, "predictions_eICU.pkl")
    save_checkpoint(pred_path, fold_predictions=fold_predictions)
    print(f"  Predictions saved: {pred_path}")

    experiment_results = all_results
    final_params = frozen_params

    # Summarise and persist fold-level metric table
    summary = summarize_results(experiment_results, "eICU")
    print(f"\n{'='*100}")
    print(f" eICU: Mean ± Std across {cfg.n_folds} folds")
    print(f"{'='*100}")
    print(summary[METRIC_NAMES].to_string())
    summary.to_csv(os.path.join(cfg.output_dir, "results_eICU.csv"))

    # ====================================================================
    # 5. Decision Curve Analysis (Vickers & Elkin, 2006)
    # ====================================================================
    #   Fast path: if per-fold predictions exist on disk, DCA is
    #   computed from those calibrated probabilities (no retraining).
    #   Fallback: retrain all models for n_dca_folds folds.
    # ====================================================================
    n_dca_folds = 2
    dca_model_names = ["JUCO", "JUCO-XGB", "XGBoost", "LightGBM", "MissForest+XGB"]
    thresholds = np.arange(0.01, 0.50, 0.01)

    print("\n" + "=" * 70)
    print("  Decision Curve Analysis (eICU)")
    print(f"  Net Benefit vs Threshold ({n_dca_folds} folds)")
    print("=" * 70)

    pred_path = os.path.join(cfg.output_dir, "predictions_eICU.pkl")
    dca_fold_data = {}

    if os.path.exists(pred_path):
        # ---- FAST PATH: reuse saved per-fold predictions ----
        print(f"  Found saved predictions: {pred_path}")
        print("  Running DCA from saved predictions (no retraining)...")
        dca_t0 = time.time()

        pred_ckpt = load_checkpoint(pred_path)
        fold_predictions = pred_ckpt["fold_predictions"][:n_dca_folds]

        for fp in fold_predictions:
            fold_idx = fp["fold"]
            y_test = fp["y_true"]
            available = [m for m in dca_model_names if m in fp]
            fold_probas = [fp[m] for m in available]

            dca_df = decision_curve_analysis(y_test, fold_probas, available,
                                              thresholds=thresholds)
            fold_nb = {}
            for m in dca_model_names + ["Treat All", "Treat None"]:
                sub = dca_df[dca_df["Model"] == m].sort_values("Threshold")
                fold_nb[m] = sub["Net_Benefit"].values if len(sub) > 0 else np.zeros(len(thresholds))
            dca_fold_data[fold_idx] = fold_nb

        print(f"  DCA computed in {time.time() - dca_t0:.1f} seconds (no retraining)")

    else:
        # ---- FALLBACK: retrain models for DCA ----
        print(f"  No saved predictions found. Training models for DCA...")

        dca_folds = data["folds"][:n_dca_folds]
        X_raw_dca = data["X_raw"]
        y_dca = data["y"]
        groups_dca = data["groups"]
        use_params = final_params if final_params else frozen_params

        for fold_idx, (train_idx, test_idx) in enumerate(dca_folds):
            fold_t0 = time.time()
            print(f"\n--- DCA Fold {fold_idx+1}/{len(dca_folds)} ---")

            split = three_way_split(X_raw_dca, y_dca, groups_dca, train_idx,
                                    cfg.val_ratio, cfg.cal_ratio, cfg.seed + fold_idx)
            X_tr, y_tr = split["X_tr"], split["y_tr"]
            X_cal, y_cal = split["X_cal"], split["y_cal"]
            X_test = X_raw_dca[test_idx].copy()
            y_test = y_dca[test_idx].copy()

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

            fold_probas = {}

            print("    Training JUCO...", end=" ", flush=True)
            juco = JUCOPipeline(cfg, params=use_params)
            juco.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
            pc = np.clip(juco.predict_proba(X_cal), 1e-7, 1 - 1e-7)
            pt = np.clip(juco.predict_proba(X_test), 1e-7, 1 - 1e-7)
            fold_probas["JUCO"] = isotonic_calibrate(pc, y_cal, pt)
            print("done")

            print("    Training JUCO-XGB...", end=" ", flush=True)
            jxgb = JUCOXGBPipeline(cfg, params=use_params)
            jxgb.fit(X_tr_m, y_tr, sample_weights=sw, verbose=False)
            pc_jx = np.clip(jxgb.predict_proba(X_cal), 1e-7, 1 - 1e-7)
            pt_jx = np.clip(jxgb.predict_proba(X_test), 1e-7, 1 - 1e-7)
            fold_probas["JUCO-XGB"] = isotonic_calibrate(pc_jx, y_cal, pt_jx)
            print("done")

            bl_map = {"XGBoost": BaselineXGBoost, "LightGBM": BaselineLGBM,
                       "MissForest+XGB": BaselineMissForestXGB}
            for bl_name, bl_cls in bl_map.items():
                print(f"    Training {bl_name}...", end=" ", flush=True)
                bl = bl_cls(seed=cfg.seed)
                bl.fit(X_tr_m, y_tr)
                pc_bl = np.clip(bl.predict_proba(X_cal), 1e-7, 1 - 1e-7)
                pt_bl = np.clip(bl.predict_proba(X_test), 1e-7, 1 - 1e-7)
                fold_probas[bl_name] = isotonic_calibrate(pc_bl, y_cal, pt_bl)
                print("done")

            dca_df = decision_curve_analysis(
                y_test,
                [fold_probas[m] for m in dca_model_names],
                dca_model_names,
                thresholds=thresholds,
            )

            fold_nb = {}
            for m in dca_model_names + ["Treat All", "Treat None"]:
                sub = dca_df[dca_df["Model"] == m].sort_values("Threshold")
                fold_nb[m] = sub["Net_Benefit"].values
            dca_fold_data[fold_idx] = fold_nb

            print(f"  DCA Fold {fold_idx+1} done ({(time.time()-fold_t0)/60:.1f} min)")

    # ---- Average net benefit across folds ----
    all_models = dca_model_names + ["Treat All", "Treat None"]
    nb_accum = {m: np.zeros(len(thresholds)) for m in all_models}
    n_folds_done = len(dca_fold_data)
    for fidx in dca_fold_data:
        for m in all_models:
            nb_accum[m] += dca_fold_data[fidx][m]
    if n_folds_done > 0:
        for m in nb_accum:
            nb_accum[m] /= n_folds_done

    # Build DCA summary table (Threshold × Model)
    dca_rows = []
    for i, pt_val in enumerate(thresholds):
        row = {"Threshold": pt_val}
        for m in all_models:
            row[m] = nb_accum[m][i]
        dca_rows.append(row)
    dca_table = pd.DataFrame(dca_rows).set_index("Threshold")

    print(f"\n{'='*80}")
    print(f" Decision Curve Analysis — Mean Net Benefit ({n_folds_done} folds, eICU)")
    print(f"{'='*80}")
    key_pts = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    print(f"{'Threshold':<12}", end="")
    for m in dca_model_names + ["Treat All"]:
        print(f"  {m:<16}", end="")
    print()
    for kp in key_pts:
        idx = np.argmin(np.abs(thresholds - kp))
        print(f"{thresholds[idx]:<12.2f}", end="")
        for m in dca_model_names + ["Treat All"]:
            print(f"  {nb_accum[m][idx]:<16.4f}", end="")
        print()

    dca_table.to_csv(os.path.join(cfg.output_dir, "dca_eICU.csv"))
    print(f"\n  Saved: dca_eICU.csv")

    # ====================================================================
    # 6. Save Checkpoint for Part 3 Aggregation
    # ====================================================================
    #   Bundles experiment results, frozen params, and DCA so Part 3
    #   can produce cross-dataset comparison tables without retraining.
    # ====================================================================
    ckpt_data = {
        "experiment_results":  {"eICU": experiment_results},
        "frozen_params":       {"eICU": final_params},
        "mimic_frozen_params": frozen_params,
        "dataset_name":        "eICU",
        "dca":                 {"table": dca_table, "raw": dca_fold_data},
    }
    save_checkpoint(
        os.path.join(cfg.output_dir, "checkpoint_eicu.pkl"),
        **ckpt_data,
    )
    # Clean up per-fold progress file (full experiment is complete)
    if os.path.exists(exp_progress_file):
        os.remove(exp_progress_file)
        print(f"  Cleaned up: {exp_progress_file}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Part 2 COMPLETE — Total time: {elapsed / 60:.1f} min")
    print(f"  Results saved to: {cfg.output_dir}")
    print(f"  Checkpoint: checkpoint_eicu.pkl")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
