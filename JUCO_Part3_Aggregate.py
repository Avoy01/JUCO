"""
JUCO Framework v2.0 — Part 3: Aggregation & Visualization
==========================================================
Pure post-processing script — **no model training is performed**.

Reads pre-computed CSV results from Parts 1, 1B, and 2, then produces
all tables and figures required for the MLHC submission:

  1. **Aggregate summary** — cross-dataset metric table
     (models × {AUROC, AUPRC, Brier, ECE, F1, Sens, Spec, AURC, Risk@90}).
  2. **Statistical significance** — per-fold Wilcoxon signed-rank tests
     (Bonferroni-corrected) and Friedman omnibus test, loaded from
     ``checkpoint_*.pkl`` per-fold data.
  3. **Metric comparison bar charts** — one subplot per metric, grouped
     by dataset, with gold borders on best-performing models.
  4. **Ablation study chart** — grouped bars for Full vs A1–A5 variants.
  5. **Robustness plot** — AUROC vs MNAR missingness rate (10 %–60 %).
  6. **DCA plot** — net benefit vs threshold probability (Vickers &
     Elkin, Medical Decision Making, 2006).

How to run
----------
Execute after Parts 1, 1B, and 2 have completed::

    python JUCO_Part3_Aggregate.py

Inputs (all in ``./results/``)
------------------------------
Required:
    ``results_MIMIC_IV.csv``, ``results_eICU.csv``

Optional (figures are skipped gracefully if absent):
    ``results_JUCO_XGB_MIMIC_IV.csv``,
    ``ablation_MIMIC_IV.csv``,
    ``missingness_stress_MIMIC_IV.csv`` or ``stress_results_incremental.csv``,
    ``dca_MIMIC_IV.csv``, ``dca_eICU.csv``,
    ``checkpoint_mimic.pkl``, ``checkpoint_eicu.pkl``
    (per-fold data for statistical tests)

Outputs
-------
``aggregate_results.csv``
    Combined metric table across datasets.
``statistical_tests.csv``
    Wilcoxon p-values, delta AUROC, 95 % CIs.
``metric_comparison.png / .pdf``
    Bar charts comparing all models on key metrics.
``ablation_study.png / .pdf``
    Ablation variant comparison (if data exists).
``fig_missingness.png / .pdf``
    Robustness degradation plot (if data exists).
``fig_dca.png / .pdf``
    Decision curve analysis (if data exists).
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                          # headless backend (no GUI)
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare

# Optional: per-fold data from juco_core checkpoints
try:
    from juco_core import load_checkpoint, METRIC_NAMES, BASELINES
except ImportError:
    load_checkpoint = None
    METRIC_NAMES = [
        "Accuracy", "F1_macro", "AUROC", "ECE",
        "Brier", "AURC", "Risk@90",
    ]
    BASELINES = {
        "XGBoost": None, "LightGBM": None,
        "MissForest+XGB": None, "MeanImp+XGB": None,
    }

OUTPUT_DIR = "./results/"


# ============================================================================
# Helper: Parse "mean +/- std" Strings
# ============================================================================

def parse_mean_std(val):
    """Parse ``'0.7908 +/- 0.0043'`` into ``(0.7908, 0.0043)``.

    Handles plain numeric values (returns std = 0) and non-numeric
    strings (returns NaN, NaN).
    """
    if isinstance(val, (int, float)):
        return float(val), 0.0
    s = str(val).strip()
    if "+/-" in s:
        parts = s.split("+/-")
        return float(parts[0].strip()), float(parts[1].strip())
    try:
        return float(s), 0.0
    except ValueError:
        return np.nan, np.nan


# ============================================================================
# CSV Loader
# ============================================================================

def load_result_csv(path):
    """Load a results CSV, returning a DataFrame or ``None`` if absent."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"  Loaded {path} ({len(df)} rows)")
        return df
    else:
        print(f"  Not found: {path}")
        return None


# ============================================================================
# Statistical Significance Tests
# ============================================================================

def statistical_tests_from_checkpoint(experiment_results, output_dir):
    """Run paired Wilcoxon signed-rank tests (JUCO vs each baseline).

    For each dataset, computes:
      - Delta AUROC (JUCO − baseline) averaged across folds.
      - Wilcoxon p-value (Bonferroni-corrected at α = 0.05 / n_baselines).
      - Percentile-based 95 % CI of per-fold AUROC differences.
      - Friedman omnibus test across all models.

    Results are saved to ``statistical_tests.csv``.
    """
    all_rows = []
    baseline_names = [k for k in list(BASELINES.keys()) if k != "JUCO"]

    for ds_name, results in experiment_results.items():
        if "JUCO" not in results:
            continue
        juco_aurocs = np.array(results["JUCO"].get("AUROC", []))
        if len(juco_aurocs) == 0:
            continue

        n_bl = len(baseline_names)
        bonf_alpha = 0.05 / max(1, n_bl)

        print(f"\n{'='*60}")
        print(f" Statistical Tests: {ds_name}")
        print(f"{'='*60}")

        for bl_name in baseline_names:
            if bl_name not in results:
                continue
            bl_aurocs = np.array(results[bl_name].get("AUROC", []))
            if len(bl_aurocs) != len(juco_aurocs):
                continue

            delta = np.mean(juco_aurocs) - np.mean(bl_aurocs)

            if len(juco_aurocs) >= 5:
                try:
                    stat, p_wil = wilcoxon(juco_aurocs, bl_aurocs)
                except ValueError:
                    p_wil = np.nan
            else:
                p_wil = np.nan

            diffs = juco_aurocs - bl_aurocs
            if len(diffs) >= 5:
                lo, hi = np.percentile(diffs, [2.5, 97.5])
            else:
                lo, hi = delta, delta

            sig = "Yes" if (not np.isnan(p_wil) and p_wil < bonf_alpha) else "No"
            row = {
                "Dataset": ds_name, "Comparison": f"JUCO vs {bl_name}",
                "Delta_AUROC": f"{delta:+.4f}",
                "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                "Wilcoxon_p": f"{p_wil:.4f}" if not np.isnan(p_wil) else "N/A",
                "Significant": sig,
            }
            all_rows.append(row)
            print(f"  JUCO vs {bl_name:<15}: Delta={delta:+.4f}, p={f'{p_wil:.4f}' if not np.isnan(p_wil) else 'N/A'}, {sig}")

        # Friedman test across all models
        all_model_names = ["JUCO"] + baseline_names
        all_aurocs = []
        for m in all_model_names:
            if m in results and "AUROC" in results[m]:
                all_aurocs.append(np.array(results[m]["AUROC"]))
        if len(all_aurocs) >= 3 and all(len(a) == len(all_aurocs[0]) for a in all_aurocs) and len(all_aurocs[0]) >= 3:
            try:
                stat_f, p_f = friedmanchisquare(*all_aurocs)
                print(f"  Friedman: chi2={stat_f:.2f}, p={p_f:.4f}")
            except Exception:
                print("  Friedman: could not compute")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(output_dir, "statistical_tests.csv"), index=False)
        print("\n" + df.to_string(index=False))
        return df
    return None


# ============================================================================
# Figure 1: Metric Comparison Bar Charts
# ============================================================================

def plot_metric_comparison(datasets_results, output_dir):
    """Grouped bar chart: models × metrics, one row per dataset.

    Gold border highlights the best model per metric (highest for
    AUROC, lowest for ECE / Brier / AURC).
    """
    metrics_to_plot = ["AUROC", "ECE", "Brier", "AURC"]
    n_ds = len(datasets_results)
    n_m = len(metrics_to_plot)

    fig, axes = plt.subplots(n_ds, n_m, figsize=(5 * n_m, 5 * n_ds), squeeze=False)

    colors_map = {"JUCO": "#2196F3", "XGBoost": "#FF5722", "LightGBM": "#4CAF50",
                  "MissForest+XGB": "#9C27B0", "MeanImp+XGB": "#FF9800"}

    for d_i, (ds_name, df_res) in enumerate(datasets_results.items()):
        if df_res is None or "Model" not in df_res.columns:
            continue
        models = df_res["Model"].tolist()
        for m_i, metric in enumerate(metrics_to_plot):
            ax = axes[d_i, m_i]
            if metric not in df_res.columns:
                ax.set_visible(False)
                continue
            means, stds = [], []
            for _, row in df_res.iterrows():
                m, s = parse_mean_std(row[metric])
                means.append(m)
                stds.append(s)
            cols = [colors_map.get(m, "#607D8B") for m in models]
            bars = ax.bar(range(len(models)), means, yerr=stds, color=cols, capsize=4, alpha=0.85)
            valid_means = [m for m in means if not np.isnan(m)]
            if valid_means:
                best = np.argmin(means) if metric in ["ECE", "Brier", "AURC"] else np.argmax(means)
                bars[best].set_edgecolor("gold")
                bars[best].set_linewidth(3)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
            ax.set_title(f"{metric} -- {ds_name}", fontsize=12, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metric_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "metric_comparison.png"), dpi=300)
    plt.close()
    print("  Saved metric_comparison.pdf/png")


# ============================================================================
# Figure 2: Ablation Study
# ============================================================================

def plot_ablation(df_abl, output_dir):
    """Grouped bar chart of ablation variants (Full, A1–A5) on key metrics."""
    if df_abl is None or len(df_abl) == 0:
        print("  No ablation data — skipping.")
        return

    # Determine the variant column (first column that isn't a metric)
    metric_cols = [c for c in ["AUROC", "ECE", "Brier", "AURC", "Accuracy", "F1_macro", "Risk@90"]
                   if c in df_abl.columns]
    variant_col = [c for c in df_abl.columns if c not in metric_cols and "_mean" not in c][0]

    fig, ax = plt.subplots(figsize=(10, 6))
    abl_colors = ["#2196F3", "#FF5722", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    plot_metrics = [m for m in ["AUROC", "ECE", "Brier", "AURC"] if m in df_abl.columns]
    n_v = len(df_abl)
    n_m = len(plot_metrics)
    x = np.arange(n_m)
    w = 0.8 / max(1, n_v)

    variants = df_abl[variant_col].tolist()
    for v_i, (_, row) in enumerate(df_abl.iterrows()):
        means = []
        for m in plot_metrics:
            mv, _ = parse_mean_std(row[m])
            means.append(mv)
        ax.bar(x + v_i * w, means, w, label=variants[v_i],
               color=abl_colors[v_i % len(abl_colors)], alpha=0.85)

    ax.set_xticks(x + w * (n_v - 1) / 2)
    ax.set_xticklabels(plot_metrics, fontsize=11)
    ax.set_title("Ablation Study -- MIMIC-IV", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_study.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "ablation_study.png"), dpi=300)
    plt.close()
    print("  Saved ablation_study.pdf/png")


# ============================================================================
# Figure 3: Robustness — Missingness Stress Test
# ============================================================================

def plot_missingness_stress(df_stress, output_dir):
    """Line plot: AUROC vs MNAR missingness rate for each model.

    JUCO is drawn with a solid line; baselines use dashed lines.
    Gracefully detects varying column naming conventions.
    """
    if df_stress is None or len(df_stress) == 0:
        print("  No missingness stress data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {"JUCO": "#2196F3", "XGBoost": "#FF5722", "LightGBM": "#4CAF50",
                  "MissForest+XGB": "#9C27B0", "MeanImp+XGB": "#FF9800"}

    # Detect column names (could be "Model", "model", "rate", "mnar_rate", "AUROC", etc.)
    rate_col = None
    for c in ["rate", "mnar_rate", "missing_rate", "Rate", "MNAR_Rate"]:
        if c in df_stress.columns:
            rate_col = c
            break
    model_col = None
    for c in ["Model", "model", "Method", "method"]:
        if c in df_stress.columns:
            model_col = c
            break
    auroc_col = None
    for c in ["AUROC", "auroc", "mean_auroc", "AUROC_mean"]:
        if c in df_stress.columns:
            auroc_col = c
            break

    if rate_col is None or model_col is None or auroc_col is None:
        print(f"  Cannot identify columns for stress plot. Columns: {list(df_stress.columns)}")
        return

    models = df_stress[model_col].unique()
    for model in models:
        subset = df_stress[df_stress[model_col] == model].sort_values(rate_col)
        rates = subset[rate_col].values
        aurocs = []
        for v in subset[auroc_col].values:
            mv, _ = parse_mean_std(v)
            aurocs.append(mv)
        style = "-o" if model == "JUCO" else "--s"
        lw = 2.5 if model == "JUCO" else 1.5
        ax.plot(rates, aurocs, style, label=model,
                color=colors_map.get(model, "#607D8B"), linewidth=lw, markersize=6)

    ax.set_xlabel("MNAR Missingness Rate", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Robustness: AUROC vs Missingness Rate (MIMIC-IV)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_missingness.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig_missingness.png"), dpi=300)
    plt.close()
    print("  Saved fig_missingness.pdf/png")


# ============================================================================
# Figure 4: Decision Curve Analysis (Net Benefit)
# ============================================================================

def plot_dca(dca_dfs, output_dir):
    """Net benefit vs threshold probability, one panel per dataset.

    Reference strategies *Treat All* and *Treat None* are included.
    A model is clinically useful at threshold p_t if its net benefit
    exceeds both reference curves.
    """
    valid = {k: v for k, v in dca_dfs.items() if v is not None and len(v) > 0}
    if not valid:
        print("  No DCA data — skipping.")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 6), squeeze=False)
    colors_map = {"JUCO": "#2196F3", "XGBoost": "#FF5722", "LightGBM": "#4CAF50",
                  "MissForest+XGB": "#9C27B0", "MeanImp+XGB": "#FF9800",
                  "Treat All": "#888888", "Treat None": "#BBBBBB"}

    for i, (ds_name, df_dca) in enumerate(valid.items()):
        ax = axes[0, i]

        # Detect columns
        thresh_col = None
        for c in ["threshold", "Threshold", "threshold_prob", "pt"]:
            if c in df_dca.columns:
                thresh_col = c
                break
        model_col = None
        for c in ["Model", "model", "Method", "method"]:
            if c in df_dca.columns:
                model_col = c
                break
        nb_col = None
        for c in ["net_benefit", "Net_Benefit", "NB", "net_benefit_mean"]:
            if c in df_dca.columns:
                nb_col = c
                break

        if thresh_col is None or model_col is None or nb_col is None:
            ax.text(0.5, 0.5, f"Cannot parse DCA columns\n{list(df_dca.columns)}",
                    transform=ax.transAxes, ha="center")
            continue

        for model in df_dca[model_col].unique():
            subset = df_dca[df_dca[model_col] == model].sort_values(thresh_col)
            style = "-" if model in ("JUCO", "Treat All", "Treat None") else "--"
            lw = 2.5 if model == "JUCO" else 1.5
            ax.plot(subset[thresh_col], subset[nb_col], style, label=model,
                    color=colors_map.get(model, "#607D8B"), linewidth=lw)

        ax.set_xlabel("Threshold Probability", fontsize=12)
        ax.set_ylabel("Net Benefit", fontsize=12)
        ax.set_title(f"Decision Curve Analysis -- {ds_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-0.05)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_dca.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "fig_dca.png"), dpi=300)
    plt.close()
    print("  Saved fig_dca.pdf/png")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Aggregate results and generate all MLHC submission figures."""
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  JUCO Part 3: Aggregation & Visualization (Post-Processing)")
    print("  No training — reads existing CSV results only")
    print("=" * 70)

    # ====================================================================
    # 1. Load Result CSVs
    # ====================================================================
    print("\n[1/5] Loading result CSVs...")

    res_mimic = load_result_csv(os.path.join(OUTPUT_DIR, "results_MIMIC_IV.csv"))
    res_eicu = load_result_csv(os.path.join(OUTPUT_DIR, "results_eICU.csv"))
    res_xgb = load_result_csv(os.path.join(OUTPUT_DIR, "results_JUCO_XGB_MIMIC_IV.csv"))
    abl_mimic = load_result_csv(os.path.join(OUTPUT_DIR, "ablation_MIMIC_IV.csv"))
    stress = load_result_csv(os.path.join(OUTPUT_DIR, "missingness_stress_MIMIC_IV.csv"))
    if stress is None:
        stress = load_result_csv(os.path.join(OUTPUT_DIR, "stress_results_incremental.csv"))
    dca_mimic = load_result_csv(os.path.join(OUTPUT_DIR, "dca_MIMIC_IV.csv"))
    dca_eicu = load_result_csv(os.path.join(OUTPUT_DIR, "dca_eICU.csv"))

    # ====================================================================
    # 2. Display & Save Aggregate Summary
    # ====================================================================
    print("\n[2/5] Aggregated Results Summary")
    print("=" * 100)

    all_summaries = []
    datasets_results = {}

    for ds_label, df_res in [("MIMIC-IV", res_mimic), ("eICU", res_eicu)]:
        if df_res is None:
            continue
        print(f"\n  --- {ds_label} ---")
        print(df_res.to_string(index=False))
        df_res = df_res.copy()
        df_res["Dataset"] = ds_label
        all_summaries.append(df_res)
        datasets_results[ds_label] = df_res

    if res_xgb is not None:
        print(f"\n  --- JUCO-XGB Variant (MIMIC-IV) ---")
        print(res_xgb.to_string(index=False))

    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "aggregate_results.csv"), index=False)
        print(f"\n  Saved aggregate_results.csv")

    # ====================================================================
    # 3. Statistical Significance Tests (per-fold from checkpoints)
    # ====================================================================
    print("\n[3/5] Statistical Significance Tests...")
    experiment_results = {}

    if load_checkpoint is not None:
        for ckpt_file, ds_name in [("checkpoint_mimic.pkl", "MIMIC-IV"),
                                    ("checkpoint_eicu.pkl", "eICU")]:
            ckpt_path = os.path.join(OUTPUT_DIR, ckpt_file)
            if os.path.exists(ckpt_path):
                try:
                    ckpt = load_checkpoint(ckpt_path)
                    if "experiment_results" in ckpt:
                        experiment_results.update(ckpt["experiment_results"])
                        print(f"  Loaded per-fold data from {ckpt_file}")
                except Exception as e:
                    print(f"  Could not load {ckpt_file}: {e}")

    if experiment_results:
        statistical_tests_from_checkpoint(experiment_results, OUTPUT_DIR)
    else:
        print("  No checkpoint files available for per-fold statistical tests.")
        print("  (Statistical tests require per-fold AUROC values from checkpoint_*.pkl)")

    # ====================================================================
    # 4. Generate All Figures
    # ====================================================================
    print("\n[4/5] Generating Figures...")

    # Metric comparison bar charts
    if datasets_results:
        print("\n  Metric comparison bar charts...")
        plot_metric_comparison(datasets_results, OUTPUT_DIR)

    # Ablation study
    if abl_mimic is not None:
        print("\n  Ablation study...")
        plot_ablation(abl_mimic, OUTPUT_DIR)

    # Robustness / missingness stress
    if stress is not None:
        print("\n  Robustness (missingness stress)...")
        plot_missingness_stress(stress, OUTPUT_DIR)

    # DCA
    dca_dfs = {}
    if dca_mimic is not None:
        dca_dfs["MIMIC-IV"] = dca_mimic
    if dca_eicu is not None:
        dca_dfs["eICU"] = dca_eicu
    if dca_dfs:
        print("\n  Decision Curve Analysis...")
        plot_dca(dca_dfs, OUTPUT_DIR)

    # ====================================================================
    # 5. Summary
    # ====================================================================
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Part 3 COMPLETE — Total time: {elapsed:.1f} sec")
    print(f"  All results and figures saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    # List generated files
    print("\n  Generated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        if size > 0:
            print(f"    {f:45s} ({size:>10,} bytes)")


if __name__ == "__main__":
    main()
