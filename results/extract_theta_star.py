import pickle, os, sys

CHECKPOINT = "./results/checkpoint_mimic.pkl"

if not os.path.exists(CHECKPOINT):
    sys.exit(f"Checkpoint not found at {CHECKPOINT}. "
             f"Run JUCO_Part1_MIMIC.py first.")

with open(CHECKPOINT, "rb") as f:
    ckpt = pickle.load(f)

# frozen_params is stored as {"MIMIC-IV": {...}}
fp = ckpt.get("frozen_params", {})
params = fp.get("MIMIC-IV", fp)   # handle both nesting styles

print("\n╔══════════════════════════════════════════════════════════╗")
print("║  DE-Optimised Parameters θ*  (paste into Table A.2)     ║")
print("╠══════════════════════════════════════════════════════════╣")
keys = ["alpha_L", "alpha_U", "eta", "max_depth", "min_leaf_samples"]
labels = {
    "alpha_L":          "Lower quantile level     α_L",
    "alpha_U":          "Upper quantile level     α_U",
    "eta":              "Fuzzy proportion factor  η  ",
    "max_depth":        "Max tree depth           d_max",
    "min_leaf_samples": "Min leaf samples         n_min",
}
for k in keys:
    v = params.get(k, "NOT FOUND")
    if isinstance(v, float):
        print(f"║  {labels[k]} : {v:.4f}              ║")
    else:
        print(f"║  {labels[k]} : {v}              ║")
print("╚══════════════════════════════════════════════════════════╝\n")

print("→ In juco_main.tex, find Table A.2 (label: tab:de_params)")
print("  Replace each [PLACEHOLDER] with the value above.\n")