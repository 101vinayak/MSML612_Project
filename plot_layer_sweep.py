import json
import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

layer_files = [
    "lowrank_r32_layers5_metrics.json",
    "lowrank_r32_layers4-5_metrics.json",
    "lowrank_r32_layers3-4-5_metrics.json",
    "lowrank_r32_layers2-3-4-5_metrics.json",
    "lowrank_r32_layers1-2-3-4-5_metrics.json",
    "lowrank_r32_layers0-1-2-3-4-5_metrics.json",
]

rows = []

for fname in layer_files:
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "r") as f:
        data = json.load(f)

    rows.append({
        "layers": str(data["factorized_layers"]),
        "num_layers": len(data["factorized_layers"]),
        "params_m": data["trainable_params"] / 1e6,
        "accuracy_pct": data["best_val_accuracy"] * 100,
    })

df = pd.DataFrame(rows).sort_values("num_layers")

df.to_csv(os.path.join(RESULTS_DIR, "layer_sweep_summary.csv"), index=False)

print("\nLayer Sweep Summary:")
print(df.round(3).to_string(index=False))

# Accuracy vs number of factorized layers
plt.figure(figsize=(7, 5))
plt.plot(df["num_layers"], df["accuracy_pct"], marker="o")
for _, row in df.iterrows():
    plt.text(row["num_layers"] + 0.03, row["accuracy_pct"] + 0.05, row["layers"], fontsize=8)

plt.xlabel("Number of Factorized Layers")
plt.ylabel("Validation Accuracy (%)")
plt.title("Layer-wise Low-Rank Compression Sensitivity")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "layer_sensitivity_accuracy.png"), dpi=300)
plt.close()

# Params vs number of factorized layers
plt.figure(figsize=(7, 5))
plt.plot(df["num_layers"], df["params_m"], marker="o")
for _, row in df.iterrows():
    plt.text(row["num_layers"] + 0.03, row["params_m"] + 0.05, row["layers"], fontsize=8)

plt.xlabel("Number of Factorized Layers")
plt.ylabel("Trainable Parameters (M)")
plt.title("Parameter Reduction with Increasing Compression Depth")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "layer_sensitivity_params.png"), dpi=300)
plt.close()

# Accuracy vs params for layer sweep
plt.figure(figsize=(7, 5))
plt.scatter(df["params_m"], df["accuracy_pct"])
for _, row in df.iterrows():
    plt.text(row["params_m"] + 0.05, row["accuracy_pct"] + 0.05, row["layers"], fontsize=8)

plt.xlabel("Trainable Parameters (M)")
plt.ylabel("Validation Accuracy (%)")
plt.title("Layer Sweep: Accuracy vs Parameter Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "layer_sweep_accuracy_vs_params.png"), dpi=300)
plt.close()

print("\nSaved:")
print(f"- {RESULTS_DIR}/layer_sweep_summary.csv")
print(f"- {FIGURES_DIR}/layer_sensitivity_accuracy.png")
print(f"- {FIGURES_DIR}/layer_sensitivity_params.png")
print(f"- {FIGURES_DIR}/layer_sweep_accuracy_vs_params.png")