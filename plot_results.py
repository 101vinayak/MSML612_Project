import json
import os
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)

benchmark_path = os.path.join(RESULTS_DIR, "benchmark_results.json")

with open(benchmark_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Clean display names
name_map = {
    "baseline": "Baseline",
    "lowrank_r32_layers4-5": "Low-rank",
    "moe_r16_layers4-5_e4_k1": "MoE",
    "moe_r16_layers4-5_e4_k1_kd": "MoE + KD",
}

df["display_name"] = df["name"].map(name_map).fillna(df["name"])
df["accuracy_pct"] = df["accuracy"] * 100
df["params_reduction_pct"] = (1 - df["params_m"] / df.loc[df["name"] == "baseline", "params_m"].iloc[0]) * 100

# Save clean CSV
summary_cols = [
    "display_name",
    "params_m",
    "params_reduction_pct",
    "accuracy_pct",
    "avg_ms_per_sample",
    "samples_per_second",
]
summary = df[summary_cols].copy()
summary.columns = [
    "Model",
    "Params (M)",
    "Param Reduction (%)",
    "Accuracy (%)",
    "Latency (ms/sample)",
    "Throughput (samples/sec)",
]
summary.to_csv(os.path.join(RESULTS_DIR, "final_benchmark_summary.csv"), index=False)

print("\nFinal Benchmark Summary:")
print(summary.round(3).to_string(index=False))


# -----------------------------
# Figure 1: Accuracy vs Params
# -----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(df["params_m"], df["accuracy_pct"])

for _, row in df.iterrows():
    plt.text(
        row["params_m"] + 0.03,
        row["accuracy_pct"] + 0.02,
        row["display_name"],
        fontsize=9,
    )

plt.xlabel("Trainable Parameters (M)")
plt.ylabel("Validation Accuracy (%)")
plt.title("Accuracy vs Parameter Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "accuracy_vs_params.png"), dpi=300)
plt.close()


# -----------------------------
# Figure 2: Latency vs Accuracy
# -----------------------------
plt.figure(figsize=(7, 5))
plt.scatter(df["avg_ms_per_sample"], df["accuracy_pct"])

for _, row in df.iterrows():
    plt.text(
        row["avg_ms_per_sample"] + 0.005,
        row["accuracy_pct"] + 0.02,
        row["display_name"],
        fontsize=9,
    )

plt.xlabel("Latency (ms/sample)")
plt.ylabel("Validation Accuracy (%)")
plt.title("Accuracy vs Inference Latency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "accuracy_vs_latency.png"), dpi=300)
plt.close()


# -----------------------------
# Figure 3: Throughput bar chart
# -----------------------------
plt.figure(figsize=(7, 5))
plt.bar(df["display_name"], df["samples_per_second"])
plt.ylabel("Throughput (samples/sec)")
plt.title("Inference Throughput Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "throughput_comparison.png"), dpi=300)
plt.close()


# -----------------------------
# Figure 4: Parameter count bar chart
# -----------------------------
plt.figure(figsize=(7, 5))
plt.bar(df["display_name"], df["params_m"])
plt.ylabel("Trainable Parameters (M)")
plt.title("Model Size Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "params_comparison.png"), dpi=300)
plt.close()


# -----------------------------
# Figure 5: Latency bar chart
# -----------------------------
plt.figure(figsize=(7, 5))
plt.bar(df["display_name"], df["avg_ms_per_sample"])
plt.ylabel("Latency (ms/sample)")
plt.title("Inference Latency Comparison")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "latency_comparison.png"), dpi=300)
plt.close()

print("\nSaved:")
print(f"- {RESULTS_DIR}/final_benchmark_summary.csv")
print(f"- {FIGURES_DIR}/accuracy_vs_params.png")
print(f"- {FIGURES_DIR}/accuracy_vs_latency.png")
print(f"- {FIGURES_DIR}/throughput_comparison.png")
print(f"- {FIGURES_DIR}/params_comparison.png")
print(f"- {FIGURES_DIR}/latency_comparison.png")