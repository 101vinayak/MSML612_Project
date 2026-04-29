import re
import pandas as pd
import matplotlib.pyplot as plt

log_path = "logs/moe_train_3epoch_clean.log"

rows = []
global_step = 0

# tqdm lines usually contain:
# Epoch 1/3:  12% ... loss=0.123, ce=0.120, kd=0
pattern_epoch = re.compile(r"Epoch\s+(\d+)/(\d+)")
pattern_loss = re.compile(r"loss=([0-9]*\.?[0-9]+)")
pattern_ce = re.compile(r"ce=([0-9]*\.?[0-9]+)")
pattern_kd = re.compile(r"kd=([0-9]*\.?[0-9]+)")

with open(log_path, "r", errors="ignore") as f:
    for line in f:
        if "loss=" not in line:
            continue

        epoch_match = pattern_epoch.search(line)
        loss_match = pattern_loss.search(line)
        ce_match = pattern_ce.search(line)
        kd_match = pattern_kd.search(line)

        if not loss_match:
            continue

        global_step += 1

        rows.append({
            "step": global_step,
            "epoch": int(epoch_match.group(1)) if epoch_match else None,
            "loss": float(loss_match.group(1)),
            "ce": float(ce_match.group(1)) if ce_match else None,
            "kd": float(kd_match.group(1)) if kd_match else None,
        })

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("No losses parsed. Check the log file format.")

# tqdm logs can be dense/noisy; smooth for presentation
window = max(20, len(df) // 100)
df["loss_smooth"] = df["loss"].rolling(window=window, min_periods=1).mean()

df.to_csv("logs/moe_train_3epoch_losses.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(df["step"], df["loss"], alpha=0.25, label="Raw training loss")
plt.plot(df["step"], df["loss_smooth"], linewidth=2, label=f"Smoothed loss (window={window})")

plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("MoE Training Loss over 3 Epochs")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/moe_training_loss_curve.png", dpi=300)
plt.show()

print(df.head())
print(df.tail())
print(f"Parsed {len(df)} loss points.")
print("Saved: logs/moe_train_3epoch_losses.csv")
print("Saved: figures/moe_training_loss_curve.png")