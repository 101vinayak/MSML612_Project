import time
import torch
from transformers import AutoTokenizer

from src.models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_CONFIGS = {
    "Baseline": {
        "model_type": "baseline",
        "checkpoint": "checkpoints/baseline_best.pt",
        "factorized_layers": [4, 5],
        "rank": 64,
        "num_experts": 4,
        "top_k": 1,
    },
    "Low-Rank": {
        "model_type": "lowrank",
        "checkpoint": "checkpoints/lowrank_r32_layers4-5_best.pt",
        "factorized_layers": [4, 5],
        "rank": 32,
        "num_experts": 4,
        "top_k": 1,
    },
    "MoE": {
        "model_type": "moe",
        "checkpoint": "checkpoints/moe_r16_layers4-5_e4_k1_best.pt",
        "factorized_layers": [4, 5],
        "rank": 16,
        "num_experts": 4,
        "top_k": 1,
    },
    "MoE + KD": {
        "model_type": "moe",
        "checkpoint": "checkpoints/moe_r16_layers4-5_e4_k1_kd_best.pt",
        "factorized_layers": [4, 5],
        "rank": 16,
        "num_experts": 4,
        "top_k": 1,
    },
}

TEXTS = [
    "This film is an absolute masterpiece.",
    "The movie was boring, predictable, and painfully slow.",
    "The acting was great, but the story felt unfinished.",
    "I would happily recommend this to my friends.",
    "This was one of the worst films I have seen all year.",
    "A charming and emotional story with beautiful performances.",
    "The plot made no sense and the ending was terrible.",
    "It was not perfect, but I still enjoyed most of it.",
    "A dull experience with very little to remember.",
    "Brilliant direction, sharp writing, and a satisfying ending.",
]


def load_checkpoint_model(name, cfg):
    print(f"Loading {name}...")
    model = build_model(
        model_type=cfg["model_type"],
        factorized_layers=cfg["factorized_layers"],
        rank=cfg["rank"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
    ).to(DEVICE)

    ckpt = torch.load(cfg["checkpoint"], map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


def predict_batch(model, tokenizer, texts):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(**inputs)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    probs = torch.softmax(outputs.logits, dim=-1)
    preds = probs.argmax(dim=-1).detach().cpu().tolist()
    confs = probs.max(dim=-1).values.detach().cpu().tolist()

    elapsed_ms = (end - start) * 1000
    ms_per_sample = elapsed_ms / len(texts)

    return preds, confs, elapsed_ms, ms_per_sample


def label_name(pred):
    return "POSITIVE" if pred == 1 else "NEGATIVE"

import csv
import os

def print_saved_benchmark():
    path = "results/final_benchmark_summary.csv"
    if not os.path.exists(path):
        print("\nSaved benchmark file not found.")
        return

    print("\n" + "=" * 90)
    print("REPRODUCED BENCHMARK SUMMARY")
    print("=" * 90)

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"{'Model':<12} {'Acc (%)':>10} {'Params (M)':>12} {'ms/sample':>12} {'Throughput/s':>14}")
    print("-" * 68)
    for r in rows:
        print(
            f"{r['Model']:<12} "
            f"{float(r['Accuracy (%)']):>10.2f} "
            f"{float(r['Params (M)']):>12.2f} "
            f"{float(r['Latency (ms/sample)']):>12.3f} "
            f"{float(r['Throughput (samples/sec)']):>14.2f}"
        )

def main():
    print("=" * 90)
    print("MSML612 Live Demo: Baseline vs Low-Rank vs MoE vs MoE + KD")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {len(TEXTS)}")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    models = {
        name: load_checkpoint_model(name, cfg)
        for name, cfg in MODEL_CONFIGS.items()
    }

    print("\nINPUT BATCH")
    for i, text in enumerate(TEXTS, 1):
        print(f"{i:02d}. {text}")

    print("\n" + "=" * 90)
    print("LIVE MODEL COMPARISON")
    print("=" * 90)

    summary = []

    all_outputs = {}
    for name, model in models.items():
        preds, confs, batch_ms, ms_per_sample = predict_batch(model, tokenizer, TEXTS)
        all_outputs[name] = preds

        summary.append({
            "model": name,
            "batch_ms": batch_ms,
            "ms_per_sample": ms_per_sample,
            "preds": preds,
            "confs": confs,
        })

        print(f"\n{name}")
        print("-" * 60)
        print(f"Batch latency: {batch_ms:.3f} ms")
        print(f"Latency/sample: {ms_per_sample:.3f} ms")
        for i, (text, pred, conf) in enumerate(zip(TEXTS, preds, confs), 1):
            print(f"{i:02d}. {label_name(pred):8s} | conf={conf:.3f} | {text}")

    print_saved_benchmark()

    print("\nTakeaway:")
    print("- Low-Rank should be fastest because it replaces dense W1 with smaller dense factors.")
    print("- MoE / MoE+KD may be accurate, but routing overhead makes them slower in this implementation.")
    print("- Same input batch is used for all models, so the comparison is fair for live demo.")


if __name__ == "__main__":
    main()