import argparse
import json
import os
import time
from typing import Dict, List

import torch

from src.data import build_dataloaders, get_tokenizer
from src.models import build_model
from src.utils import count_trainable_parameters, ensure_dir, format_param_count, save_json, set_seed


def evaluate_accuracy(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            total_loss += outputs.loss.item() * batch["labels"].size(0)

    return {
        "accuracy": correct / total,
        "loss": total_loss / total,
    }


def benchmark_latency(model, dataloader, device, warmup_batches: int = 5, measured_batches: int = 30) -> Dict[str, float]:
    model.eval()

    batches = []
    for batch in dataloader:
        batches.append({k: v.to(device) for k, v in batch.items()})
        if len(batches) >= warmup_batches + measured_batches:
            break

    if len(batches) <= warmup_batches:
        raise ValueError("Not enough batches available for benchmarking.")

    with torch.no_grad():
        for batch in batches[:warmup_batches]:
            _ = model(**batch)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        total_samples = 0

        for batch in batches[warmup_batches:warmup_batches + measured_batches]:
            _ = model(**batch)
            total_samples += batch["input_ids"].size(0)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    num_batches = min(measured_batches, len(batches) - warmup_batches)

    return {
        "avg_ms_per_batch": 1000.0 * elapsed / num_batches,
        "avg_ms_per_sample": 1000.0 * elapsed / total_samples,
        "samples_per_second": total_samples / elapsed,
        "measured_batches": num_batches,
        "measured_samples": total_samples,
    }


def load_checkpoint_if_available(model, checkpoint_path: str, device):
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return True
    return False


def main(args):
    set_seed(args.seed)
    ensure_dir(args.results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    _, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
    )

    configs: List[Dict] = [
        {
            "name": "baseline",
            "model_type": "baseline",
            "factorized_layers": [4, 5],
            "rank": 64,
            "num_experts": 4,
            "top_k": 1,
            "checkpoint": os.path.join(args.checkpoint_dir, "baseline_best.pt"),
        },
        {
            "name": "lowrank_r32_layers4-5",
            "model_type": "lowrank",
            "factorized_layers": [4, 5],
            "rank": 32,
            "num_experts": 4,
            "top_k": 1,
            "checkpoint": os.path.join(args.checkpoint_dir, "lowrank_r32_layers4-5_best.pt"),
        },
        {
            "name": "moe_r16_layers4-5_e4_k1",
            "model_type": "moe",
            "factorized_layers": [4, 5],
            "rank": 16,
            "num_experts": 4,
            "top_k": 1,
            "checkpoint": os.path.join(args.checkpoint_dir, "moe_r16_layers4-5_e4_k1_best.pt"),
        },
        {
            "name": "moe_r16_layers4-5_e4_k1_kd",
            "model_type": "moe",
            "factorized_layers": [4, 5],
            "rank": 16,
            "num_experts": 4,
            "top_k": 1,
            "checkpoint": os.path.join(args.checkpoint_dir, "moe_r16_layers4-5_e4_k1_kd_best.pt"),
        },
    ]

    rows = []

    for cfg in configs:
        print("=" * 80)
        print(f"Benchmarking {cfg['name']}")

        model = build_model(
            model_type=cfg["model_type"],
            factorized_layers=cfg["factorized_layers"],
            rank=cfg["rank"],
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
        ).to(device)

        loaded = load_checkpoint_if_available(model, cfg["checkpoint"], device)
        print(f"Loaded checkpoint: {loaded}")
        print(f"Params: {format_param_count(count_trainable_parameters(model))}")

        acc_metrics = evaluate_accuracy(model, val_loader, device)
        latency_metrics = benchmark_latency(
            model=model,
            dataloader=val_loader,
            device=device,
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
        )

        row = {
            **cfg,
            "checkpoint_loaded": loaded,
            "trainable_params": count_trainable_parameters(model),
            "params_m": count_trainable_parameters(model) / 1e6,
            "accuracy": acc_metrics["accuracy"],
            "loss": acc_metrics["loss"],
            **latency_metrics,
        }
        rows.append(row)

        print(json.dumps(row, indent=2))

    output_path = os.path.join(args.results_dir, "benchmark_results.json")
    save_json(rows, output_path)
    print(f"Saved benchmark results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--measured_batches", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)