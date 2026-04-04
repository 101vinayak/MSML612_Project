import argparse
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.data import build_dataloaders, get_tokenizer
from src.models import TEACHER_CKPT, build_model
from src.utils import count_trainable_parameters, ensure_dir, format_param_count, save_json, set_seed


def evaluate(model, dataloader, device) -> Dict[str, float]:
    model.eval()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            preds = outputs.logits.argmax(dim=-1)

            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            total_loss += loss.item() * batch["labels"].size(0)

    return {
        "accuracy": correct / total,
        "loss": total_loss / total,
    }


def kd_loss(student_logits, teacher_logits, temperature: float = 2.0) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def get_run_name(args) -> str:
    layer_str = "-".join(map(str, args.factorized_layers))
    name = f"{args.model_type}_r{args.rank}_layers{layer_str}"

    if args.model_type == "moe":
        name += f"_e{args.num_experts}_k{args.top_k}"

    if args.use_distillation and args.model_type != "baseline":
        name += "_kd"

    return name


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer()
    train_loader, val_loader, _ = build_dataloaders(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
    )

    factorized_layers = list(args.factorized_layers)

    model = build_model(
        model_type=args.model_type,
        factorized_layers=factorized_layers,
        rank=args.rank,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device)

    teacher_model = None
    if args.use_distillation and args.model_type != "baseline":
        teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_CKPT).to(device)
        teacher_model.eval()

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    run_name = get_run_name(args)

    print(f"Model type: {args.model_type}")
    print(f"Run name: {run_name}")
    print(f"Trainable params: {format_param_count(count_trainable_parameters(model))}")
    print(f"Using distillation: {teacher_model is not None}")

    history = []
    best_acc = -1.0

    ensure_dir(args.checkpoint_dir)
    ensure_dir(args.results_dir)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            ce = outputs.loss
            loss = ce
            kd = torch.tensor(0.0, device=device)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                kd = kd_loss(outputs.logits, teacher_outputs.logits, temperature=args.temperature)
                loss = args.alpha * ce + args.beta * kd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_ce += ce.item()
            running_kd += kd.item()

            progress.set_postfix(
                loss=running_loss / max(1, progress.n),
                ce=running_ce / max(1, progress.n),
                kd=running_kd / max(1, progress.n),
            )

        val_metrics = evaluate(model, val_loader, device)

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            "train_ce_loss": running_ce / len(train_loader),
            "train_kd_loss": running_kd / len(train_loader),
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
        }

        history.append(epoch_record)
        print(epoch_record)

        last_ckpt = os.path.join(args.checkpoint_dir, f"{run_name}_last.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
            },
            last_ckpt,
        )

        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            best_ckpt = os.path.join(args.checkpoint_dir, f"{run_name}_best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                },
                best_ckpt,
            )

    metrics = {
        "run_name": run_name,
        "model_type": args.model_type,
        "factorized_layers": factorized_layers,
        "rank": args.rank,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "use_distillation": args.use_distillation,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "beta": args.beta,
        "trainable_params": count_trainable_parameters(model),
        "best_val_accuracy": best_acc,
        "history": history,
    }

    metrics_path = os.path.join(args.results_dir, f"{run_name}_metrics.json")
    save_json(metrics, metrics_path)

    print(f"Saved metrics to {metrics_path}")
    print("Saved checkpoints.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "lowrank", "moe"])
    parser.add_argument("--factorized_layers", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--use_distillation", action="store_true")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()
    train(args)