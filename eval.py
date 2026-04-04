import argparse

import torch

from src.data import build_dataloaders, get_tokenizer
from src.models import build_model
from src.utils import count_trainable_parameters, format_param_count, set_seed


def evaluate(model, dataloader, device):
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
    return {"accuracy": correct / total, "loss": total_loss / total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="baseline", choices=["baseline", "lowrank", "moe"])
    parser.add_argument("--factorized_layers", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer()
    _, val_loader, _ = build_dataloaders(tokenizer=tokenizer, max_length=args.max_length, train_batch_size=args.batch_size, eval_batch_size=args.batch_size)

    model = build_model(
        model_type=args.model_type,
        factorized_layers=args.factorized_layers,
        rank=args.rank,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    metrics = evaluate(model, val_loader, device)
    print(f"Model type: {args.model_type}")
    print(f"Trainable params: {format_param_count(count_trainable_parameters(model))}")
    print(metrics)
