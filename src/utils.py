import json
import os
import random
import time
from typing import Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def format_param_count(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def time_forward_pass(model: torch.nn.Module, batch: Dict[str, torch.Tensor], device: torch.device, warmup: int = 5, steps: int = 20) -> float:
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(steps):
            _ = model(**batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()
    return (end - start) / steps
