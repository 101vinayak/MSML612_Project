from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


TEACHER_CKPT = "distilbert-base-uncased-finetuned-sst-2-english"


class LowRankLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.u = nn.Linear(in_features, rank, bias=False)
        self.v = nn.Linear(rank, out_features, bias=bias)
        nn.init.normal_(self.u.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.02)
        if bias and self.v.bias is not None:
            nn.init.zeros_(self.v.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(self.u(x))


class LowRankFFN(nn.Module):
    def __init__(self, original_ffn, rank: int):
        super().__init__()
        hidden_dim = original_ffn.lin1.weight.shape[1]
        ffn_dim = original_ffn.lin1.weight.shape[0]
        self.lin1 = LowRankLinear(hidden_dim, ffn_dim, rank=rank, bias=False)
        self.lin2 = original_ffn.lin2
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(original_ffn.dropout.p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class MoEFFN(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, rank, num_experts=4, top_k=1, dropout=0.1, original_lin2=None):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, rank, bias=False),
                nn.Linear(rank, ffn_dim, bias=False)
            )
            for _ in range(num_experts)
        ])

        self.activation = nn.GELU()
        self.lin2 = original_lin2 if original_lin2 is not None else nn.Linear(ffn_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden_dim]
        router_logits = self.router(x)                          # [B, S, E]
        routing_weights = F.softmax(router_logits, dim=-1)     # [B, S, E]
        _, topk_idx = torch.topk(routing_weights, k=self.top_k, dim=-1)

        # Mixture in FFN space, not hidden space
        mixed_output = torch.zeros(
            x.size(0), x.size(1), self.lin2.in_features,
            device=x.device, dtype=x.dtype
        )  # [B, S, ffn_dim]

        for expert_id, expert in enumerate(self.experts):
            expert_out = expert(x)  # [B, S, ffn_dim]
            mask = (topk_idx == expert_id).any(dim=-1).float().unsqueeze(-1)   # [B, S, 1]
            weight = routing_weights[..., expert_id].unsqueeze(-1)              # [B, S, 1]
            mixed_output = mixed_output + expert_out * mask * weight

        x = self.activation(mixed_output)
        x = self.lin2(x)     # back to hidden_dim
        x = self.dropout(x)
        return x


def _get_dims(model) -> tuple:
    config = model.config
    hidden_dim = config.dim
    ffn_dim = config.hidden_dim
    dropout = config.dropout
    return hidden_dim, ffn_dim, dropout


def build_model(model_type: str = "baseline", factorized_layers: Optional[Iterable[int]] = None, rank: int = 64, num_experts: int = 4, top_k: int = 1):
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_CKPT)
    if model_type == "baseline":
        return model

    factorized_layers = list(factorized_layers or [4, 5])
    hidden_dim, ffn_dim, dropout = _get_dims(model)

    for idx in factorized_layers:
        layer = model.distilbert.transformer.layer[idx]
        if model_type == "lowrank":
            new_ffn = LowRankFFN(layer.ffn, rank=rank)
        elif model_type == "moe":
            new_ffn = MoEFFN(
                hidden_dim=hidden_dim,
                ffn_dim=ffn_dim,
                rank=rank,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                original_lin2=layer.ffn.lin2,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        layer.ffn = new_ffn
    return model
