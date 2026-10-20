from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class Evo2LoRALinear(nn.Module):
    """LoRA wrapper for Evo2's standard ``nn.Linear`` modules."""

    def __init__(
        self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05
    ) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_output + update * self.scaling
