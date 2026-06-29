from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        h1 = 512 if num_classes < 100 else 768
        h2 = 128 if num_classes < 100 else 256
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.LayerNorm(h1),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.LayerNorm(h2),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, task_dims: Mapping[str, int], dropout: float = 0.2):
        super().__init__()
        self.task_names = list(task_dims)
        self.heads = nn.ModuleDict(
            {name: SequenceMLP(input_dim, int(dim), dropout=dropout) for name, dim in task_dims.items()}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: self.heads[name](x) for name in self.task_names}


class GatedMILHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_dims: Mapping[str, int],
        attention_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.attention_v = nn.Linear(input_dim, attention_dim)
        self.attention_u = nn.Linear(input_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        self.classifier = MultiTaskMLP(input_dim, task_dims, dropout=dropout)

    def forward(
        self,
        windows: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ):
        scores = self.attention_w(torch.tanh(self.attention_v(windows)) * torch.sigmoid(self.attention_u(windows)))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=1).to(windows.dtype)
        pooled = torch.bmm(weights.unsqueeze(1), windows).squeeze(1)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, weights
        return logits


GatedMultiTaskMILHead = GatedMILHead
