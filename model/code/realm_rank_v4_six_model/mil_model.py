#!/usr/bin/env python3
"""Gated-attention MIL wrapper for ViraLM sequence classification backbones."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import last_hidden


class MILBase(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.backbone_trainable = False
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone_trainable = False
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        self.backbone_trainable = True
        self.backbone.train()
        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


class ViraLM_MIL_Gated(MILBase):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int = 768,
        attention_dim: int = 256,
        num_classes: int = 2,
    ):
        super().__init__(backbone)
        self.hidden_size = int(hidden_size)
        self.attention_dim = int(attention_dim)
        self.num_classes = int(num_classes)
        self.attention_V = nn.Linear(self.hidden_size, self.attention_dim)
        self.attention_U = nn.Linear(self.hidden_size, self.attention_dim)
        self.attention_w = nn.Linear(self.attention_dim, 1)
        self.seq_classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.hidden_size // 2, self.num_classes),
        )

    def forward(
        self,
        input_ids_list: Sequence[torch.Tensor],
        attention_mask_list: Sequence[torch.Tensor],
        sub_chunk_size: int = 8,
        return_frag_logits: bool = True,
        return_hidden: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        seq_logits = []
        seq_attentions = []
        all_frag_logits = []
        h_grouped = []
        device = self.device

        for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            n_frag = input_ids.size(0)

            h_parts = []
            z_parts = []
            enable_grad = torch.is_grad_enabled() and self.backbone_trainable
            with torch.set_grad_enabled(enable_grad):
                for start in range(0, n_frag, sub_chunk_size):
                    out = self.backbone(
                        input_ids=input_ids[start : start + sub_chunk_size],
                        attention_mask=attention_mask[start : start + sub_chunk_size],
                        output_hidden_states=True,
                    )
                    hidden = last_hidden(out.hidden_states)
                    if hidden is None:
                        raise RuntimeError("Backbone did not return hidden states")
                    h_parts.append(hidden[:, 0, :].contiguous())
                    z_parts.append(out.logits)
                    del out, hidden

                h_i = torch.cat(h_parts, dim=0)
                z_i = torch.cat(z_parts, dim=0)

            e_i = self.attention_w(torch.tanh(self.attention_V(h_i)) * torch.sigmoid(self.attention_U(h_i)))
            a_i = F.softmax(e_i.float(), dim=0).to(h_i.dtype)
            seq_attentions.append(a_i.detach().cpu())
            h_seq = torch.mm(a_i.t(), h_i)
            seq_logits.append(self.seq_classifier(h_seq))

            if return_frag_logits:
                all_frag_logits.append(z_i.detach().cpu())
            if return_hidden:
                h_grouped.append(h_i)

        return (
            torch.cat(seq_logits, dim=0),
            seq_attentions,
            all_frag_logits if return_frag_logits else None,
            h_grouped if return_hidden else None,
        )
