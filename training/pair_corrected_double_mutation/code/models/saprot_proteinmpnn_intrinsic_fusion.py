#!/usr/bin/env python3
"""Intrinsic SaProt + ProteinMPNN fusion head.

SaProt is the primary residue representation. ProteinMPNN is treated as a
frozen inverse-folding structural prior and fused before the mutation readout
through residue-level cross-attention plus mutation-specific log-odds features.
This is not late prediction averaging.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from models.stability_head import AA_TO_INDEX, NUM_AMINO_ACIDS


AA_EMBED_DIM = 64
DEFAULT_HIDDEN = 768
LOCAL_CONTACT_TOP_K = 16
LOCAL_CONTACT_CUTOFF_A = 10.0
LOCAL_CONTACT_DISTANCE_SCALE_A = 4.0


class ResidualBlock(nn.Module):
    def __init__(self, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.linear = nn.Linear(d_hidden, d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.act(self.linear(self.norm(x))))


class SaProtProteinMPNNIntrinsicFusionHead(nn.Module):
    """Mutation-aware head with explicit SaProt/ProteinMPNN representation fusion."""

    architecture_name = "saprot_proteinmpnn_intrinsic_cross_attention_v1"

    def __init__(
        self,
        *,
        d_saprot: int,
        d_hidden: int = DEFAULT_HIDDEN,
        aa_embed_dim: int = AA_EMBED_DIM,
        dropout: float = 0.10,
        num_attention_heads: int = 8,
        num_residual_blocks: int = 2,
        local_contact_top_k: int = LOCAL_CONTACT_TOP_K,
        local_contact_cutoff: float = LOCAL_CONTACT_CUTOFF_A,
        local_contact_distance_scale: float = LOCAL_CONTACT_DISTANCE_SCALE_A,
    ) -> None:
        super().__init__()
        self.saprot_norm = nn.LayerNorm(d_saprot)
        self.saprot_proj = nn.Linear(d_saprot, d_hidden)
        self.mpnn_logit_norm = nn.LayerNorm(NUM_AMINO_ACIDS)
        self.mpnn_residue_proj = nn.Linear(NUM_AMINO_ACIDS, d_hidden)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_hidden,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_hidden * 3),
            nn.Linear(d_hidden * 3, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.Sigmoid(),
        )
        self.fused_norm = nn.LayerNorm(d_hidden)
        self.protein_context_proj = nn.Linear(d_hidden, d_hidden)
        self.local_context_proj = nn.Linear(d_hidden, d_hidden)

        self.aa_embedding = nn.Embedding(NUM_AMINO_ACIDS, aa_embed_dim)
        self.substitution_pair_embedding = nn.Embedding(NUM_AMINO_ACIDS * NUM_AMINO_ACIDS, aa_embed_dim)
        self.aa_pair_norm = nn.LayerNorm(aa_embed_dim * 4)
        self.aa_pair_proj = nn.Linear(aa_embed_dim * 4, d_hidden)

        self.mpnn_pair_norm = nn.LayerNorm(4)
        self.mpnn_pair_proj = nn.Linear(4, d_hidden)
        self.fusion_norm = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(d_hidden, dropout=dropout) for _ in range(num_residual_blocks)]
        )
        self.output = nn.Linear(d_hidden, 1)
        self.local_contact_top_k = int(local_contact_top_k)
        self.local_contact_cutoff = float(local_contact_cutoff)
        self.local_contact_distance_scale = float(local_contact_distance_scale)

    @staticmethod
    def _length_mask(lengths: Sequence[int], max_len: int, device: torch.device) -> torch.Tensor:
        arange = torch.arange(max_len, device=device).unsqueeze(0)
        length_tensor = torch.tensor(list(lengths), dtype=torch.long, device=device).unsqueeze(1)
        return arange < length_tensor

    def _local_context_single(self, context: torch.Tensor, ca_coordinates: torch.Tensor | None) -> torch.Tensor:
        length = context.shape[0]
        if ca_coordinates is None:
            return context.new_zeros((length, context.shape[-1]))
        if ca_coordinates.ndim != 2 or ca_coordinates.shape != (length, 3):
            raise ValueError(f"C-alpha coordinates must have shape ({length}, 3), got {tuple(ca_coordinates.shape)}")
        coords = ca_coordinates.to(device=context.device, dtype=context.dtype)
        valid = torch.isfinite(coords).all(dim=-1)
        if valid.sum() < 2:
            return context.new_zeros((length, context.shape[-1]))
        coords_clean = torch.where(valid.unsqueeze(-1), coords, torch.zeros_like(coords))
        distances = torch.cdist(coords_clean.unsqueeze(0), coords_clean.unsqueeze(0)).squeeze(0)
        pair_valid = valid.unsqueeze(0) & valid.unsqueeze(1)
        pair_valid = pair_valid & ~torch.eye(length, dtype=torch.bool, device=context.device)
        pair_valid = pair_valid & (distances <= self.local_contact_cutoff)
        weights = torch.exp(-distances / max(self.local_contact_distance_scale, 1e-6)) * pair_valid.to(context.dtype)
        if self.local_contact_top_k > 0 and self.local_contact_top_k < length:
            top_values, top_indices = torch.topk(weights, k=self.local_contact_top_k, dim=1, largest=True, sorted=False)
            sparse = torch.zeros_like(weights)
            sparse.scatter_(1, top_indices, top_values)
            weights = sparse
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        local = weights @ context / denom
        return local * (weights.sum(dim=1, keepdim=True) > 0).to(context.dtype)

    def _fuse_residues(
        self,
        saprot_embeddings: torch.Tensor,
        proteinmpnn_logits: torch.Tensor,
        *,
        lengths: Sequence[int],
        proteinmpnn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        saprot_hidden = self.saprot_proj(self.saprot_norm(saprot_embeddings))
        mpnn_hidden = self.mpnn_residue_proj(self.mpnn_logit_norm(proteinmpnn_logits))
        sequence_mask = self._length_mask(lengths, saprot_hidden.shape[1], saprot_hidden.device)
        attention_mask = sequence_mask
        if proteinmpnn_mask is not None:
            attention_mask = sequence_mask & proteinmpnn_mask.to(device=saprot_hidden.device).bool()
            has_attention_key = attention_mask.any(dim=1, keepdim=True)
            attention_mask = torch.where(has_attention_key, attention_mask, sequence_mask)
        attn_out, _ = self.cross_attention(
            query=saprot_hidden,
            key=mpnn_hidden,
            value=mpnn_hidden,
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        gate = self.fusion_gate(torch.cat([saprot_hidden, mpnn_hidden, attn_out], dim=-1))
        fused = saprot_hidden + gate * attn_out + (1.0 - gate) * mpnn_hidden
        return self.fused_norm(fused) * sequence_mask.unsqueeze(-1).to(fused.dtype)

    def _raw_scores_single(
        self,
        fused: torch.Tensor,
        proteinmpnn_logits: torch.Tensor,
        wt_sequence: str,
        ca_coordinates: torch.Tensor | None,
    ) -> torch.Tensor:
        length = fused.shape[0]
        if len(wt_sequence) != length:
            raise ValueError(f"WT sequence length {len(wt_sequence)} does not match fused length {length}")
        wt_indices = torch.tensor([AA_TO_INDEX[aa] for aa in wt_sequence], dtype=torch.long, device=fused.device)

        residue_hidden = fused
        pooled = self.protein_context_proj(fused.mean(dim=0, keepdim=True)).expand(length, -1)
        local = self.local_context_proj(self._local_context_single(fused, ca_coordinates))

        aa_ids = torch.arange(NUM_AMINO_ACIDS, device=fused.device)
        wt_aa = self.aa_embedding(wt_indices).unsqueeze(1).expand(-1, NUM_AMINO_ACIDS, -1)
        mut_aa = self.aa_embedding(aa_ids).unsqueeze(0).expand(length, -1, -1)
        aa_delta = mut_aa - wt_aa
        pair_ids = wt_indices.unsqueeze(1) * NUM_AMINO_ACIDS + aa_ids.unsqueeze(0)
        pair_aa = self.substitution_pair_embedding(pair_ids)
        aa_hidden = self.aa_pair_proj(self.aa_pair_norm(torch.cat([wt_aa, mut_aa, aa_delta, pair_aa], dim=-1)))

        log_probs = torch.log_softmax(proteinmpnn_logits, dim=-1)
        wt_log_prob = log_probs.gather(dim=1, index=wt_indices.unsqueeze(1)).expand(-1, NUM_AMINO_ACIDS)
        mut_log_prob = log_probs
        log_odds = mut_log_prob - wt_log_prob
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1, keepdim=True).expand(-1, NUM_AMINO_ACIDS)
        mpnn_pair = torch.stack([wt_log_prob, mut_log_prob, log_odds, entropy], dim=-1)
        mpnn_pair_hidden = self.mpnn_pair_proj(self.mpnn_pair_norm(mpnn_pair))

        x = residue_hidden.unsqueeze(1) + pooled.unsqueeze(1) + local.unsqueeze(1) + aa_hidden + mpnn_pair_hidden
        x = self.dropout(self.act(self.fusion_norm(x)))
        for block in self.residual_blocks:
            x = block(x)
        return self.output(x).squeeze(-1)

    def forward(
        self,
        saprot_embeddings: torch.Tensor,
        proteinmpnn_logits: torch.Tensor,
        wt_sequence: str | Sequence[str],
        *,
        lengths: Sequence[int] | None = None,
        ca_coordinates: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
        proteinmpnn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        has_batch = saprot_embeddings.ndim == 3
        if not has_batch:
            saprot_embeddings = saprot_embeddings.unsqueeze(0)
            proteinmpnn_logits = proteinmpnn_logits.unsqueeze(0)
        sequences = [wt_sequence] if isinstance(wt_sequence, str) else list(wt_sequence)
        batch, length_max, _ = saprot_embeddings.shape
        if len(sequences) != batch:
            raise ValueError(f"Expected {batch} sequences, got {len(sequences)}")
        if lengths is None:
            lengths = [len(sequence) for sequence in sequences]

        if ca_coordinates is None:
            ca_batches: list[torch.Tensor | None] = [None] * batch
        elif torch.is_tensor(ca_coordinates):
            coords = ca_coordinates
            if coords.ndim == 2 and batch == 1:
                coords = coords.unsqueeze(0)
            ca_batches = [coords[i] for i in range(batch)]
        else:
            ca_batches = list(ca_coordinates)

        fused = self._fuse_residues(
            saprot_embeddings,
            proteinmpnn_logits,
            lengths=lengths,
            proteinmpnn_mask=proteinmpnn_mask,
        )
        ddg = saprot_embeddings.new_zeros((batch, length_max, NUM_AMINO_ACIDS))
        for i, sequence in enumerate(sequences):
            length = int(lengths[i])
            raw = self._raw_scores_single(
                fused[i, :length],
                proteinmpnn_logits[i, :length].to(device=fused.device, dtype=fused.dtype),
                sequence[:length],
                ca_batches[i],
            )
            wt_indices = torch.tensor([AA_TO_INDEX[aa] for aa in sequence[:length]], dtype=torch.long, device=raw.device)
            wt_raw = raw.gather(dim=1, index=wt_indices.unsqueeze(-1))
            ddg[i, :length] = raw - wt_raw
        return ddg.squeeze(0) if not has_batch else ddg
