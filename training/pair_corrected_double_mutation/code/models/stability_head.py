#!/usr/bin/env python3
"""
stability_head.py
-----------------
Stability heads for the supervised cached-embedding workflow.

Mutation-aware architecture:
    LayerNorm(1536)
      → Linear(1536→768, bias=True) → GELU → Dropout(p_phase)
      → Residual block 1:
            LayerNorm(768) → Linear(768→768, bias=True) → GELU → Dropout(p_phase) + skip
      → Residual block 2:
            LayerNorm(768) → Linear(768→768, bias=True) → GELU → Dropout(p_phase) + skip
      → Linear(768→20, bias=True)

Input:  per-residue SaProt embeddings (pre-final-LayerNorm), shape (*, 1536)
Output: raw per-residue score estimates for 20 standard amino acids, shape (*, 20)

Selected-model readout:
    MutationAwareStabilityHead uses the same WT sequence + WT structure SaProt
    context, then conditions the readout on learned WT and mutant amino-acid
    identity embeddings. These amino-acid embeddings are trainable lookup-table
    parameters in the head; they do not require mutant structures.

ΔΔG derivation:
    ΔΔG(i, a) = φ(i, a) − φ(i, a_wt)

Properties:
    - ΔΔG = 0 for identity substitutions (enforced by subtraction, not trained)
    - WT subtraction is not detached, so gradients flow through mutant and WT channels
    - Natural antisymmetry holds by construction
    - ~2.38M trainable parameters in the Phase 1 head

Parameter count:
    LayerNorm(1536):                         3,072
    Linear(1536→768) + bias:            1,180,416
    Residual block 1:                     592,128
    Residual block 2:                     592,128
    Linear(768→20) + bias:                 15,380
    Total:                              2,383,124
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Amino acid constants
# ---------------------------------------------------------------------------
# Standard 20 amino acids, alphabetical by 1-letter code.
# This ordering defines columns 0–19 of the head output and ΔΔG matrix.
AMINO_ACIDS_20: str = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS_20)}
INDEX_TO_AA: dict[int, str] = {i: aa for i, aa in enumerate(AMINO_ACIDS_20)}
NUM_AMINO_ACIDS: int = 20

# ---------------------------------------------------------------------------
# Head architecture constants
# ---------------------------------------------------------------------------
D_MODEL: int = 1536
D_HIDDEN: int = 768
D_OUT: int = NUM_AMINO_ACIDS
NUM_RESIDUAL_BLOCKS: int = 2
AA_EMBED_DIM: int = 64
HEAD_DROPOUT_BY_PHASE: dict[int, float] = {
    1: 0.10,
    2: 0.10,
    3: 0.15,
    4: 0.20,
}
DEFAULT_HEAD_DROPOUT: float = HEAD_DROPOUT_BY_PHASE[1]


class ResidualBlock(nn.Module):
    """Simple residual MLP block used by the supervised stability head."""

    def __init__(self, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_hidden)
        self.linear = nn.Linear(d_hidden, d_hidden, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.act(x)
        x = self.dropout(x)
        return residual + x


class StabilityHead(nn.Module):
    """
    Deep residual MLP stability prediction head.

    Takes per-residue SaProt embeddings and outputs raw score estimates for each
    of the 20 standard amino acids at every residue position.

    The leading LayerNorm re-normalises the pre-final-LayerNorm hidden state
    that SaProt exposes as ``output.embeddings``.

    Parameters
    ----------
    d_model : int
        Input embedding dimension (default 1536 = SaProt d_model).
    d_hidden : int
        Hidden layer dimension (default 768).
    d_out : int
        Output dimension (default 20 = standard amino acids).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_hidden: int = D_HIDDEN,
        d_out: int = D_OUT,
        dropout: float = DEFAULT_HEAD_DROPOUT,
        num_residual_blocks: int = NUM_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        if num_residual_blocks != NUM_RESIDUAL_BLOCKS:
            raise ValueError(
                f"StabilityHead locks num_residual_blocks={NUM_RESIDUAL_BLOCKS}, "
                f"got {num_residual_blocks}."
            )
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_hidden, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(d_hidden, dropout=dropout) for _ in range(num_residual_blocks)]
        )
        self.output = nn.Linear(d_hidden, d_out, bias=True)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : torch.Tensor
            Per-residue SaProt embeddings (pre-final-LayerNorm).
            Shape: ``(L, 1536)`` or ``(B, L, 1536)``.

        Returns
        -------
        dg_raw : torch.Tensor
            Raw ΔG estimates for 20 amino acids, same leading dims as input.
            Shape: ``(L, 20)`` or ``(B, L, 20)``.
        """
        x = self.norm(embeddings)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output(x)
        return x


class MutationAwareStabilityHead(nn.Module):
    """
    Mutation-conditioned stability readout.

    The SaProt input remains the wild-type sequence plus wild-type structure. For
    each candidate substitution, the head combines:

    - the residue's WT SaProt context embedding,
    - a pooled whole-protein WT SaProt context vector,
    - a learned WT amino-acid identity embedding,
    - a learned mutant amino-acid identity embedding,
    - the learned mutant-minus-WT amino-acid identity delta.

    The amino-acid identity embeddings are ordinary trainable parameters, not
    embeddings generated by running SaProt on mutant structures.
    """

    architecture_name = "mutation_aware_v1"

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_hidden: int = D_HIDDEN,
        aa_embed_dim: int = AA_EMBED_DIM,
        dropout: float = DEFAULT_HEAD_DROPOUT,
        num_residual_blocks: int = NUM_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        if num_residual_blocks != NUM_RESIDUAL_BLOCKS:
            raise ValueError(
                f"MutationAwareStabilityHead locks num_residual_blocks={NUM_RESIDUAL_BLOCKS}, "
                f"got {num_residual_blocks}."
            )
        self.embedding_norm = nn.LayerNorm(d_model)
        self.residue_proj = nn.Linear(d_model, d_hidden, bias=True)
        self.protein_context_norm = nn.LayerNorm(d_model)
        self.protein_context_proj = nn.Linear(d_model, d_hidden, bias=True)
        self.aa_embedding = nn.Embedding(NUM_AMINO_ACIDS, aa_embed_dim)
        self.aa_pair_norm = nn.LayerNorm(aa_embed_dim * 3)
        self.aa_pair_proj = nn.Linear(aa_embed_dim * 3, d_hidden, bias=True)
        self.fusion_norm = nn.LayerNorm(d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(d_hidden, dropout=dropout) for _ in range(num_residual_blocks)]
        )
        self.output = nn.Linear(d_hidden, 1, bias=True)

    def _raw_scores_single(
        self,
        embeddings: torch.Tensor,
        wt_sequence: str,
    ) -> torch.Tensor:
        L = embeddings.shape[0]
        if len(wt_sequence) != L:
            raise ValueError(
                f"WT sequence length ({len(wt_sequence)}) does not match "
                f"residue dimension ({L})."
            )
        wt_indices = sequence_to_wt_indices(wt_sequence).to(embeddings.device)

        context = self.embedding_norm(embeddings)
        residue_hidden = self.residue_proj(context)  # (L, H)

        pooled_context = embeddings.mean(dim=0, keepdim=True)
        pooled_hidden = self.protein_context_proj(
            self.protein_context_norm(pooled_context)
        ).expand(L, -1)  # (L, H)

        aa_ids = torch.arange(NUM_AMINO_ACIDS, device=embeddings.device)
        wt_aa = self.aa_embedding(wt_indices).unsqueeze(1).expand(-1, NUM_AMINO_ACIDS, -1)
        mut_aa = self.aa_embedding(aa_ids).unsqueeze(0).expand(L, -1, -1)
        aa_delta = mut_aa - wt_aa
        aa_pair = torch.cat([wt_aa, mut_aa, aa_delta], dim=-1)
        aa_hidden = self.aa_pair_proj(self.aa_pair_norm(aa_pair))  # (L, 20, H)

        x = residue_hidden.unsqueeze(1) + pooled_hidden.unsqueeze(1) + aa_hidden
        x = self.dropout(self.act(self.fusion_norm(x)))
        for block in self.residual_blocks:
            x = block(x)
        return self.output(x).squeeze(-1)  # (L, 20)

    def forward(
        self,
        embeddings: torch.Tensor,
        wt_sequence: str | Sequence[str],
        *,
        lengths: Sequence[int] | None = None,
        return_raw: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Predict a full ``(L, 20)`` or ``(B, L, 20)`` ΔΔG matrix.

        Identity substitutions are made exactly zero by subtracting the raw
        identity-substitution score at each residue.
        """
        has_batch = embeddings.ndim == 3
        if not has_batch:
            embeddings = embeddings.unsqueeze(0)

        if isinstance(wt_sequence, str):
            sequences = [wt_sequence]
        else:
            sequences = list(wt_sequence)

        B, L_max, _ = embeddings.shape
        if len(sequences) != B:
            raise ValueError(
                f"Expected {B} WT sequence(s) for batch, received {len(sequences)}."
            )

        ddg = embeddings.new_zeros((B, L_max, NUM_AMINO_ACIDS))
        raw_out = embeddings.new_zeros((B, L_max, NUM_AMINO_ACIDS))
        for batch_idx, sequence in enumerate(sequences):
            L = int(lengths[batch_idx]) if lengths is not None else len(sequence)
            sequence = sequence[:L]
            raw = self._raw_scores_single(embeddings[batch_idx, :L], sequence)
            wt_indices = sequence_to_wt_indices(sequence).to(raw.device)
            wt_raw = raw.gather(dim=1, index=wt_indices.unsqueeze(-1))
            raw_out[batch_idx, :L] = raw
            ddg[batch_idx, :L] = raw - wt_raw

        if not has_batch:
            ddg = ddg.squeeze(0)
            raw_out = raw_out.squeeze(0)
        if return_raw:
            return ddg, raw_out
        return ddg


# ---------------------------------------------------------------------------
# ΔΔG computation
# ---------------------------------------------------------------------------

def compute_ddg(
    dg_raw: torch.Tensor,
    wt_sequence: str,
) -> torch.Tensor:
    """
    Derive ΔΔG from raw ΔG and the wild-type sequence.

    .. math::
        \\Delta\\Delta G(i, a) = \\Delta G_{raw}(i, a) - \\Delta G_{raw}(i, a_{wt})

    Parameters
    ----------
    dg_raw : torch.Tensor
        Raw ΔG from :class:`StabilityHead`.
        Shape ``(L, 20)`` or ``(B, L, 20)``.
    wt_sequence : str
        Wild-type amino acid sequence (standard 1-letter codes), length *L*.
        Non-standard residues (X, B, U, Z, etc.) are not supported — the
        caller must filter them before calling this function.

    Returns
    -------
    ddg : torch.Tensor
        ΔΔG predictions, same shape as *dg_raw*.
        Identity substitutions are exactly zero by construction.

    Raises
    ------
    ValueError
        If *wt_sequence* length does not match the residue dimension, or
        if it contains residues outside the standard 20.
    """
    has_batch = dg_raw.ndim == 3
    L = dg_raw.shape[1] if has_batch else dg_raw.shape[0]

    if len(wt_sequence) != L:
        raise ValueError(
            f"WT sequence length ({len(wt_sequence)}) does not match "
            f"residue dimension ({L})."
        )

    # Validate all residues are in the standard 20.
    bad = [
        (i, aa) for i, aa in enumerate(wt_sequence) if aa not in AA_TO_INDEX
    ]
    if bad:
        positions = ", ".join(f"{i}:{aa}" for i, aa in bad[:5])
        raise ValueError(
            f"Non-standard residues in WT sequence at positions: {positions}"
            f"{' (and more)' if len(bad) > 5 else ''}. "
            f"Only the 20 standard amino acids ({AMINO_ACIDS_20}) are supported."
        )

    # Build wild-type index tensor: (L,)
    wt_indices = torch.tensor(
        [AA_TO_INDEX[aa] for aa in wt_sequence],
        dtype=torch.long,
        device=dg_raw.device,
    )

    # Gather the WT channel ΔG at every position.
    if has_batch:
        B = dg_raw.shape[0]
        idx = wt_indices.unsqueeze(0).unsqueeze(-1).expand(B, L, 1)  # (B, L, 1)
        wt_dg = dg_raw.gather(dim=2, index=idx)                     # (B, L, 1)
    else:
        idx = wt_indices.unsqueeze(-1)                                # (L, 1)
        wt_dg = dg_raw.gather(dim=1, index=idx)                     # (L, 1)

    return dg_raw - wt_dg


def sequence_to_wt_indices(wt_sequence: str) -> torch.Tensor:
    """Convert a WT sequence string to a tensor of column indices into the 20-AA output."""
    return torch.tensor(
        [AA_TO_INDEX[aa] for aa in wt_sequence], dtype=torch.long
    )


def predict_ddg_from_head(
    head: nn.Module,
    embeddings: torch.Tensor,
    wt_sequence: str | Sequence[str],
    *,
    lengths: Sequence[int] | None = None,
    return_raw: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform prediction helper for the current mutation-aware head and legacy head.

    New packages use :class:`MutationAwareStabilityHead`, which already predicts
    ΔΔG directly from WT SaProt context plus WT/mutant amino-acid identity. The
    legacy head is still supported for older checkpoints by applying the original
    channel-subtraction rule.
    """
    if isinstance(head, MutationAwareStabilityHead):
        return head(
            embeddings,
            wt_sequence,
            lengths=lengths,
            return_raw=return_raw,
        )

    raw = head(embeddings)
    has_batch = raw.ndim == 3
    if isinstance(wt_sequence, str):
        sequences = [wt_sequence]
    else:
        sequences = list(wt_sequence)

    if not has_batch:
        ddg = compute_ddg(raw, sequences[0])
        return (ddg, raw) if return_raw else ddg

    ddg = torch.zeros_like(raw)
    for batch_idx, sequence in enumerate(sequences):
        L = int(lengths[batch_idx]) if lengths is not None else len(sequence)
        ddg[batch_idx, :L] = compute_ddg(
            raw[batch_idx, :L].unsqueeze(0),
            sequence[:L],
        ).squeeze(0)
    if return_raw:
        return ddg, raw
    return ddg
