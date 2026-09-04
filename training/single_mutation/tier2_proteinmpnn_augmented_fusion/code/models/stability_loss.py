#!/usr/bin/env python3
"""
stability_loss.py
-----------------
Loss functions for the supervised cached-embedding stability head.

Primary evaluation loss: **Masked Mean Squared Error (MSE)**.

Training can use a calibrated weighted MSE plus within-protein rank loss. The
weights are intentionally based only on the true DDG label magnitude/sign, not
on held-out dataset identity or non-DDG proxy performance.

Per-protein loss (training objective):

    L_p  =  (1 / N_p) × Σ_i Σ_a  M_p(i,a) × ( ΔΔG_pred(i,a) − ΔΔG_true(i,a) )²

    where N_p = Σ_{i,a} M_p(i,a)   (number of observed mutations for protein p)

Batch loss:

    L_batch  =  (1 / B_active) × Σ_p  L_p

    where B_active = number of proteins with N_p > 0.

Mask conventions:
    M(i, a) = 1   if an experimental ΔΔG label exists for position i → amino acid a.
    M(i, a_wt) = 0   always — the wild-type channel has ΔΔG ≡ 0 by construction
                       and is never trained against.

Why MSE (not Huber):
    - MegaScale has high precision (median 95 % CI of 0.14 kcal/mol)
    - MSE gives stronger gradient signal for large residuals
    - Consistent with ThermoMPNN and reference
    - One fewer hyperparameter
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from models.stability_head import AA_TO_INDEX, NUM_AMINO_ACIDS


class MaskedMSELoss(nn.Module):
    """
    Masked MSE loss for stability prediction.

    Computes per-protein masked MSE, then averages over active proteins in the
    batch. In the workflow-aligned Phase 1 trainer the batch size is one protein,
    so the returned scalar is the per-protein masked MSE directly.
    """

    def forward(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        ddg_pred : torch.Tensor
            Predicted ΔΔG.  Shape ``(B, L, 20)``.
        ddg_true : torch.Tensor
            Target ΔΔG.  Shape ``(B, L, 20)``.
        mask : torch.Tensor
            Binary mask.  Shape ``(B, L, 20)``.
            ``1`` where an experimental label exists, ``0`` otherwise.
            The wild-type column at each position must be ``0``.

        Returns
        -------
        loss : torch.Tensor
            Scalar batch loss (requires grad).
        """
        sq_err = (ddg_pred - ddg_true).pow(2)        # (B, L, 20)
        masked_sq_err = sq_err * mask                 # (B, L, 20)

        per_protein_sum = masked_sq_err.sum(dim=(1, 2))   # (B,)
        per_protein_count = mask.sum(dim=(1, 2))          # (B,)

        # Per-protein MSE; proteins with zero labels → 0.
        per_protein_loss = torch.where(
            per_protein_count > 0,
            per_protein_sum / per_protein_count,
            torch.zeros_like(per_protein_sum),
        )

        # Average only over proteins that contributed.
        n_active = (per_protein_count > 0).sum()
        if n_active == 0:
            return torch.tensor(0.0, device=ddg_pred.device, requires_grad=True)
        return per_protein_loss.sum() / n_active


class MagnitudeCalibratedMaskedMSELoss(nn.Module):
    """
    Masked MSE with capped continuous weights for sign and DDG magnitude.

    The unweighted MSE is still used for validation/test reporting elsewhere; this
    weighted variant is meant for the training objective only.
    """

    def __init__(
        self,
        stabilizing_weight: float = 0.25,
        destabilizing_weight: float = 0.15,
        magnitude_weight: float = 0.75,
        magnitude_start: float = 0.50,
        magnitude_scale: float = 1.50,
        max_label_weight: float = 2.50,
    ) -> None:
        super().__init__()
        self.stabilizing_weight = stabilizing_weight
        self.destabilizing_weight = destabilizing_weight
        self.magnitude_weight = magnitude_weight
        self.magnitude_start = magnitude_start
        self.magnitude_scale = magnitude_scale
        self.max_label_weight = max_label_weight

    def build_label_weights(self, ddg_true: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(ddg_true)

        if self.magnitude_weight:
            scale = max(float(self.magnitude_scale), 1e-8)
            magnitude_fraction = ((ddg_true.abs() - self.magnitude_start) / scale).clamp(
                min=0.0,
                max=1.0,
            )
            weights = weights + self.magnitude_weight * magnitude_fraction

        if self.stabilizing_weight:
            weights = weights + self.stabilizing_weight * (
                ddg_true <= -self.magnitude_start
            ).to(ddg_true.dtype)

        if self.destabilizing_weight:
            weights = weights + self.destabilizing_weight * (
                ddg_true >= self.magnitude_start
            ).to(ddg_true.dtype)

        if self.max_label_weight > 0:
            weights = weights.clamp(max=self.max_label_weight)
        return weights

    def forward(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.build_label_weights(ddg_true)
        weighted_mask = mask * weights
        sq_err = (ddg_pred - ddg_true).pow(2) * weighted_mask

        per_protein_sum = sq_err.sum(dim=(1, 2))
        per_protein_weight = weighted_mask.sum(dim=(1, 2))
        per_protein_loss = torch.where(
            per_protein_weight > 0,
            per_protein_sum / per_protein_weight,
            torch.zeros_like(per_protein_sum),
        )
        n_active = (per_protein_weight > 0).sum()
        if n_active == 0:
            return torch.tensor(0.0, device=ddg_pred.device, requires_grad=True)
        return per_protein_loss.sum() / n_active


WeightedMaskedMSELoss = MagnitudeCalibratedMaskedMSELoss


class WithinProteinRankingLoss(nn.Module):
    """
    Pairwise ranking loss over measured mutations within each protein.

    It samples a deterministic, target-sorted subset when a protein has many
    measured mutations so memory remains bounded on full-forward SaProt phases.
    """

    def __init__(
        self,
        margin: float = 0.10,
        temperature: float = 1.0,
        max_rank_items: int = 384,
        magnitude_weight: float = 0.50,
        magnitude_start: float = 0.50,
        magnitude_scale: float = 1.50,
        max_pair_weight: float = 2.00,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.max_rank_items = max_rank_items
        self.magnitude_weight = magnitude_weight
        self.magnitude_start = magnitude_start
        self.magnitude_scale = magnitude_scale
        self.max_pair_weight = max_pair_weight

    def build_pair_weights(self, true: torch.Tensor) -> torch.Tensor:
        weights = torch.ones((true.numel(), true.numel()), device=true.device, dtype=true.dtype)
        if self.magnitude_weight:
            scale = max(float(self.magnitude_scale), 1e-8)
            pair_abs = torch.maximum(true.abs().unsqueeze(0), true.abs().unsqueeze(1))
            magnitude_fraction = ((pair_abs - self.magnitude_start) / scale).clamp(
                min=0.0,
                max=1.0,
            )
            weights = weights + self.magnitude_weight * magnitude_fraction
        if self.max_pair_weight > 0:
            weights = weights.clamp(max=self.max_pair_weight)
        return weights

    def forward(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        B = ddg_pred.shape[0]
        for batch_idx in range(B):
            active = mask[batch_idx] > 0
            if active.sum() < 2:
                continue

            pred = ddg_pred[batch_idx][active].flatten()
            true = ddg_true[batch_idx][active].flatten()
            order = torch.argsort(true)
            pred = pred[order]
            true = true[order]

            if pred.numel() > self.max_rank_items:
                sample_idx = torch.linspace(
                    0,
                    pred.numel() - 1,
                    steps=self.max_rank_items,
                    device=pred.device,
                ).round().long()
                pred = pred[sample_idx]
                true = true[sample_idx]

            true_diff = true.unsqueeze(0) - true.unsqueeze(1)
            upper_triangle = torch.triu(
                torch.ones_like(true_diff, dtype=torch.bool),
                diagonal=1,
            )
            pair_mask = upper_triangle & (true_diff.abs() >= self.margin)
            if pair_mask.sum() == 0:
                continue

            pred_diff = pred.unsqueeze(0) - pred.unsqueeze(1)
            signs = true_diff.sign()
            pair_loss = torch.nn.functional.softplus(
                -signs[pair_mask] * pred_diff[pair_mask] / self.temperature
            )
            pair_weights = self.build_pair_weights(true)[pair_mask]
            losses.append(
                (pair_loss * pair_weights).sum()
                / pair_weights.sum().clamp_min(torch.finfo(pair_weights.dtype).eps)
            )

        if not losses:
            return torch.tensor(0.0, device=ddg_pred.device, requires_grad=True)
        return torch.stack(losses).mean()


class OneSidedSpreadCalibrationLoss(nn.Module):
    """
    Penalize within-protein prediction compression without rewarding overspread.

    Pearson is invariant to a global affine rescale, so this term targets the
    per-protein spread of the learned function. It is one-sided by design:
    only ``std_pred < std_true`` contributes.
    """

    def forward(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        B = ddg_pred.shape[0]
        for batch_idx in range(B):
            active = mask[batch_idx] > 0
            if active.sum() < 2:
                continue

            pred = ddg_pred[batch_idx][active].flatten()
            true = ddg_true[batch_idx][active].flatten()
            true_std = true.std(unbiased=False)
            pred_std = pred.std(unbiased=False)
            losses.append(torch.relu(true_std - pred_std).pow(2))

        if not losses:
            return torch.tensor(0.0, device=ddg_pred.device, requires_grad=True)
        return torch.stack(losses).mean()


class CompositeStabilityLoss(nn.Module):
    """
    Training objective for the mutation-aware readout.

    Validation and test metrics should continue to use unweighted
    :class:`MaskedMSELoss` plus the existing Pearson/Spearman reporting.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        rank_weight: float = 0.15,
        stabilizing_weight: float = 0.25,
        destabilizing_weight: float = 0.15,
        magnitude_weight: float = 0.75,
        magnitude_start: float = 0.50,
        magnitude_scale: float = 1.50,
        max_label_weight: float = 2.50,
        rank_margin: float = 0.10,
        rank_temperature: float = 1.0,
        max_rank_items: int = 384,
        rank_magnitude_weight: float = 0.50,
        rank_magnitude_start: float = 0.50,
        rank_magnitude_scale: float = 1.50,
        rank_max_pair_weight: float = 2.00,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.weighted_mse = MagnitudeCalibratedMaskedMSELoss(
            stabilizing_weight=stabilizing_weight,
            destabilizing_weight=destabilizing_weight,
            magnitude_weight=magnitude_weight,
            magnitude_start=magnitude_start,
            magnitude_scale=magnitude_scale,
            max_label_weight=max_label_weight,
        )
        self.ranking = WithinProteinRankingLoss(
            margin=rank_margin,
            temperature=rank_temperature,
            max_rank_items=max_rank_items,
            magnitude_weight=rank_magnitude_weight,
            magnitude_start=rank_magnitude_start,
            magnitude_scale=rank_magnitude_scale,
            max_pair_weight=rank_max_pair_weight,
        )

    def forward_terms(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mse = self.weighted_mse(ddg_pred, ddg_true, mask)
        rank = self.ranking(ddg_pred, ddg_true, mask)
        total = self.mse_weight * mse + self.rank_weight * rank
        return {
            "total": total,
            "mse": mse,
            "rank": rank,
        }

    def forward(
        self,
        ddg_pred: torch.Tensor,
        ddg_true: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_terms(ddg_pred, ddg_true, mask)["total"]

    def to_config(self) -> dict[str, float | int | str]:
        return {
            "name": self.__class__.__name__,
            "mse_weight": self.mse_weight,
            "rank_weight": self.rank_weight,
            "stabilizing_weight": self.weighted_mse.stabilizing_weight,
            "destabilizing_weight": self.weighted_mse.destabilizing_weight,
            "magnitude_weight": self.weighted_mse.magnitude_weight,
            "magnitude_start": self.weighted_mse.magnitude_start,
            "magnitude_scale": self.weighted_mse.magnitude_scale,
            "max_label_weight": self.weighted_mse.max_label_weight,
            "rank_margin": self.ranking.margin,
            "rank_temperature": self.ranking.temperature,
            "max_rank_items": self.ranking.max_rank_items,
            "rank_magnitude_weight": self.ranking.magnitude_weight,
            "rank_magnitude_start": self.ranking.magnitude_start,
            "rank_magnitude_scale": self.ranking.magnitude_scale,
            "rank_max_pair_weight": self.ranking.max_pair_weight,
        }


def add_composite_loss_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("magnitude-calibrated training objective")
    group.add_argument("--mse-weight", type=float, default=1.0)
    group.add_argument("--rank-weight", type=float, default=0.15)
    group.add_argument("--stabilizing-weight", type=float, default=0.25)
    group.add_argument("--destabilizing-weight", type=float, default=0.15)
    group.add_argument("--magnitude-weight", type=float, default=0.75)
    group.add_argument("--magnitude-start", type=float, default=0.50)
    group.add_argument("--magnitude-scale", type=float, default=1.50)
    group.add_argument("--max-label-weight", type=float, default=2.50)
    group.add_argument("--rank-margin", type=float, default=0.10)
    group.add_argument("--rank-temperature", type=float, default=1.0)
    group.add_argument("--max-rank-items", type=int, default=384)
    group.add_argument("--rank-magnitude-weight", type=float, default=0.50)
    group.add_argument("--rank-magnitude-start", type=float, default=0.50)
    group.add_argument("--rank-magnitude-scale", type=float, default=1.50)
    group.add_argument("--rank-max-pair-weight", type=float, default=2.00)

def composite_loss_from_args(args: argparse.Namespace) -> CompositeStabilityLoss:
    return CompositeStabilityLoss(
        mse_weight=args.mse_weight,
        rank_weight=args.rank_weight,
        stabilizing_weight=args.stabilizing_weight,
        destabilizing_weight=args.destabilizing_weight,
        magnitude_weight=args.magnitude_weight,
        magnitude_start=args.magnitude_start,
        magnitude_scale=args.magnitude_scale,
        max_label_weight=args.max_label_weight,
        rank_margin=args.rank_margin,
        rank_temperature=args.rank_temperature,
        max_rank_items=args.max_rank_items,
        rank_magnitude_weight=args.rank_magnitude_weight,
        rank_magnitude_start=args.rank_magnitude_start,
        rank_magnitude_scale=args.rank_magnitude_scale,
        rank_max_pair_weight=args.rank_max_pair_weight,
    )


LOSS_CLI_ARG_NAMES = (
    "mse_weight",
    "rank_weight",
    "stabilizing_weight",
    "destabilizing_weight",
    "magnitude_weight",
    "magnitude_start",
    "magnitude_scale",
    "max_label_weight",
    "rank_margin",
    "rank_temperature",
    "max_rank_items",
    "rank_magnitude_weight",
    "rank_magnitude_start",
    "rank_magnitude_scale",
    "rank_max_pair_weight",
)


def composite_loss_cli_args(args: argparse.Namespace) -> list[str]:
    cli_args: list[str] = []
    for name in LOSS_CLI_ARG_NAMES:
        value = getattr(args, name)
        cli_args.extend([f"--{name.replace('_', '-')}", str(value)])
    return cli_args


# ---------------------------------------------------------------------------
# Mask construction helper
# ---------------------------------------------------------------------------

def build_wt_mask(
    wt_sequence: str,
    L: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Build a mask that zeros out the wild-type column at every position.

    This is the *base* mask that enforces M(i, a_wt) = 0.  The training
    pipeline should further AND this with the data-availability mask
    (which entries actually have experimental ΔΔG labels).

    Parameters
    ----------
    wt_sequence : str
        Wild-type sequence, length *L*.
    L : int
        Expected sequence length (must equal ``len(wt_sequence)``).
    device : torch.device or str
        Device for the returned tensor.

    Returns
    -------
    mask : torch.Tensor
        Shape ``(L, 20)``.  All ones except the WT column at each position.
    """
    if len(wt_sequence) != L:
        raise ValueError(
            f"WT sequence length ({len(wt_sequence)}) != L ({L})."
        )
    mask = torch.ones(L, NUM_AMINO_ACIDS, device=device)
    for i, aa in enumerate(wt_sequence):
        if aa in AA_TO_INDEX:
            mask[i, AA_TO_INDEX[aa]] = 0.0
    return mask
