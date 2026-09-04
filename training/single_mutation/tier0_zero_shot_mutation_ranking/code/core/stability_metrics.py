#!/usr/bin/env python3
"""
stability_metrics.py
--------------------
Evaluation metrics for the v3 stability predictor workflow.

All metrics from the evaluation protocol:

    Metric                 Purpose                                    Target
    ──────────────────     ──────────────────────────────────────     ──────
    Per-protein Spearman   Ranking accuracy within each protein       > 0.75
    Per-protein Pearson    Linear correlation within each protein     > 0.75
    Global Spearman        Overall ranking across all mutations       > 0.80
    Global Pearson         Overall linear correlation                 > 0.80
    RMSE (kcal/mol)        Absolute prediction accuracy               < 0.80
    MAE (kcal/mol)         Robust absolute error                      < 0.60
    Stabilizing PPV        Precision for DDG < -0.5 kcal/mol          > 0.50

Per-protein summary statistics use the median across proteins. In this branch,
checkpoint selection uses the global pooled Pearson across all mutations. WT subtraction
still enforces DDG(A->B) = -DDG(B->A) by construction, but this MegaScale-only
workflow does not report a separate Ssym metric because the dataset does not
provide inverse mutation pairs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from scipy import stats


# ---------------------------------------------------------------------------
# Single source of truth for the val_mse / val_rmse pair
# ---------------------------------------------------------------------------

def derive_val_mse_rmse(val_mse: float) -> tuple[float, float]:
    """
    Return ``(val_mse, val_rmse)`` where ``val_rmse == sqrt(val_mse)`` by construction.

    Use this everywhere both fields are written into the same dict / package /
    CSV row, so the two values cannot drift apart. NaN- and negative-MSE-safe.

    Notes
    -----
    The training-time ``val_mse`` is the per-protein-averaged masked MSE returned
    by ``MaskedMSELoss``. The pooled-across-mutations RMSE produced by
    ``EvalResult.rmse`` is a *different* metric and must NOT be substituted in
    here — it is reported separately under the EvalResult dict's own ``rmse``
    field.
    """
    if val_mse != val_mse:  # NaN
        return val_mse, float("nan")
    if val_mse < 0:
        return val_mse, float("nan")
    return val_mse, math.sqrt(val_mse)


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def spearman_rho(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Spearman rank correlation. Returns NaN if fewer than 3 data points."""
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    if len(pred_np) < 3:
        return float("nan")
    rho, _ = stats.spearmanr(pred_np, true_np)
    return float(rho)


def pearson_r(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Pearson correlation coefficient. Returns NaN if fewer than 3 data points."""
    pred_np = pred.detach().cpu().numpy()
    true_np = true.detach().cpu().numpy()
    if len(pred_np) < 3:
        return float("nan")
    r, _ = stats.pearsonr(pred_np, true_np)
    return float(r)


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Root mean squared error (kcal/mol)."""
    return float(((pred - true) ** 2).mean().sqrt().item())


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean absolute error (kcal/mol)."""
    return float((pred - true).abs().mean().item())


def stabilizing_ppv(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = -0.5,
) -> float:
    """
    Positive predictive value for stabilising mutations.

    Stabilising = DDG < threshold (default -0.5 kcal/mol).
    PPV = TP / (TP + FP), where positive = predicted stabilising.

    Returns NaN if no predicted positives.
    """
    pred_positive = pred < threshold
    true_positive = true < threshold
    tp = (pred_positive & true_positive).sum().item()
    fp = (pred_positive & ~true_positive).sum().item()
    if tp + fp == 0:
        return float("nan")
    return tp / (tp + fp)


# ---------------------------------------------------------------------------
# Per-protein summary helpers
# ---------------------------------------------------------------------------

def median_ignore_nan(values: list[float]) -> float:
    """Median of the non-NaN values, or NaN if none remain."""
    clean = sorted(v for v in values if v == v)
    if not clean:
        return float("nan")
    mid = len(clean) // 2
    if len(clean) % 2 == 1:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def mean_ignore_nan(values: list[float]) -> float:
    """Mean of the non-NaN values, or NaN if none remain."""
    clean = [v for v in values if v == v]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


# ---------------------------------------------------------------------------
# Aggregated evaluation result
# ---------------------------------------------------------------------------

@dataclass
class PerProteinResult:
    """Metrics for a single protein."""
    protein_name: str = ""
    spearman: float = float("nan")
    pearson: float = float("nan")
    rmse: float = float("nan")
    mae: float = float("nan")
    n_mutations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "protein_name": self.protein_name,
            "spearman": self.spearman,
            "pearson": self.pearson,
            "rmse": self.rmse,
            "mae": self.mae,
            "n_mutations": self.n_mutations,
        }


@dataclass
class EvalResult:
    """Container for all evaluation metrics from a single eval pass."""
    median_per_protein_spearman: float = float("nan")
    median_per_protein_pearson: float = float("nan")
    global_spearman: float = float("nan")
    global_pearson: float = float("nan")
    rmse: float = float("nan")
    mae: float = float("nan")
    stabilizing_ppv: float = float("nan")
    dev_score: float = float("nan")
    megascale_validation_pearson: float = float("nan")
    external_homolog_validation_mean_pearson: float = float("nan")
    dev_score_weight_megascale: float = 0.3
    dev_score_weight_external_homolog: float = 0.7
    source_database_pearsons: dict[str, float] = field(default_factory=dict)
    n_proteins: int = 0
    n_mutations: int = 0
    per_protein_details: list[PerProteinResult] = field(default_factory=list)

    @property
    def per_protein_spearman(self) -> float:
        """Backward-compatible alias for median-per-protein Spearman."""
        return self.median_per_protein_spearman

    @property
    def per_protein_pearson(self) -> float:
        """Backward-compatible alias for median-per-protein Pearson."""
        return self.median_per_protein_pearson

    def to_dict(self) -> dict[str, Any]:
        return {
            "median_per_protein_spearman": self.median_per_protein_spearman,
            "median_per_protein_pearson": self.median_per_protein_pearson,
            "per_protein_spearman": self.per_protein_spearman,
            "per_protein_pearson": self.per_protein_pearson,
            "global_spearman": self.global_spearman,
            "global_pearson": self.global_pearson,
            "rmse": self.rmse,
            "mae": self.mae,
            "stabilizing_ppv": self.stabilizing_ppv,
            "dev_score": self.dev_score,
            "megascale_validation_pearson": self.megascale_validation_pearson,
            "external_homolog_validation_mean_pearson": self.external_homolog_validation_mean_pearson,
            "dev_score_weight_megascale": self.dev_score_weight_megascale,
            "dev_score_weight_external_homolog": self.dev_score_weight_external_homolog,
            "source_database_pearsons": dict(self.source_database_pearsons),
            "n_proteins": self.n_proteins,
            "n_mutations": self.n_mutations,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Like to_dict() but includes per-protein breakdown."""
        d = self.to_dict()
        d["per_protein_details"] = [p.to_dict() for p in self.per_protein_details]
        return d

    def summary_str(self) -> str:
        return (
            f"proteins={self.n_proteins}  mutations={self.n_mutations}  "
            f"val_g_spearman={self.global_spearman:.4f}  "
            f"val_g_pearson={self.global_pearson:.4f}  "
            f"dev_score={self.dev_score:.4f}  "
            f"val_megascale_pearson={self.megascale_validation_pearson:.4f}  "
            f"val_external_mean_pearson={self.external_homolog_validation_mean_pearson:.4f}  "
            f"val_median_pp_spearman={self.median_per_protein_spearman:.4f}  "
            f"val_median_pp_pearson={self.median_per_protein_pearson:.4f}  "
            f"rmse={self.rmse:.4f}  mae={self.mae:.4f}  "
            f"stab_ppv={self.stabilizing_ppv:.4f}"
        )


# ---------------------------------------------------------------------------
# Full evaluation pass
# ---------------------------------------------------------------------------

def evaluate(
    ddg_pred_list: list[torch.Tensor],
    ddg_true_list: list[torch.Tensor],
    mask_list: list[torch.Tensor],
    protein_names: list[str] | None = None,
    source_databases: list[str] | None = None,
    dev_score_weight_megascale: float = 0.3,
    dev_score_weight_external_homolog: float = 0.7,
) -> EvalResult:
    """
    Compute all metrics from lists of per-protein DDG predictions, targets, and masks.

    Parameters
    ----------
    ddg_pred_list : list of (L_p, 20) tensors — predicted DDG.
    ddg_true_list : list of (L_p, 20) tensors — target DDG.
    mask_list     : list of (L_p, 20) tensors — binary masks.
    protein_names : optional list of protein names (same order as the tensors).
    source_databases : optional source labels for dev-score component metrics.
    dev_score_weight_megascale : MegaScale component weight for the validation score.
    dev_score_weight_external_homolog : external homolog component weight for DAV.

    Returns
    -------
    EvalResult with all metrics populated, including per-protein breakdown.
    """
    per_protein_preds: list[torch.Tensor] = []
    per_protein_trues: list[torch.Tensor] = []
    per_protein_names: list[str] = []
    per_protein_sources: list[str] = []
    all_preds: list[torch.Tensor] = []
    all_trues: list[torch.Tensor] = []

    for i, (pred, true, m) in enumerate(zip(ddg_pred_list, ddg_true_list, mask_list)):
        # Extract masked entries for this protein
        idx = m.bool().flatten()
        p_flat = pred.flatten()[idx]
        t_flat = true.flatten()[idx]

        if p_flat.numel() == 0:
            continue

        per_protein_preds.append(p_flat)
        per_protein_trues.append(t_flat)
        per_protein_names.append(
            protein_names[i] if protein_names is not None else f"protein_{i}"
        )
        per_protein_sources.append(
            source_databases[i] if source_databases is not None else "unknown"
        )
        all_preds.append(p_flat)
        all_trues.append(t_flat)

    if not all_preds:
        return EvalResult()

    # Concatenate for global metrics
    global_pred = torch.cat(all_preds)
    global_true = torch.cat(all_trues)

    # Per-protein details
    details: list[PerProteinResult] = []
    for name, p, t in zip(per_protein_names, per_protein_preds, per_protein_trues):
        details.append(PerProteinResult(
            protein_name=name,
            spearman=spearman_rho(p, t),
            pearson=pearson_r(p, t),
            rmse=rmse(p, t),
            mae=mae(p, t),
            n_mutations=int(p.numel()),
        ))

    per_protein_spearmans = [detail.spearman for detail in details]
    per_protein_pearsons = [detail.pearson for detail in details]
    source_preds: dict[str, list[torch.Tensor]] = {}
    source_trues: dict[str, list[torch.Tensor]] = {}
    for source, p, t in zip(per_protein_sources, per_protein_preds, per_protein_trues):
        source_preds.setdefault(source, []).append(p)
        source_trues.setdefault(source, []).append(t)

    source_database_pearsons = {
        source: pearson_r(torch.cat(source_preds[source]), torch.cat(source_trues[source]))
        for source in sorted(source_preds)
    }
    megascale_validation_pearson = source_database_pearsons.get(
        "megascale_validation",
        float("nan"),
    )
    external_homolog_validation_mean_pearson = mean_ignore_nan(
        [
            pearson
            for source, pearson in source_database_pearsons.items()
            if source != "megascale_validation"
        ]
    )
    if (
        megascale_validation_pearson == megascale_validation_pearson
        and external_homolog_validation_mean_pearson == external_homolog_validation_mean_pearson
    ):
        dev_score = (
            dev_score_weight_megascale * megascale_validation_pearson
            + dev_score_weight_external_homolog * external_homolog_validation_mean_pearson
        )
    else:
        dev_score = pearson_r(global_pred, global_true)

    result = EvalResult(
        median_per_protein_spearman=median_ignore_nan(per_protein_spearmans),
        median_per_protein_pearson=median_ignore_nan(per_protein_pearsons),
        global_spearman=spearman_rho(global_pred, global_true),
        global_pearson=pearson_r(global_pred, global_true),
        rmse=rmse(global_pred, global_true),
        mae=mae(global_pred, global_true),
        stabilizing_ppv=stabilizing_ppv(global_pred, global_true),
        dev_score=dev_score,
        megascale_validation_pearson=megascale_validation_pearson,
        external_homolog_validation_mean_pearson=external_homolog_validation_mean_pearson,
        dev_score_weight_megascale=dev_score_weight_megascale,
        dev_score_weight_external_homolog=dev_score_weight_external_homolog,
        source_database_pearsons=source_database_pearsons,
        n_proteins=len(per_protein_preds),
        n_mutations=int(global_pred.numel()),
        per_protein_details=details,
    )

    return result
