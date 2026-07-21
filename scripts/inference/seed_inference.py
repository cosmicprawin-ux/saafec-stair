"""Single-seed full-matrix model inference."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def predict_batch(
    batch: dict[str, Any],
    head: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    saprot_embeddings = batch["saprot_embeddings"].to(device)
    proteinmpnn_logits = batch["proteinmpnn_logits"].to(device)
    proteinmpnn_mask = batch.get("proteinmpnn_masks")
    if proteinmpnn_mask is not None:
        proteinmpnn_mask = proteinmpnn_mask.to(device)
    ca_coordinates = batch.get("ca_coordinates")
    if ca_coordinates is not None:
        ca_coordinates = ca_coordinates.to(device)
    return head(
        saprot_embeddings,
        proteinmpnn_logits,
        batch["sequences"],
        lengths=batch["lengths"],
        ca_coordinates=ca_coordinates,
        proteinmpnn_mask=proteinmpnn_mask,
    )


@torch.no_grad()
def run_inference(
    loader: DataLoader,
    head: torch.nn.Module,
    device: torch.device,
    *,
    full_prediction_matrices: dict[str, dict[str, Any]],
) -> None:
    head.eval()

    for batch in loader:
        names = batch["names"]
        sequences = batch["sequences"]
        ddg_pred = predict_batch(batch, head, device)

        for i in range(ddg_pred.shape[0]):
            length = int(batch["lengths"][i])
            full_prediction_matrices[names[i]] = {
                "protein_name": names[i],
                "sequence": sequences[i],
                "predicted_ddg_matrix": ddg_pred[i, :length].detach().cpu(),
            }

    return None
