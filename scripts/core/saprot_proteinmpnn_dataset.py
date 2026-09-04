#!/usr/bin/env python3
"""Paired SaProt embedding + ProteinMPNN logits dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch.utils.data import Dataset

from core.mutation_dataset import MutationDataset, ProteinSample
from core.pipeline_config import resolve_output_path


class SaProtProteinMPNNSample(NamedTuple):
    saprot_embeddings: torch.Tensor
    proteinmpnn_logits: torch.Tensor
    proteinmpnn_mask: torch.Tensor
    ca_coordinates: torch.Tensor
    mutation_mask: torch.Tensor
    wt_sequence: str
    protein_name: str
    mutation_resolution: dict[str, Any]


def resolve_proteinmpnn_cache_root(cache_dir: str | Path) -> Path:
    root = resolve_output_path(cache_dir)
    by_protein = root / "by_protein"
    return by_protein if by_protein.is_dir() else root


def _load_payload(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class ProteinMPNNLogitsCache:
    """Load per-protein ``L x 20`` ProteinMPNN logits."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = resolve_proteinmpnn_cache_root(cache_dir)
        if not self.root.is_dir():
            raise FileNotFoundError(f"ProteinMPNN cache directory does not exist: {self.root}")

    def _candidate_paths(self, protein_name: str) -> list[Path]:
        protein_dir = self.root / protein_name
        return [
            protein_dir / "proteinmpnn_logits.pt",
            protein_dir / "outputs" / "proteinmpnn_logits.pt",
            protein_dir / "outputs" / "proteinmpnn" / "proteinmpnn_logits.pt",
            self.root / f"{protein_name}.pt",
        ]

    @staticmethod
    def _mask_candidates(path: Path) -> list[Path]:
        return [
            path.with_name("proteinmpnn_mask.pt"),
            path.parent / "proteinmpnn_coordinate_mask.pt",
        ]

    def load(self, protein_name: str, *, expected_length: int, wt_sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        for path in self._candidate_paths(protein_name):
            if not path.is_file():
                continue
            payload = _load_payload(path)
            mask = None
            if torch.is_tensor(payload):
                logits = payload
            elif isinstance(payload, dict):
                logits = payload.get("proteinmpnn_logits")
                if logits is None:
                    logits = payload.get("logits")
                mask = payload.get("proteinmpnn_mask")
                if mask is None:
                    mask = payload.get("coordinate_mask")
                cached_sequence = payload.get("sequence")
                if cached_sequence is not None and str(cached_sequence) != wt_sequence:
                    raise ValueError(
                        f"ProteinMPNN cache sequence mismatch for {protein_name}: "
                        f"{len(str(cached_sequence))} cached residues vs {len(wt_sequence)} dataset residues."
                    )
            else:
                logits = None
            if not torch.is_tensor(logits):
                raise TypeError(f"ProteinMPNN cache payload lacks logits tensor: {path}")
            logits = logits.detach().float().cpu()
            if logits.ndim != 2 or logits.shape[1] != 20:
                raise ValueError(f"Expected ProteinMPNN logits shape (L, 20), got {tuple(logits.shape)} at {path}")
            if logits.shape[0] != expected_length:
                raise ValueError(
                    f"ProteinMPNN length mismatch for {protein_name}: logits L={logits.shape[0]} "
                    f"but SaProt/dataset L={expected_length}."
                )
            if mask is None:
                for mask_path in self._mask_candidates(path):
                    if mask_path.is_file():
                        mask = _load_payload(mask_path)
                        break
            if mask is None:
                mask_tensor = torch.ones(expected_length, dtype=torch.bool)
            elif torch.is_tensor(mask):
                mask_tensor = mask.detach().cpu().bool().flatten()
            else:
                raise TypeError(f"ProteinMPNN mask payload is not a tensor for {protein_name}: {path}")
            if mask_tensor.shape[0] != expected_length:
                raise ValueError(
                    f"ProteinMPNN mask length mismatch for {protein_name}: mask L={mask_tensor.shape[0]} "
                    f"but logits/dataset L={expected_length}."
                )
            return logits, mask_tensor
        tried = "\n  - ".join(str(path) for path in self._candidate_paths(protein_name))
        raise FileNotFoundError(f"No ProteinMPNN logits cache found for {protein_name}. Tried:\n  - {tried}")


class SaProtProteinMPNNDataset(Dataset[SaProtProteinMPNNSample]):
    """Wrap ``MutationDataset`` with a paired ProteinMPNN logits cache."""

    def __init__(
        self,
        *,
        mutations_table: str | Path,
        saprot_embeddings_dir: str | Path,
        proteinmpnn_cache_dir: str | Path,
        table_sheet_name: str = "refined_sorted_clean",
    ) -> None:
        self.base = MutationDataset(
            mutations_table=mutations_table,
            embeddings_dir=saprot_embeddings_dir,
            table_sheet_name=table_sheet_name,
        )
        self.mpnn_cache = ProteinMPNNLogitsCache(proteinmpnn_cache_dir)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> SaProtProteinMPNNSample:
        sample: ProteinSample = self.base[idx]
        proteinmpnn_logits, proteinmpnn_mask = self.mpnn_cache.load(
            sample.protein_name,
            expected_length=int(sample.embedding.shape[0]),
            wt_sequence=sample.wt_sequence,
        )
        return SaProtProteinMPNNSample(
            saprot_embeddings=sample.embedding,
            proteinmpnn_logits=proteinmpnn_logits,
            proteinmpnn_mask=proteinmpnn_mask,
            ca_coordinates=sample.ca_coordinates,
            mutation_mask=sample.mutation_mask,
            wt_sequence=sample.wt_sequence,
            protein_name=sample.protein_name,
            mutation_resolution=sample.mutation_resolution,
        )


def saprot_proteinmpnn_collate_fn(samples: list[SaProtProteinMPNNSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    batch_size = len(samples)
    max_len = max(sample.saprot_embeddings.shape[0] for sample in samples)
    d_saprot = int(samples[0].saprot_embeddings.shape[1])

    saprot_embeddings = torch.zeros(batch_size, max_len, d_saprot, dtype=torch.float32)
    proteinmpnn_logits = torch.zeros(batch_size, max_len, 20, dtype=torch.float32)
    proteinmpnn_masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    ca_coordinates = torch.full((batch_size, max_len, 3), float("nan"), dtype=torch.float32)
    mutation_masks = torch.zeros(batch_size, max_len, 20, dtype=torch.float32)
    lengths: list[int] = []
    sequences: list[str] = []
    names: list[str] = []
    mutation_resolutions: list[dict[str, Any]] = []

    for i, sample in enumerate(samples):
        length = int(sample.saprot_embeddings.shape[0])
        lengths.append(length)
        sequences.append(sample.wt_sequence)
        names.append(sample.protein_name)
        mutation_resolutions.append(sample.mutation_resolution)
        saprot_embeddings[i, :length] = sample.saprot_embeddings.float()
        proteinmpnn_logits[i, :length] = sample.proteinmpnn_logits.float()
        proteinmpnn_masks[i, :length] = sample.proteinmpnn_mask.bool()
        ca_coordinates[i, :length] = sample.ca_coordinates.float()
        mutation_masks[i, :length] = sample.mutation_mask.float()

    return {
        "saprot_embeddings": saprot_embeddings,
        "proteinmpnn_logits": proteinmpnn_logits,
        "proteinmpnn_masks": proteinmpnn_masks,
        "ca_coordinates": ca_coordinates,
        "mutation_masks": mutation_masks,
        "lengths": lengths,
        "sequences": sequences,
        "names": names,
        "mutation_resolutions": mutation_resolutions,
    }
