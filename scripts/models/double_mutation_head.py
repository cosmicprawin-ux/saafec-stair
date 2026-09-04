#!/usr/bin/env python3
"""Double-mutation residual heads built on top of single-mutant predictions."""
from __future__ import annotations

import torch
import torch.nn as nn

from core.amino_acids import AA_TO_INDEX, NUM_AMINO_ACIDS


class PairResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.act(self.linear(self.norm(x))))


class DoubleMutationInteractionHead(nn.Module):
    """Predict double-mutant DDG from a calibrated single-mutant prior.

    The input single-mutant matrix is expected to come from the finalized
    upstream SaProt + ProteinMPNN intrinsic-fusion model. The raw additive
    single-mutant baseline is kept unchanged for auditing, then this module
    learns two bounded double-mutation-only terms:

    ``DDG(i->a, j->b) = calibrated_single_prior + contact_gate(i,j) * gated_interaction``.

    Pair features are constructed symmetrically, so swapping mutation order does
    not change the prediction. Calibration and interaction heads are initialized
    near zero so the initial model is close to the raw additive single-mutant
    baseline. ProteinMPNN logits are also provided directly as inverse-folding
    pair features, gated near zero at initialization. The interaction residual
    is additionally shrunk by a C-alpha contact prior so distant residue pairs
    stay close to the calibrated additive model unless the learned gate strongly
    supports a correction.
    """

    architecture_name = "double_mutation_contact_gated_calibrated_single_prior_plus_mpnn_residual_v1"

    def __init__(
        self,
        d_model: int,
        d_hidden: int = 512,
        aa_embed_dim: int = 64,
        dropout: float = 0.10,
        num_residual_blocks: int = 2,
        max_scale_delta: float = 0.35,
        max_shift: float = 1.50,
        max_global_scale_delta: float = 0.20,
        max_global_shift: float = 0.75,
        max_interaction: float = 3.00,
        gate_bias_init: float = -2.0,
        mpnn_pair_gate_init: float = -3.0,
        contact_cutoff: float = 10.0,
        contact_temperature: float = 2.0,
        contact_gate_floor: float = 0.12,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_hidden = int(d_hidden)
        self.max_scale_delta = float(max_scale_delta)
        self.max_shift = float(max_shift)
        self.max_global_scale_delta = float(max_global_scale_delta)
        self.max_global_shift = float(max_global_shift)
        self.max_interaction = float(max_interaction)
        self.contact_cutoff = float(contact_cutoff)
        self.contact_temperature = float(contact_temperature)
        self.contact_gate_floor = float(contact_gate_floor)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.residue_proj = nn.Linear(d_model, d_hidden)

        self.aa_embedding = nn.Embedding(NUM_AMINO_ACIDS, aa_embed_dim)
        self.substitution_embedding = nn.Embedding(NUM_AMINO_ACIDS * NUM_AMINO_ACIDS, aa_embed_dim)
        self.site_aa_norm = nn.LayerNorm(aa_embed_dim * 4 + 1)
        self.site_aa_proj = nn.Linear(aa_embed_dim * 4 + 1, d_hidden)

        self.protein_context_norm = nn.LayerNorm(d_model)
        self.protein_context_proj = nn.Linear(d_model, d_hidden)
        self.geometry_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, d_hidden),
            nn.GELU(),
        )
        self.baseline_scalar_norm = nn.LayerNorm(5)
        self.baseline_scalar_proj = nn.Sequential(
            nn.Linear(5, d_hidden),
            nn.GELU(),
        )
        self.proteinmpnn_pair_norm = nn.LayerNorm(16)
        self.proteinmpnn_pair_proj = nn.Sequential(
            nn.Linear(16, d_hidden),
            nn.GELU(),
        )
        self.proteinmpnn_pair_gate = nn.Parameter(torch.tensor(float(mpnn_pair_gate_init)))

        pair_width = d_hidden * 7
        self.pair_norm = nn.LayerNorm(pair_width)
        self.input_proj = nn.Linear(pair_width, d_hidden)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [PairResidualBlock(d_hidden, dropout) for _ in range(num_residual_blocks)]
        )
        self.global_scale_delta = nn.Parameter(torch.zeros(()))
        self.global_shift = nn.Parameter(torch.zeros(()))
        self.calibration_scale_output = nn.Linear(d_hidden, 1)
        self.calibration_shift_output = nn.Linear(d_hidden, 1)
        self.interaction_output = nn.Linear(d_hidden, 1)
        self.gate_output = nn.Linear(d_hidden, 1)
        self._init_output_heads(gate_bias_init=gate_bias_init)

    def _init_output_heads(self, *, gate_bias_init: float) -> None:
        for layer in (self.calibration_scale_output, self.calibration_shift_output, self.interaction_output):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.gate_output.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.gate_output.bias, float(gate_bias_init))

    def _site_hidden(
        self,
        context: torch.Tensor,
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        wt_indices: torch.Tensor,
        mt_indices: torch.Tensor,
    ) -> torch.Tensor:
        residue_hidden = self.residue_proj(context[positions])
        wt_aa = self.aa_embedding(wt_indices)
        mt_aa = self.aa_embedding(mt_indices)
        aa_delta = mt_aa - wt_aa
        substitution_ids = wt_indices * NUM_AMINO_ACIDS + mt_indices
        substitution = self.substitution_embedding(substitution_ids)
        single_values = single_ddg[positions, mt_indices].unsqueeze(-1)
        aa_features = torch.cat([wt_aa, mt_aa, aa_delta, substitution, single_values], dim=-1)
        return residue_hidden + self.site_aa_proj(self.site_aa_norm(aa_features))

    @staticmethod
    def additive_baseline(
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        mt_indices: torch.Tensor,
    ) -> torch.Tensor:
        return (
            single_ddg[positions[:, 0], mt_indices[:, 0]]
            + single_ddg[positions[:, 1], mt_indices[:, 1]]
        )

    def _geometry_features(
        self,
        positions: torch.Tensor,
        ca_coordinates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_sep = (positions[:, 0] - positions[:, 1]).abs().to(torch.float32)
        log_seq_sep = torch.log1p(seq_sep)
        sequence_proximity = 1.0 / (1.0 + seq_sep)
        if ca_coordinates is None:
            ca_distance = torch.full_like(log_seq_sep, 999.0)
            contact_weight = torch.zeros_like(log_seq_sep)
            contact_prior = torch.zeros_like(log_seq_sep)
            valid_f = torch.zeros_like(log_seq_sep)
        else:
            coords = ca_coordinates.to(device=positions.device, dtype=torch.float32)
            c1 = coords[positions[:, 0]]
            c2 = coords[positions[:, 1]]
            valid = torch.isfinite(c1).all(dim=-1) & torch.isfinite(c2).all(dim=-1)
            dist = torch.linalg.norm(torch.where(valid.unsqueeze(-1), c1 - c2, torch.zeros_like(c1)), dim=-1)
            ca_distance = torch.where(valid, dist, torch.full_like(dist, 999.0))
            contact_weight = torch.where(valid, torch.exp(-ca_distance / 8.0), torch.zeros_like(ca_distance))
            contact_prior = torch.where(
                valid,
                torch.sigmoid((self.contact_cutoff - ca_distance) / max(self.contact_temperature, 1e-6)),
                torch.zeros_like(ca_distance),
            )
            valid_f = valid.to(torch.float32)
        scalars = torch.stack(
            [
                log_seq_sep,
                ca_distance.clamp(max=999.0) / 50.0,
                contact_weight,
                contact_prior,
                sequence_proximity,
                valid_f,
            ],
            dim=-1,
        )
        return self.geometry_proj(scalars.to(device=positions.device)), contact_prior.to(device=positions.device)

    def _baseline_scalar_features(
        self,
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        mt_indices: torch.Tensor,
    ) -> torch.Tensor:
        single_1 = single_ddg[positions[:, 0], mt_indices[:, 0]]
        single_2 = single_ddg[positions[:, 1], mt_indices[:, 1]]
        base = single_1 + single_2
        scalars = torch.stack(
            [
                base,
                (single_1 - single_2).abs(),
                single_1 * single_2,
                torch.minimum(single_1, single_2),
                torch.maximum(single_1, single_2),
            ],
            dim=-1,
        )
        return self.baseline_scalar_proj(self.baseline_scalar_norm(scalars))

    def _proteinmpnn_pair_features(
        self,
        proteinmpnn_logits: torch.Tensor | None,
        proteinmpnn_mask: torch.Tensor | None,
        positions: torch.Tensor,
        wt_indices: torch.Tensor,
        mt_indices: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if proteinmpnn_logits is None:
            return torch.zeros(positions.shape[0], self.d_hidden, device=positions.device, dtype=dtype)
        if proteinmpnn_logits.ndim != 2 or proteinmpnn_logits.shape[1] != NUM_AMINO_ACIDS:
            raise ValueError(
                "proteinmpnn_logits must have shape (L, 20), "
                f"got {tuple(proteinmpnn_logits.shape)}"
            )

        logits = proteinmpnn_logits.to(device=positions.device, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1)

        if proteinmpnn_mask is None:
            mask = torch.ones(logits.shape[0], dtype=torch.bool, device=positions.device)
        else:
            mask = proteinmpnn_mask.to(device=positions.device, dtype=torch.bool).flatten()
            if mask.shape[0] != logits.shape[0]:
                raise ValueError(
                    "proteinmpnn_mask length must match logits length, "
                    f"got {mask.shape[0]} vs {logits.shape[0]}"
                )

        pos1 = positions[:, 0]
        pos2 = positions[:, 1]
        valid1 = mask[pos1]
        valid2 = mask[pos2]
        site1 = log_probs[pos1]
        site2 = log_probs[pos2]

        wt1 = site1.gather(1, wt_indices[:, 0:1]).squeeze(-1)
        wt2 = site2.gather(1, wt_indices[:, 1:2]).squeeze(-1)
        mut1 = site1.gather(1, mt_indices[:, 0:1]).squeeze(-1)
        mut2 = site2.gather(1, mt_indices[:, 1:2]).squeeze(-1)
        odds1 = mut1 - wt1
        odds2 = mut2 - wt2
        ent1 = entropy[pos1]
        ent2 = entropy[pos2]

        def keep_valid(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
            return torch.where(valid, values, torch.zeros_like(values))

        wt1 = keep_valid(wt1, valid1)
        wt2 = keep_valid(wt2, valid2)
        mut1 = keep_valid(mut1, valid1)
        mut2 = keep_valid(mut2, valid2)
        odds1 = keep_valid(odds1, valid1)
        odds2 = keep_valid(odds2, valid2)
        ent1 = keep_valid(ent1, valid1)
        ent2 = keep_valid(ent2, valid2)
        valid1_f = valid1.to(torch.float32)
        valid2_f = valid2.to(torch.float32)

        scalars = torch.stack(
            [
                wt1 + wt2,
                (wt1 - wt2).abs(),
                mut1 + mut2,
                (mut1 - mut2).abs(),
                odds1 + odds2,
                (odds1 - odds2).abs(),
                odds1 * odds2,
                torch.minimum(odds1, odds2),
                torch.maximum(odds1, odds2),
                ent1 + ent2,
                (ent1 - ent2).abs(),
                0.5 * (ent1 + ent2),
                ent1 * ent2,
                valid1_f + valid2_f,
                valid1_f * valid2_f,
                torch.maximum(valid1_f, valid2_f),
            ],
            dim=-1,
        )
        hidden = self.proteinmpnn_pair_proj(self.proteinmpnn_pair_norm(scalars.to(dtype=dtype)))
        return hidden * torch.sigmoid(self.proteinmpnn_pair_gate).to(dtype=dtype)

    def _pair_hidden(
        self,
        embeddings: torch.Tensor,
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        wt_indices: torch.Tensor,
        mt_indices: torch.Tensor,
        *,
        ca_coordinates: torch.Tensor | None = None,
        proteinmpnn_logits: torch.Tensor | None = None,
        proteinmpnn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(f"positions must have shape (N, 2), got {tuple(positions.shape)}")
        if wt_indices.shape != positions.shape or mt_indices.shape != positions.shape:
            raise ValueError("wt_indices and mt_indices must match positions shape")
        if embeddings.ndim != 2 or single_ddg.ndim != 2:
            raise ValueError("embeddings and single_ddg must be per-protein matrices")

        device = embeddings.device
        positions = positions.to(device=device, dtype=torch.long)
        wt_indices = wt_indices.to(device=device, dtype=torch.long)
        mt_indices = mt_indices.to(device=device, dtype=torch.long)
        single_ddg = single_ddg.to(device=device, dtype=embeddings.dtype)

        context = self.embedding_norm(embeddings)
        site1 = self._site_hidden(
            context,
            single_ddg,
            positions[:, 0],
            wt_indices[:, 0],
            mt_indices[:, 0],
        )
        site2 = self._site_hidden(
            context,
            single_ddg,
            positions[:, 1],
            wt_indices[:, 1],
            mt_indices[:, 1],
        )

        pooled = self.protein_context_proj(
            self.protein_context_norm(embeddings.mean(dim=0, keepdim=True))
        ).expand(positions.shape[0], -1)
        geometry, contact_prior = self._geometry_features(positions, ca_coordinates)
        geometry = geometry.to(dtype=embeddings.dtype)
        contact_prior = contact_prior.to(dtype=embeddings.dtype)
        baseline_features = self._baseline_scalar_features(single_ddg, positions, mt_indices).to(dtype=embeddings.dtype)
        proteinmpnn_features = self._proteinmpnn_pair_features(
            proteinmpnn_logits,
            proteinmpnn_mask,
            positions,
            wt_indices,
            mt_indices,
            dtype=embeddings.dtype,
        )

        pair_features = torch.cat(
            [
                site1 + site2,
                (site1 - site2).abs(),
                site1 * site2,
                pooled,
                geometry,
                baseline_features,
                proteinmpnn_features,
            ],
            dim=-1,
        )
        x = self.dropout(self.act(self.input_proj(self.pair_norm(pair_features))))
        for block in self.blocks:
            x = block(x)
        base = self.additive_baseline(single_ddg, positions, mt_indices)
        return x, base, contact_prior

    def correction_components(
        self,
        embeddings: torch.Tensor,
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        wt_indices: torch.Tensor,
        mt_indices: torch.Tensor,
        *,
        ca_coordinates: torch.Tensor | None = None,
        proteinmpnn_logits: torch.Tensor | None = None,
        proteinmpnn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw additive prior, calibration delta, interaction delta, contact prior, and residual gate."""
        x, base, contact_prior = self._pair_hidden(
            embeddings,
            single_ddg,
            positions,
            wt_indices,
            mt_indices,
            ca_coordinates=ca_coordinates,
            proteinmpnn_logits=proteinmpnn_logits,
            proteinmpnn_mask=proteinmpnn_mask,
        )
        contextual_scale_delta = self.max_scale_delta * torch.tanh(
            self.calibration_scale_output(x).squeeze(-1)
        )
        contextual_shift = self.max_shift * torch.tanh(self.calibration_shift_output(x).squeeze(-1))
        global_scale_delta = self.max_global_scale_delta * torch.tanh(self.global_scale_delta)
        global_shift = self.max_global_shift * torch.tanh(self.global_shift)
        calibration_delta = base * (global_scale_delta + contextual_scale_delta) + global_shift + contextual_shift
        learned_gate = torch.sigmoid(self.gate_output(x).squeeze(-1))
        contact_multiplier = self.contact_gate_floor + (1.0 - self.contact_gate_floor) * contact_prior
        residual_gate = learned_gate * contact_multiplier
        interaction = self.max_interaction * torch.tanh(self.interaction_output(x).squeeze(-1)) * residual_gate
        return base, calibration_delta, interaction, contact_prior, residual_gate

    def residual(
        self,
        embeddings: torch.Tensor,
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        wt_indices: torch.Tensor,
        mt_indices: torch.Tensor,
        *,
        ca_coordinates: torch.Tensor | None = None,
        proteinmpnn_logits: torch.Tensor | None = None,
        proteinmpnn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _base, calibration_delta, interaction, _contact_prior, _residual_gate = self.correction_components(
            embeddings,
            single_ddg,
            positions,
            wt_indices,
            mt_indices,
            ca_coordinates=ca_coordinates,
            proteinmpnn_logits=proteinmpnn_logits,
            proteinmpnn_mask=proteinmpnn_mask,
        )
        return calibration_delta + interaction

    def forward(
        self,
        embeddings: torch.Tensor,
        single_ddg: torch.Tensor,
        positions: torch.Tensor,
        wt_indices: torch.Tensor,
        mt_indices: torch.Tensor,
        *,
        ca_coordinates: torch.Tensor | None = None,
        proteinmpnn_logits: torch.Tensor | None = None,
        proteinmpnn_mask: torch.Tensor | None = None,
        return_components: bool = False,
        return_detail_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base, calibration_delta, interaction, contact_prior, residual_gate = self.correction_components(
            embeddings,
            single_ddg,
            positions,
            wt_indices,
            mt_indices,
            ca_coordinates=ca_coordinates,
            proteinmpnn_logits=proteinmpnn_logits,
            proteinmpnn_mask=proteinmpnn_mask,
        )
        calibrated_base = base + calibration_delta
        correction = calibration_delta + interaction
        pred = base + correction
        if return_detail_components:
            return pred, base, correction, calibrated_base, calibration_delta, interaction, contact_prior, residual_gate
        if return_components:
            return pred, base, correction
        return pred


def validate_double_mutation_indices(
    wt_sequence: str,
    positions: torch.Tensor,
    wt_indices: torch.Tensor,
) -> None:
    """Raise if workbook WT residues and sequence positions disagree."""
    index_to_aa = {idx: aa for aa, idx in AA_TO_INDEX.items()}
    for row_idx, (pos_pair, wt_pair) in enumerate(zip(positions.tolist(), wt_indices.tolist())):
        for site_idx, (pos, wt_idx) in enumerate(zip(pos_pair, wt_pair), start=1):
            if wt_sequence[pos] != index_to_aa[int(wt_idx)]:
                raise ValueError(
                    f"row {row_idx} site {site_idx}: wt_sequence[{pos + 1}] is "
                    f"{wt_sequence[pos]}, expected {index_to_aa[int(wt_idx)]}"
                )
