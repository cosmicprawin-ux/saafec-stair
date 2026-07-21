# Methods and Citations

## SAAFEC-STAIR

SAAFEC-STAIR paper (citation forthcoming).

Predictions use
`ΔΔG = ΔG(mutant) - ΔG(wild type)`, where both ΔG terms are folding free
energies. Values are reported in kcal/mol: `ΔΔG > 0` indicates a destabilizing
substitution, whereas `ΔΔG < 0` indicates a stabilizing substitution.

SAAFEC-STAIR uses the following external methods during inference. Cite the
relevant works when reporting results produced by this pipeline.

For software and model licensing, plus attribution notes, see
[Third-party licenses and notices](THIRD_PARTY_LICENSES_AND_NOTICES.md).

## SaProt

SaProt provides structure-aware protein language-model representations.

Su, J. et al. "SaProt: Protein Language Modeling with Structure-aware
Vocabulary." *International Conference on Learning Representations* (2024).
Preprint: <https://doi.org/10.1101/2023.10.01.560349>

Code: <https://github.com/westlake-repl/SaProt>

## ProteinMPNN

ProteinMPNN supplies structure-conditioned amino-acid logits used by the
SAAFEC-STAIR fusion head.

Dauparas, J. et al. "Robust deep learning-based protein sequence design using
ProteinMPNN." *Science* 378, 49-56 (2022).
<https://doi.org/10.1126/science.add2187>

Code: <https://github.com/dauparas/ProteinMPNN>

## Foldseek

Foldseek converts structures to the representation consumed by SaProt.

van Kempen, M. et al. "Fast and accurate protein structure search with
Foldseek." *Nature Biotechnology* 42, 243-246 (2024).
<https://doi.org/10.1038/s41587-023-01773-0>

Code: <https://github.com/steineggerlab/foldseek>

## Implementation Attribution

The runtime ProteinMPNN class is loaded from the pinned
`protein_mpnn_utils.py` in ThermoMPNN. This attribution records the
implementation source; SAAFEC-STAIR does not execute ThermoMPNN's stability
predictor.

Dieckhaus, H. et al. "Transfer learning to leverage larger datasets for improved
prediction of protein stability changes." *Proceedings of the National Academy
of Sciences* 121, e2314853121 (2024).
<https://doi.org/10.1073/pnas.2314853121>

ThermoMPNN code: <https://github.com/Kuhlman-Lab/ThermoMPNN>

The ProteinMPNN implementation lineage also builds on graph-based protein
design:

Ingraham, J. et al. "Generative Models for Graph-Based Protein Design."
*Advances in Neural Information Processing Systems* 32 (2019).
<https://proceedings.neurips.cc/paper/2019/hash/f3a4ff4839c56a5f460c88cce3666a2b-Abstract.html>

Code: <https://github.com/jingraham/neurips19-graph-protein-design>

## Scope

This document attributes methods and software used by the inference package.
It intentionally does not describe or attribute datasets or structure
collections.
