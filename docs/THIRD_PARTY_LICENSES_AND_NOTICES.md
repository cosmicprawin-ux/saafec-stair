# Third-Party Licenses and Notices

SAAFEC-STAIR is an inference package that integrates external tools, model
weights, and an upstream utility module. No third-party source code,
executables, or model weights are distributed in this repository. Users restore
those materials separately from official upstream sources and remain
responsible for following the upstream licenses and model terms.

The SAAFEC-STAIR source code distributed in this repository is licensed under
the MIT License. That license does not replace or modify the licenses of
third-party tools, source modules, model weights, or other assets.

The labels below are provided for orientation. Review the license and any model
terms at the exact upstream revision or release you install.

| Component | How SAAFEC-STAIR uses it | Source | License/terms noted for this release |
| --- | --- | --- | --- |
| SaProt 650M PDB | Provides structure-aware protein language-model representations after Foldseek structural-token generation. | Model: <https://huggingface.co/westlake-repl/SaProt_650M_PDB>; code: <https://github.com/westlake-repl/SaProt> | MIT license for the upstream SaProt code/model repository, subject to the exact upstream files and Hugging Face model terms. |
| Foldseek | Generates the structural alphabet representation consumed by SaProt. | Releases/code: <https://github.com/steineggerlab/foldseek> | GPL-3.0 license for the Foldseek executable/source distribution. |
| ProteinMPNN | Supplies the official `v_48_020.pt` structure-conditioned residue-logit checkpoint used as an input feature source. | <https://github.com/dauparas/ProteinMPNN> | MIT license for the upstream ProteinMPNN repository, subject to the exact upstream checkpoint terms. |
| ThermoMPNN `protein_mpnn_utils.py` | Supplies the pinned runtime `ProteinMPNN` class implementation used to load the official ProteinMPNN checkpoint. SAAFEC-STAIR does not run ThermoMPNN as a stability predictor in this pipeline. | <https://github.com/Kuhlman-Lab/ThermoMPNN> | MIT license for the upstream ThermoMPNN repository. |
| graph-protein-design | Historical implementation lineage for ProteinMPNN-style graph-based protein design code. | <https://github.com/jingraham/neurips19-graph-protein-design> | MIT license for the upstream repository. |

## Expected Local Asset Paths

The runtime checks expect the restored assets at these ignored paths unless
environment variables override them:

```text
assets/external/models/SaProt_650M_PDB/
assets/external/bin/foldseek
assets/external/models/proteinmpnn/v_48_020.pt
assets/external/source/ThermoMPNN/protein_mpnn_utils.py
assets/checkpoints/single_mutation/seeds/seed_1337/best_head.pt
assets/checkpoints/single_mutation/seeds/seed_2027/best_head.pt
assets/checkpoints/single_mutation/seeds/seed_3407/best_head.pt
assets/checkpoints/double_mutation/best_double_mutation_package.pt
```

The SAAFEC-STAIR checkpoints are project assets included in
[`inference_checkpoints/`](../inference_checkpoints/README.md). They are
also available from the public model repository:

<https://huggingface.co/multiverse49/SAAFEC-STAIR-inference-checkpoints>

## Citation Pointers

When reporting results from this package, cite SAAFEC-STAIR and the external
methods used in the inference stack. The scholarly references are listed in:

- [Methods and citations](METHODS_AND_CITATIONS.md)
- [External assets](EXTERNAL_ASSETS.md)

In practical terms, reports using the default SAAFEC-STAIR inference path
should credit:

- SaProt for structure-aware PLM representations.
- Foldseek for structural-token generation used by SaProt.
- ProteinMPNN for structure-conditioned residue-logit features.
- ThermoMPNN for the pinned `protein_mpnn_utils.py` implementation used to
  instantiate the ProteinMPNN architecture.
