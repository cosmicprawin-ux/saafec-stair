# SAAFEC-STAIR

SAAFEC-STAIR is a research inference package for predicting
mutation-induced protein folding free-energy changes, reported as ΔΔG, from
wild-type protein structures. The method combines SaProt structure-aware
representations, ProteinMPNN residue logits, a three-seed single-mutation
ensemble, and a contact-gated double-mutation head.

SAAFEC-STAIR defines the change in folding free energy upon mutation as
`ΔΔG = ΔG(mutant) - ΔG(wild type)`, where both ΔG terms are folding free
energies. Values are reported in kcal/mol: `ΔΔG > 0` indicates a destabilizing
substitution, whereas `ΔΔG < 0` indicates a stabilizing substitution.

The repository contains the SAAFEC-STAIR inference code, setup documentation,
quickstart inputs, reproducible worked examples, reusable input templates, and
larger prediction examples. Required external tools, model weights, and
SAAFEC-STAIR checkpoints are restored separately during setup.

## Supported Inference Tasks

- `MODE=single`: full `L x 20` single-mutant ΔΔG matrices, specified-mutation
  tables, and HTML heatmaps.
- `MODE=double`: predictions for listed double mutations, with implicit
  single-mutation priors.
- CPU and CUDA execution through the same scheduler-independent runner.

## Installation

```bash
git clone https://github.com/cosmicprawin-ux/saafec-stair.git
cd saafec-stair
conda env create -f environment/conda.yml
conda activate SAAFEC_STAIR_inference
```

An already-active virtual environment created with `venv` is also supported;
see [Installation](docs/INSTALLATION.md).

Prepare ignored asset directories:

```bash
bash scripts/setup/prepare_asset_directories.sh
```

Restore the required tools, models, upstream ProteinMPNN utility module, and
SAAFEC-STAIR inference checkpoints by following:

- [Installation](docs/INSTALLATION.md)
- [External assets](docs/EXTERNAL_ASSETS.md)
- [SAAFEC-STAIR inference checkpoints](docs/CHECKPOINTS.md)

Verify each inference mode:

```bash
python scripts/tools/check_installation.py --mode single
python scripts/tools/check_installation.py --mode double
```

## Quick Start Examples

Run the bundled single-mutation example:

```bash
MODE=single RUN_NAME=single_demo bash run_saafec_stair_inference.sh
```

Run the bundled double-mutation example:

```bash
MODE=double RUN_NAME=double_demo bash run_saafec_stair_inference.sh
```

Outputs are written under `output/<run-name>/` by default.

For compact reproducibility checks against selected reference CSVs, use the
[worked examples](examples/worked_examples/README.md).

For a single structure:

```bash
MODE=single \
INPUT_PDB=/path/to/protein.pdb \
CHAIN=A \
RUN_NAME=my_protein \
bash run_saafec_stair_inference.sh
```

For table-based inference:

```bash
MODE=single \
INPUT_TABLE=/path/to/single_mutations.txt \
SPECIFIED_MUTATION_TABLE=/path/to/single_mutations.txt \
PDB_DIR=/path/to/pdbs \
RUN_NAME=my_single_batch \
bash run_saafec_stair_inference.sh
```

`INPUT_TABLE` selects the proteins for full `L x 20` prediction matrices.
Set `SPECIFIED_MUTATION_TABLE` when a separate table containing only the listed
single-mutation predictions is also wanted.

```bash
MODE=double \
INPUT_TABLE=/path/to/double_mutations.txt \
PDB_DIR=/path/to/pdbs \
RUN_NAME=my_double_batch \
bash run_saafec_stair_inference.sh
```

Set `CACHE_DEVICE=cuda` and `INFERENCE_DEVICE=cuda` on a CUDA-capable system.
Both default to `cpu`.

## Documentation

- [Repository architecture](docs/ARCHITECTURE.md)
- [Methods and citations](docs/METHODS_AND_CITATIONS.md)
- [Third-party licenses and notices](docs/THIRD_PARTY_LICENSES_AND_NOTICES.md)
- [Examples overview](examples/README.md)
- [Worked examples](examples/worked_examples/README.md)
- [Input templates](examples/input_templates/README.md)
- [Larger example datasets](examples/example_datasets/README.md)

## Citation

Please cite the **SAAFEC-STAIR paper (citation forthcoming)** and the external
methods listed in [Methods and citations](docs/METHODS_AND_CITATIONS.md).

## License

The SAAFEC-STAIR source code distributed in this repository is licensed under
the [MIT License](LICENSE). External tools, source modules, model weights, and
other third-party assets are not included in this repository and remain subject
to their own licenses and terms.
