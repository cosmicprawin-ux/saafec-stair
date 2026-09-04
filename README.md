# SAAFEC-STAIR

SAAFEC-STAIR predicts mutation-induced changes in protein folding free energy
(ΔΔG) from wild-type protein structures. It combines SaProt representations,
ProteinMPNN residue logits, a three-seed single-mutation ensemble, and a
contact-gated double-mutation head.

SAAFEC-STAIR defines the change in folding free energy upon mutation as
`ΔΔG = ΔG(mutant) - ΔG(wild type)`, where both ΔG terms are folding free
energies. Values are reported in kcal/mol: `ΔΔG > 0` indicates a destabilizing
substitution, whereas `ΔΔG < 0` indicates a stabilizing substitution.

This repository includes inference code, four trained checkpoints, example
inputs and outputs, selected training workflows, and the data reported in the
manuscript. SaProt and ProteinMPNN weights, Foldseek, and the required
ProteinMPNN utility module are installed separately.

## Repository Contents

| Folder | Contents |
| --- | --- |
| [docs](docs/) | Installation, external assets, checkpoint setup, and method documentation. |
| [environment](environment/) | Inference environment and Python dependencies. |
| [scripts](scripts/) | Inference, feature generation, visualization, and setup code. |
| [inference_checkpoints](inference_checkpoints/README.md) | The four trained SAAFEC-STAIR checkpoints, with checksums. |
| [assets](assets/) | Local destinations for external dependencies and runtime checkpoint copies. |
| [examples](examples/README.md) | Quickstart inputs and worked examples. |
| [tests](tests/) | Automated validation tests. |
| [training](training/README.md) | Training workflows for the reported models. |
| [data_and_benchmarks](data_and_benchmarks/README.md) | Excel datasets, predictions, metrics, and filtering audits. |

SAAFEC-STAIR checkpoints and benchmark workbooks are included in this repository.
SaProt and ProteinMPNN weights, Foldseek, and the pinned ThermoMPNN utility
module are obtained from their [official sources](docs/EXTERNAL_ASSETS.md).

## Architecture

![SAAFEC-STAIR single- and double-mutation architecture](docs/images/saafec_stair_architecture_1500dpi.png)

Panel **A** shows the single-mutation model, which combines SaProt features with
ProteinMPNN residue preferences and averages three predictors to produce an
`L x 20` ΔΔG matrix. Panel **B** shows the double-mutation model, which adds
calibration and interaction corrections to the two single-site predictions.

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

Install the external tools, backbone models, and upstream ProteinMPNN utility
module, then configure the bundled SAAFEC-STAIR checkpoints by following:

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
Tabular prediction outputs are CSV-only. Their top-left cell reports
`Unit(DDG)=kcal/mol`, followed by the column-header row. Final displayed DDG
predictions are written to two decimal places; internal calculations retain
full precision.

Use the [worked examples](examples/worked_examples/README.md) to compare an
installation with the included outputs.

### Interactive single-mutation output

![Interactive SAAFEC-STAIR single-mutation prediction interface](docs/images/interactive_single_mutation_prediction_interface_1500dpi.png)

The interface links the complete ΔΔG matrix to a three-dimensional structure
view, the selected mutant sequence, residue-specific substitution values, and
summary counts across predicted stability-effect classes.

### Choosing single-mutation inputs

Use one of the following input patterns:

| Goal | Supply | Result |
| --- | --- | --- |
| Profile one structure | `INPUT_PDB` and `CHAIN` | A complete `L x 20` matrix and heatmap |
| Profile one structure and highlight selected substitutions | `INPUT_PDB`, `CHAIN`, and `SINGLE_MUTATION_LIST` | The complete matrix and heatmap, plus a CSV containing the listed substitutions |
| Process one or more structures | `PDB_DIR` and `SINGLE_MUTATION_LIST` | A complete matrix and heatmap for every referenced PDB/chain, plus one combined CSV containing the listed substitutions |

For one structure, no mutation list is required:

```bash
MODE=single \
INPUT_PDB=/path/to/protein.pdb \
CHAIN=A \
RUN_NAME=my_protein \
bash run_saafec_stair_inference.sh
```

Optionally add a single-mutation list:

```bash
MODE=single \
INPUT_PDB=/path/to/protein.pdb \
CHAIN=A \
SINGLE_MUTATION_LIST=/path/to/single_mutation_list.txt \
RUN_NAME=my_single_batch \
bash run_saafec_stair_inference.sh
```

In this one-structure form, every mutation-list row must identify the supplied
PDB and selected chain. Providing the list does not replace the complete
profile; it adds the compact specified-mutation CSV.

For one or more structures in a directory:

```bash
MODE=single \
SINGLE_MUTATION_LIST=/path/to/single_mutation_list.txt \
PDB_DIR=/path/to/pdbs \
RUN_NAME=my_single_batch \
bash run_saafec_stair_inference.sh
```

The list uses `pdb`, `chain`, and `mut`, with one row per requested
substitution. The `pdb` value is the matching filename without the `.pdb`
extension, and `chain` is always supplied separately. A compact value such as
`A10V` means wild-type amino acid A, residue 10, mutant amino acid V. Multiple
rows may refer to the same PDB/chain or to different PDB/chain inputs.

Mutation-list inputs are whitespace-separated `.txt` files. Columns may be
separated by spaces or tabs; the bundled lists use tabs. Lines beginning with
`#` are optional explanatory comments, and comments and blank lines are ignored
by the parser.

Double-mutation inference requires a mutation list:

```bash
MODE=double \
DOUBLE_MUTATION_LIST=/path/to/double_mutation_list.txt \
PDB_DIR=/path/to/pdbs \
RUN_NAME=my_double_batch \
bash run_saafec_stair_inference.sh
```

Double-mutation lists use `pdb`, `chain`, `mut_1`, and `mut_2`. Both mutation
columns use the same wild-type amino acid, residue number, mutant amino acid
order.

Set `CACHE_DEVICE=cuda` and `INFERENCE_DEVICE=cuda` on a CUDA-capable system.
Both default to `cpu`.

## Documentation

- [Repository architecture](docs/ARCHITECTURE.md)
- [Methods and citations](docs/METHODS_AND_CITATIONS.md)
- [Third-party licenses and notices](docs/THIRD_PARTY_LICENSES_AND_NOTICES.md)
- [Examples overview](examples/README.md)
- [Worked examples](examples/worked_examples/README.md)

## Training Workflows

[`training/`](training/README.md) contains the SaProt Tier 0 and Tier 1
workflows, the three-seed SaProt–ProteinMPNN Tier 2 workflow, and the
pair-corrected double-mutation workflow. The training code is kept separate
from the inference code in `scripts/`. Required datasets and assets are listed
in [`training/docs/DATA_AND_ASSETS.md`](training/docs/DATA_AND_ASSETS.md).

## Data and Benchmarks

The [`data_and_benchmarks/`](data_and_benchmarks/README.md) folder contains the
datasets, prediction results, metrics, and filtering records used in the
manuscript:

- [Datasets](data_and_benchmarks/01_Datasets/README.md): source data and
  training, validation, and test sets for single and double mutations.
- [Model-tier comparisons](data_and_benchmarks/02_Model_Tier_Comparisons/README.md):
  Tier 0 ranking, Tier 1 supervised prediction, and Tier 2 ProteinMPNN-augmented
  fusion results.
- [Existing-predictor comparisons](data_and_benchmarks/03_Existing_Predictor_Comparisons/README.md):
  benchmark predictions, metrics, and homology/overlap-control results.

Each section includes a short guide to its files, conventions, and sources.

## Citation

Please cite SAAFEC-STAIR using the software citation metadata in
[`CITATION.cff`](CITATION.cff). If a preprint or peer-reviewed article becomes
available, the preferred citation will be updated here.

Citations for the external methods used by the inference pipeline are listed
in [Methods and citations](docs/METHODS_AND_CITATIONS.md).

## License

The SAAFEC-STAIR source code distributed in this repository is licensed under
the [MIT License](LICENSE). External tools, source modules, model weights, and
other third-party assets are not included in this repository and remain subject
to their own licenses and terms.
