# SAAFEC-STAIR Training

This directory contains the workflows used for SaProt zero-shot evaluation,
supervised single-mutation training, ProteinMPNN-augmented fusion, and
pair-corrected double-mutation training. Dataset workbooks are in
[`data_and_benchmarks/`](../data_and_benchmarks/README.md), and released
checkpoints are in
[`inference_checkpoints/`](../inference_checkpoints/README.md). Exact
wild-type structures and third-party software and weights must be supplied
separately.

Loss functions, model architecture, optimization settings, random seeds, and
model-selection criteria are defined in the workflow code.

## Workflows

| Directory | Manuscript role | Entrypoint |
|---|---|---|
| `single_mutation/tier0_zero_shot_mutation_ranking/` | Tier 0 zero-shot mutation ranking | `python run.py` |
| `single_mutation/tier1_supervised_mutation_aware_prediction/` | Tier 1 supervised mutation-aware prediction | `python run.py` |
| `single_mutation/tier2_proteinmpnn_augmented_fusion/` | Tier 2 ProteinMPNN-augmented fusion and the final three-seed predictor | `python run.py` |
| `pair_corrected_double_mutation/` | Pair-corrected double-mutation training with the single-mutation ensemble frozen | `python run.py` |

Each workflow folder has a `RUNNING.md` with its run command and a `code/`
directory with the required data, model, training, and evaluation modules.
The inference implementation remains in the root `scripts/` directory.

This package does not include the broader pretrained-model screen,
third-party predictors, dataset-curation utilities, or manuscript assembly.

## Training Order

Tier 2 trains the single-mutation models with seeds 1337, 2027, and 3407. The
double-mutation workflow loads those checkpoints as a frozen ensemble and
trains the pair-correction head; it does not update the single-mutation models.

Run Tier 2 before the double-mutation workflow, or supply the three released
single-mutation checkpoints when running only the double-mutation stage.

## Datasets

From the repository root:

```bash
export SAAFEC_TRAINING_DATA_ROOT="$PWD/data_and_benchmarks/01_Datasets"
```

| Use | Relative workbook path |
|---|---|
| Single-mutation training | `01_Single_Mutation/01_Training/01_MegaScale_Training_Set.xlsx` |
| Single-mutation validation | `01_Single_Mutation/02_Validation/03_Final_Validation_Set.xlsx` |
| Single-mutation test | `01_Single_Mutation/03_Testing/04_Final_Test_Set.xlsx` |
| Double-mutation training | `02_Double_Mutation/01_Training/01_MegaScale_Training_Set.xlsx` |
| Double-mutation validation | `02_Double_Mutation/02_Validation/01_MegaScale_Validation_Set.xlsx` |
| Double-mutation test | `02_Double_Mutation/03_Testing/01_MegaScale_Test_Set.xlsx` |

The exact ColabFold-generated wild-type structures used for the manuscript are
not included in these workbooks. Set `STRUCTURE_ROOT` to a directory containing
those structures. Regenerating them with different software or databases may
produce different inputs. The expected layout is:

```text
STRUCTURE_ROOT/
  single_mutation/{training,validation,test}/
  double_mutation/{training,validation,test}/
```

Command-line arguments can point to another layout. See
[`docs/DATA_AND_ASSETS.md`](docs/DATA_AND_ASSETS.md) for the required structure
records and checksums.

## External Assets

Asset paths are resolved in this order: command-line argument, environment
variable, then the repository-relative path below.

| Variable | Repository-relative fallback |
|---|---|
| `SAPROT_MODEL_DIR` | `assets/external/models/SaProt_650M_PDB/` |
| `FOLDSEEK_BIN` | `assets/external/bin/foldseek` |
| `PROTEINMPNN_CHECKPOINT` | `assets/external/models/proteinmpnn/v_48_020.pt` |
| `PROTEINMPNN_SOURCE` | `assets/external/source/ThermoMPNN/protein_mpnn_utils.py` |
| `SINGLE_CHECKPOINT_DIR` | `assets/checkpoints/single_mutation/seeds/` |

Official sources:

- SaProt model: <https://huggingface.co/westlake-repl/SaProt_650M_PDB>
- SaProt code: <https://github.com/westlake-repl/SaProt>
- ProteinMPNN: <https://github.com/dauparas/ProteinMPNN>
- Foldseek: <https://github.com/steineggerlab/foldseek>
- SAAFEC-STAIR checkpoints: [included files](../inference_checkpoints/README.md), also available at <https://huggingface.co/multiverse49/SAAFEC-STAIR-inference-checkpoints>

The workflows load ThermoMPNN's `protein_mpnn_utils.py` at commit
`13569795daa7689b6a6df0279b383e08c6212e79`. The required SHA-256 is
`3bbcb4342482438bb5d4ebe6509d514490dfce804617865fa55ffdcbda2fea12`.
Third-party code and model weights are not included.

See [Installation](../docs/INSTALLATION.md),
[External assets](../docs/EXTERNAL_ASSETS.md), and
[Checkpoints](../docs/CHECKPOINTS.md) for setup instructions. The environment
files install Python packages only.

## Environment and Outputs

```bash
cd training
conda env create -f environment/conda.yml
conda activate saafec-stair-training
```

The same environment supports all four workflows. See
[`environment/README.md`](environment/README.md) for other platforms and asset
requirements. Outputs are written to the workflow's `output/` directory unless
`--output-root` is supplied.

## Verification

From the repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/saafec-stair-training-pycache \
  python -m compileall -q training

python training/single_mutation/tier0_zero_shot_mutation_ranking/run.py --help
python training/single_mutation/tier1_supervised_mutation_aware_prediction/run.py --help
python training/single_mutation/tier2_proteinmpnn_augmented_fusion/run.py --help
python training/pair_corrected_double_mutation/run.py --help
```

For a release, verify the dataset and structure checksums and record the tag or
commit used for the reported results.
