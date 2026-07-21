# Installation

## Environment

The tested Conda environment is:

```bash
conda env create -f environment/conda.yml
conda activate SAAFEC_STAIR_inference
```

Alternatively, use Python 3.11 or 3.12 in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
```

PyTorch installation may need to be adjusted for the local CPU or CUDA
platform.

The inference runner respects an already-active Conda or `venv` environment.
It activates `SAAFEC_STAIR_inference` only when no environment is active and
that named Conda environment exists. `CONDA_ENV_NAME` can select another named
Conda environment, and `CONDA_ENV_PATH` can select an environment by path.

## Runtime Assets

Create the expected ignored directories:

```bash
bash scripts/setup/prepare_asset_directories.sh
```

Then restore:

1. SaProt 650M PDB model files.
2. ProteinMPNN `v_48_020.pt`.
3. The pinned upstream ThermoMPNN `protein_mpnn_utils.py`.
4. A compatible Foldseek executable.
5. Four SAAFEC-STAIR inference model-weight checkpoints.

The exact sources and destinations are documented in
[EXTERNAL_ASSETS.md](EXTERNAL_ASSETS.md) and
[CHECKPOINTS.md](CHECKPOINTS.md). These materials are deliberately ignored by
Git and must not be committed.

## Validation

Check the public source tree without runtime assets:

```bash
python scripts/tools/check_installation.py --source-only
```

After restoring assets, validate both modes:

```bash
python scripts/tools/check_installation.py --mode single
python scripts/tools/check_installation.py --mode double
```

The validator honors `SAPROT_MODEL_DIR`, `FOLDSEEK_BIN`,
`PROTEINMPNN_CHECKPOINT`, `PROTEINMPNN_SOURCE`, `SINGLE_CHECKPOINT_DIR`, and
`DOUBLE_HEAD_CHECKPOINT` when assets are stored elsewhere.
