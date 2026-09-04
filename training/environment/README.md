# Training environment

The same Python environment supports all four training workflows. Create the
tested CUDA 12.1 Conda environment with:

```bash
conda env create -f environment/conda.yml
conda activate saafec-stair-training
```

Alternatively, create a Python 3.12 virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
```

The supplied requirements select the CUDA 12.1 PyTorch build used for the
training workflows. Install the corresponding PyTorch build first and remove
the PyTorch lines from `requirements.txt` when using another CUDA version,
Apple silicon, or CPU-only execution.

ProteinMPNN does not require a separate environment. Tier 2 and the
double-mutation workflow load the externally supplied ProteinMPNN/ThermoMPNN
source with the same PyTorch and NumPy installation. The workflow-specific
external assets are:

| Workflow | SaProt | Foldseek | ProteinMPNN source and checkpoint | Completed Tier 2 checkpoints |
| --- | ---: | ---: | ---: | ---: |
| Tier 0 | Required | Required | No | No |
| Tier 1 | Required | Required | No | No |
| Tier 2 | Required | Required | Required | No |
| Pair-corrected double mutation | Required | Required | Required | Required |

Foldseek, model weights, ProteinMPNN source, datasets, and structures are
external assets and are not installed by the Python environment files.

## External models and checkpoints

The repository's inference documentation provides the corresponding asset
layout, download instructions, checksums, and checkpoint locations:

- [Installation](../../docs/INSTALLATION.md)
- [External SaProt, Foldseek, and ProteinMPNN assets](../../docs/EXTERNAL_ASSETS.md)
- [SAAFEC-STAIR checkpoints](../../docs/CHECKPOINTS.md)

The released SAAFEC-STAIR checkpoint files are hosted in the
[SAAFEC-STAIR Hugging Face repository](https://huggingface.co/multiverse49/SAAFEC-STAIR-inference-checkpoints).
The pair-corrected double-mutation workflow is a downstream training stage. It
expects the three completed Tier 2 single-mutation checkpoints for seeds 1337,
2027, and 3407 and keeps them frozen while training the pair-correction head.
For an end-to-end training sequence, these checkpoints are produced by running
Tier 2 first and are read from each seed's
`phase1/checkpoints/best_head.pt` output. The released checkpoints provide an
alternative simplified asset layout when running only the downstream
double-mutation stage. The released double-mutation checkpoint is used for
inference and is not an input to double-mutation training.

The environment files do not download or bundle these assets.
Users may place them in the repository-relative `assets/` layout documented by
the inference repository or supply alternate locations through the environment
variables and command-line arguments described in the top-level training
README.
