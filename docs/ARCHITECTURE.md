# Repository Architecture

## Directory Tree

```text
saafec-stair/
  .github/workflows/validate.yml
  docs/
  environment/
  scripts/
    cache/                     # SaProt and ProteinMPNN feature generation
    core/                      # Parsing, datasets, and shared configuration
    external/                  # Loader for user-supplied upstream code
    inference/                 # Single- and double-mutation inference
    models/                    # SAAFEC-STAIR model-head definitions
    setup/                     # Local directory preparation
    tools/                     # Installation validation
    utils/
    visualization/
  inference_checkpoints/     # Four published SAAFEC-STAIR model checkpoints
  assets/
    checkpoints/               # Ignored runtime checkpoint copies
    external/                  # Ignored third-party asset destinations
  examples/
    quickstart/
    worked_examples/
  tests/
  training/                 # Training workflows for the reported models
  data_and_benchmarks/       # Manuscript datasets, predictions, metrics and audits
  run_saafec_stair_inference.sh
```

The repository does not include scheduler configuration, third-party
executables or weights, training-run checkpoints, caches, ordinary run
outputs, or machine-specific paths. Reference outputs under
`examples/worked_examples/` and manuscript datasets and benchmark results under
[`data_and_benchmarks/`](../data_and_benchmarks/README.md) are tracked. The four
published inference checkpoints are tracked under
[`inference_checkpoints/`](../inference_checkpoints/README.md), with checksums.

## Single-Mutation Flow

1. Parse the mutation table or prepare one input structure.
2. Run Foldseek and SaProt to generate structure-aware embeddings.
3. Run ProteinMPNN to generate residue logits on the same sequence axis.
4. Apply the three SAAFEC-STAIR single-mutation inference heads.
5. Average the three seed matrices internally and write only final averaged
   matrices, optional specified-mutation tables, and heatmaps as prediction
   products.

## Double-Mutation Flow

1. Build a manifest containing only requested mutation pairs.
2. Generate SaProt embeddings and ProteinMPNN logits.
3. Use the single-mutation ensemble to create implicit prior matrices.
4. Apply the contact-gated double-mutation inference head.
5. Write final DDG predictions and component values.

## Code Layout

Inference and model-head code is in `scripts/`. Third-party files are loaded
from `assets/external/` or paths supplied through environment variables.
SAAFEC-STAIR checkpoints are loaded from `assets/checkpoints/`.

Training workflows are in `training/` and are not imported by the inference
runner. Each workflow keeps its supporting code within its own directory.

The repository includes the four SAAFEC-STAIR checkpoints. Third-party code,
executables, and model weights are installed separately under their own
licenses.
