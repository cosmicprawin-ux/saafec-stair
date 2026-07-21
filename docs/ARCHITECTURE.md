# Repository Architecture

## Public Tree

```text
saafec-stair/
  .github/workflows/validate.yml
  assets/
    checkpoints/               # Ignored checkpoint destinations
    external/                  # Ignored third-party asset destinations
  docs/
  environment/
  examples/
    quickstart/
    worked_examples/
    input_templates/
    example_datasets/
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
  run_saafec_stair_inference.sh
```

The public tree has no scheduler configuration. It also excludes executables,
third-party weights and source, model checkpoints, caches, generated outputs,
and machine-specific paths.

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

## Ownership Boundaries

SAAFEC-STAIR inference orchestration and model-head code lives in `scripts/`.
Third-party runtime material is loaded only from `assets/external/`
or explicit environment-variable paths. Project checkpoints are loaded from
`assets/checkpoints/`.

This boundary allows the source package to be reviewed and tested independently
while preserving upstream licensing and keeping large checkpoint files outside
the Git repository.
