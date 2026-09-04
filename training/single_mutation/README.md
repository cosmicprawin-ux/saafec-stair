# Single-mutation training

The three directories correspond directly to the tiered evaluation described
in the manuscript:

1. `tier0_zero_shot_mutation_ranking/`
2. `tier1_supervised_mutation_aware_prediction/`
3. `tier2_proteinmpnn_augmented_fusion/`

Each tier has a `run.py` entry point and a short `RUNNING.md`. Its `code/`
directory contains the feature generation, data handling, model, training, and
evaluation modules used by that tier.
