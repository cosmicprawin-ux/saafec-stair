# Worked Examples

These examples provide small input sets and expected outputs for checking an
installation after the required runtime assets are available.

Single-mutation example:

```bash
bash examples/worked_examples/single_mutation_example/run_example.sh
```

Inputs:

```text
examples/worked_examples/single_mutation_example/inputs/single_mutation_list.txt
examples/worked_examples/single_mutation_example/inputs/pdbs/
```

Double-mutation example:

```bash
bash examples/worked_examples/double_mutation_example/run_example.sh
```

Inputs:

```text
examples/worked_examples/double_mutation_example/inputs/double_mutation_list.txt
examples/worked_examples/double_mutation_example/inputs/pdbs/
```

Generated files are written below each example's ignored `run_outputs/`
directory.

## Included Expected Outputs

Each `expected_outputs/` directory contains results from a completed run:

- Single mutation, full-matrix use: one three-seed-averaged `L x 20` predicted
  DDG matrix per input structure under `predictions/predicted_DDG/`.
- Single mutation, specified-mutation use: one compact table containing only
  the requested substitutions under `specified_DDG_predictions/`.
- Single mutation: one interactive HTML heatmap per input structure under
  `predictions/visualizations/`.
- Double mutation: the final requested double-mutation prediction table.

The single-mutation runner averages the three seed matrices before writing the
final matrices and, when requested, the specified-mutation table. The example
also includes the corresponding heatmaps. A full run writes feature caches,
manifests, and summaries below the ignored `run_outputs/` directory. Prediction
CSVs report `Unit(DDG)=kcal/mol` in the top-left cell and display final DDG
predictions to two decimal places.

The single-mutation list is optional for one-structure inference. Without it,
the runner writes the complete matrix and heatmap. When the list is provided,
the runner writes those complete-profile outputs and the additional specified-
mutation CSV. The lists in these worked examples use tabs, can be copied and
edited for new inputs, and also accept spaces between columns.
