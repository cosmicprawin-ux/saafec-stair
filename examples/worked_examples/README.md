# Worked Examples

The worked examples are the authoritative runnable examples for the current
repository. They provide small, portable input bundles and selected final-stage
outputs for checking inference behavior after all runtime assets are installed.

Single-mutation example:

```bash
bash examples/worked_examples/single_mutation_example/run_example.sh
```

Inputs:

```text
examples/worked_examples/single_mutation_example/inputs/single_mutations.txt
examples/worked_examples/single_mutation_example/inputs/pdbs/
```

Double-mutation example:

```bash
bash examples/worked_examples/double_mutation_example/run_example.sh
```

Inputs:

```text
examples/worked_examples/double_mutation_example/inputs/double_mutations.txt
examples/worked_examples/double_mutation_example/inputs/pdbs/
```

Generated files are written below each example's ignored `run_outputs/`
directory.

## Included Expected Outputs

Each example's tracked `expected_outputs/` directory contains the final
user-facing CSV outputs from a successful run of the current runner:

- Single mutation, full-matrix use: one three-seed-averaged `L x 20` predicted
  DDG matrix per input structure under `predictions/predicted_DDG/`.
- Single mutation, specified-mutation use: one compact table containing only
  the requested substitutions under `specified_DDG_predictions/`.
- Double mutation: the final requested double-mutation prediction table.

The single-mutation runner averages the three seed matrices internally and
writes only the averaged matrices and, when requested, the specified-mutation
table as prediction products. A full run also writes heatmaps, feature caches,
manifests, summaries, and validation diagnostics below the ignored
`run_outputs/` directory.

Reusable input templates are maintained separately in
[`examples/input_templates/`](../input_templates/).
