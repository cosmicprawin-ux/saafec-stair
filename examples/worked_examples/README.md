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

Each example's tracked `expected_outputs/` directory contains a deliberately
small subset of final CSV outputs from a successful run of the current runner:

- Single mutation: the final ensemble table and the specified-mutation table.
- Double mutation: the final requested double-mutation prediction table.

These files preserve the current filenames, column names, row order, statuses,
and predicted ddG values. A full run also writes per-seed tables, matrices,
heatmaps, feature caches, manifests, summaries, and other diagnostic files.
Those generated files remain below `run_outputs/` and are not duplicated in the
repository.

Reusable input templates are maintained separately in
[`examples/input_templates/`](../input_templates/).
