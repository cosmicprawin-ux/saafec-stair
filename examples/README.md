# Examples

This directory contains four kinds of examples:

```text
examples/
  quickstart/
  worked_examples/
  input_templates/
  example_datasets/
```

## Quickstart

`quickstart/` contains the tiny default input tables and PDB files used by
`run_saafec_stair_inference.sh` when `INPUT_TABLE` and `PDB_DIR` are not set.

## Worked Examples

`worked_examples/` contains the authoritative runnable single- and
double-mutation examples. Each example includes a small input bundle, a run
script, and a selected `expected_outputs/` subset copied from a successful run
of the current runner. Use these examples first when checking an installation
or the current output schema. A run also creates additional intermediate and
diagnostic files below its ignored `run_outputs/` directory.

## Example Datasets

`example_datasets/` contains larger curated single- and double-mutation tables,
their matching PDB files, and a convenience workbook. These optional datasets
are useful for inspecting or preparing realistic inputs; they are not examples
of the runner's exact output layout. For authoritative output columns and
filenames, use `worked_examples/`.

## Input Templates

`input_templates/` contains clean CSV starting points for user-supplied single-
and double-mutation jobs. Its README documents the required columns and how the
single-mutation table controls full-matrix versus specified-mutation output.
