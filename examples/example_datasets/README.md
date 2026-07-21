# Example Datasets

This directory contains larger curated single- and double-mutation datasets,
their matching PDB files, and a convenience workbook. The tables retain
SAAFEC-STAIR predicted ΔΔG values as reference data, but they are not literal
copies of files produced by the current inference runner.

For exact current output filenames and schemas, use the selected outputs in
[`examples/worked_examples/`](../worked_examples/) or run those examples. The
runner additionally writes per-seed results, matrices, feature caches,
visualizations, and diagnostic summaries that are intentionally not duplicated
here.

```text
examples/example_datasets/
  single_mutation/
    examples.csv
    pdbs/
  double_mutation/
    examples.csv
    pdbs/
  saafec_stair_example_datasets.xlsx
```

These files are optional and are not runtime dependencies. If adapting a table
for inference, set `INPUT_TABLE` and `PDB_DIR` explicitly when calling
`run_saafec_stair_inference.sh`; the files written below the selected `OUT_ROOT`
are then authoritative.
