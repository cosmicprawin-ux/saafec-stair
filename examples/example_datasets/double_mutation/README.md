# Double-Mutation Example Dataset

This folder contains a larger curated double-mutation dataset. It is useful for
inspecting realistic mutation rows and preparing inputs, but it is not an exact
copy of the current runner output. For the current output schema, see
`examples/worked_examples/double_mutation_example/expected_outputs/`.

## Files

- `examples.csv`: reference table with protein/PDB/chain, double-mutation details, and predicted ΔΔG.
- `pdbs/`: PDB files referenced by `examples.csv`.

## Counts

- Proteins: 5
- PDB files: 5
- Prediction rows: 19

## CSV Columns

- `pdb_file`: matching PDB file in `pdbs/`.
- `chain`: protein chain used for prediction.
- `double_mutation`: paired amino-acid substitutions.
- `position_1`, `wt_aa_1`, `mt_aa_1`: first substitution.
- `position_2`, `wt_aa_2`, `mt_aa_2`: second substitution.
- `predicted_ddg`: retained SAAFEC-STAIR reference prediction; not a current runner-output column definition.
