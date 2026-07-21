# Single-Mutation Example Dataset

This folder contains a larger curated single-mutation dataset. It is useful for
inspecting realistic mutation rows and preparing inputs, but it is not an exact
copy of the current runner output. For the current output schema, see
`examples/worked_examples/single_mutation_example/expected_outputs/`.

## Files

- `examples.csv`: reference table with protein/PDB/chain, mutation details, and predicted ΔΔG.
- `pdbs/`: PDB files referenced by `examples.csv`.

## Counts

- Proteins: 10
- PDB files: 10
- Prediction rows: 10273

## CSV Columns

- `pdb_file`: matching PDB file in `pdbs/`.
- `chain`: protein chain used for prediction.
- `mutation`: single amino-acid substitution.
- `wt_aa`: wild-type amino acid.
- `position_1based`: one-based residue position used for the mutation.
- `pdb_residue_id`: chain-qualified PDB residue identifier.
- `pdb_resseq`: PDB residue sequence number.
- `mut_aa`: mutant amino acid.
- `predicted_ddg`: retained SAAFEC-STAIR reference prediction; not a current runner-output column definition.
