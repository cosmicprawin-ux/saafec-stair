# Single-Mutation Expected Outputs

This directory presents the two user-facing single-mutation output forms from
the successful worked-example run.

## Full-Matrix Predictions

`predictions/predicted_DDG/` contains one final, three-seed-averaged matrix per
input structure. Each CSV has one row per residue, residue-identification
columns, and 20 amino-acid DDG columns. The wild-type entry is zero, so each
row represents 19 actual substitutions plus the wild-type reference.

Use these matrices when the goal is to predict every single-amino-acid
substitution for each input protein.

## Specified-Mutation Predictions

`specified_DDG_predictions/specified_single_mutation_DDG_predictions.csv`
contains only the substitutions requested in the specified-mutation input
table. It is a prediction-only table and does not mix validation state into the
scientific result columns.

Use this compact table when only listed mutations are needed. The runner still
computes the matrices required for inference and averages the three seed
matrices before writing outputs. If any requested mutation cannot be extracted,
the runner writes a separate
`specified_single_mutation_DDG_validation_errors.csv` diagnostic report and
records counts in the summary JSON.
