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
table. The `status` and `message` columns report whether each requested value
could be extracted from the corresponding full matrix.

Use this compact table when only listed mutations are needed. The runner still
computes the matrices required for inference, but per-seed and ensemble-member
detail tables are not included here because they are internal aggregation and
diagnostic outputs rather than the final prediction product.
