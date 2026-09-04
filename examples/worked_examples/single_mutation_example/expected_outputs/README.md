# Single-Mutation Expected Outputs

This directory contains the two single-mutation output formats produced by the
worked example.

## Full-Matrix Predictions

`predictions/predicted_DDG/` contains one final, three-seed-averaged matrix per
input structure. Each CSV has one row per residue, residue-identification
columns, and 20 amino-acid DDG columns. The top row labels each mutant
prediction as `mut_aa_DDG`; the row beneath identifies the mutant amino acid.
Displayed DDG values use two decimal places. The wild-type entry is zero, so
each row represents 19 actual substitutions plus the wild-type reference.

These matrices contain every single-amino-acid substitution for each input
protein.

## Heatmaps

`predictions/visualizations/` contains the interactive HTML heatmap generated
for each full-matrix prediction.

## Specified-Mutation Predictions

`specified_DDG_predictions/specified_single_mutation_DDG_predictions.csv`
contains only the substitutions requested in the specified-mutation input
table.

This table contains only the mutations listed in the input file. The runner
still computes and averages the three seed matrices before writing the output.
