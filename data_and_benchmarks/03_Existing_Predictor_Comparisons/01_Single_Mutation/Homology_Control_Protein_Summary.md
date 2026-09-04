# Single-Mutation Homology-Control Protein Summary

This note summarizes how proteins were kept or removed for each comparator.

## Homology Rule

Proteins were removed from a comparator’s retained subset when they had at least 25% sequence identity and at least 80% coverage to proteins used for that comparator’s training or model exposure.
Proteins below these thresholds were retained.

The [05_Homology-Control_Summary](05_Homology-Control_Summary.xlsx) workbook contains these sheets:

- [Homology Control Proteins](05_Homology-Control_Summary.xlsx): full protein list
- [Homology Control Counts](05_Homology-Control_Summary.xlsx): compact count summary
- [Performance Summary](05_Homology-Control_Summary.xlsx): matched performance metrics

Mutation-level matrices are provided in the `Predictions` sheet in the [PROSTATA](02_PROSTATA_Homology-Filtered_Comparison.xlsx), [MutateEverything](03_MutateEverything_Homology-Filtered_Comparison.xlsx) and [GeoDDG-3D](04_GeoDDG-3D_Homology-Filtered_Comparison.xlsx) comparison workbooks.

## Protein-Level Counts

| Method | Kept proteins | Kept mutations | Removed proteins | Removed mutations |
|---|---:|---:|---:|---:|
| PROSTATA | 42 | 8,999 | 344 | 25,237 |
| MutateEverything | 354 | 15,435 | 32 | 18,801 |
| GeoDDG-3D | 99 | 24,128 | 287 | 10,108 |

## Interpretation

`Kept` proteins were used in each comparator-specific homology-control analysis.
`Removed` proteins were excluded because they met the homology threshold.

Protein lists are included for readability.
Performance is reported at the mutation level using matched retained rows.
