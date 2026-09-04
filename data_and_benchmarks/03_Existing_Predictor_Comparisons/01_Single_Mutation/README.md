# Single-Mutation Predictor Comparisons

| File | Contents |
|---|---|
| [01_Final_Test_Set_Predictions.xlsx](01_Final_Test_Set_Predictions.xlsx) | All evaluated methods on the final test set: 34,236 mutations across 386 proteins. |
| [02_PROSTATA_Homology-Filtered_Comparison.xlsx](02_PROSTATA_Homology-Filtered_Comparison.xlsx) | SAAFEC-STAIR and PROSTATA on 8,999 retained mutations. |
| [03_MutateEverything_Homology-Filtered_Comparison.xlsx](03_MutateEverything_Homology-Filtered_Comparison.xlsx) | SAAFEC-STAIR and MutateEverything on 15,435 retained mutations. |
| [04_GeoDDG-3D_Homology-Filtered_Comparison.xlsx](04_GeoDDG-3D_Homology-Filtered_Comparison.xlsx) | SAAFEC-STAIR and GeoDDG-3D on 24,128 retained mutations. |
| [05_Homology-Control_Summary.xlsx](05_Homology-Control_Summary.xlsx) | Performance, counts and protein lists for the three controls. |

For each control, proteins with at least 25% sequence identity and 80% coverage to a comparator training or exposure protein were removed. SAAFEC-STAIR and the comparator were evaluated on the same retained mutations. Missing predictions are shown as `-`.
