# Single-Mutation Test-Set Construction

The files follow the order used to build the final test set.

| Step | File | Size |
|---|---|---:|
| Source datasets | [01_Source_Databases](01_Source_Databases/README.md) | 47,325 source records; 417 protein chains |
| Combined set | [02_Combined_Test_Set.xlsx](02_Combined_Test_Set.xlsx) | 47,325 records; 417 protein chains |
| After duplicate filtering | [03_After_Duplicate_Filtering.xlsx](03_After_Duplicate_Filtering.xlsx) | 35,380 records; 412 proteins |
| Final test set | [04_Final_Test_Set.xlsx](04_Final_Test_Set.xlsx) | 34,236 mutations; 386 proteins |
| Filtering record | [05_Filter_Audit.xlsx](05_Filter_Audit.xlsx) | Duplicate groups and removed proteins and mutations |

The final set was obtained after duplicate and homology filtering and removal of V30I in 1AZP_A. It also defines the row order used in the single-mutation prediction files. ΔΔG is reported in kcal/mol using ΔG(mutant) − ΔG(wild type).
