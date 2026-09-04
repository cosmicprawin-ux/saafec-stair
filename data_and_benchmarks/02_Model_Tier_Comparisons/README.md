# Single-Mutation Model-Tier Comparisons

Each workbook contains predictions and performance metrics for the 34,236-mutation final test set.

| Tier | Contents | File |
|---|---|---|
| Tier 0 | Ranking scores from frozen models | [01_Tier_0_Zero-Shot_Mutation_Ranking.xlsx](01_Tier_0_Zero-Shot_Mutation_Ranking.xlsx) |
| Tier 1 | Supervised ΔΔG predictions from frozen residue representations | [02_Tier_1_Supervised_Mutation-Aware_Prediction.xlsx](02_Tier_1_Supervised_Mutation-Aware_Prediction.xlsx) |
| Tier 2 | Supervised predictions with ProteinMPNN input | [03_Tier_2_ProteinMPNN-Augmented_Fusion.xlsx](03_Tier_2_ProteinMPNN-Augmented_Fusion.xlsx) |

Tier 0 scores are not calibrated in kcal/mol. Tier 1 and Tier 2 use ΔΔG = ΔG(mutant) − ΔG(wild type). Missing predictions are shown as `-`.
