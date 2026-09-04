# SAAFEC-STAIR Data and Benchmarks

This directory contains the datasets and prediction results reported in the SAAFEC-STAIR study.

| Folder | Contents |
|---|---|
| [01_Datasets](01_Datasets/README.md) | Training, validation and test datasets, including filtering records. |
| [02_Model_Tier_Comparisons](02_Model_Tier_Comparisons/README.md) | Results from the three single-mutation model tiers. |
| [03_Existing_Predictor_Comparisons](03_Existing_Predictor_Comparisons/README.md) | Comparisons with existing single- and double-mutation predictors. |

## Final datasets

| Task | Training | Validation | Test |
|---|---:|---:|---:|
| Single mutation | 215,731 mutations; 239 proteins | 27,697 mutations; 55 proteins | 34,236 mutations; 386 proteins |
| Double mutation | 122,278 measurements; 116 proteins | 10,672 measurements; 17 proteins | 22,081 measurements; 20 proteins |

The files are Excel workbooks. Prediction workbooks generally contain separate sheets for row-level predictions and performance metrics; audit sheets are included where filtering was performed. Missing predictions are shown as `-`.

Unless stated otherwise, ΔΔG is defined as ΔG(mutant) − ΔG(wild type), is reported in kcal/mol, and is positive for destabilizing mutations. Tier 0 values are ranking scores rather than calibrated ΔΔG predictions.
