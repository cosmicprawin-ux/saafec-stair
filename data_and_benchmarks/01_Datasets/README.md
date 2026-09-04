# Datasets

This directory contains the single- and double-mutation datasets used in the study.

| Task | Training | Validation | Test | Details |
|---|---:|---:|---:|---|
| Single mutation | 215,731 mutations; 239 proteins | 27,697 mutations; 55 proteins | 34,236 mutations; 386 proteins | [Single-mutation datasets](01_Single_Mutation/README.md) |
| Double mutation | 122,278 measurements; 116 proteins | 10,672 measurements; 17 proteins | 22,081 measurements; 20 proteins | [Double-mutation datasets](02_Double_Mutation/README.md) |

MegaScale single-mutation measurements are available at https://doi.org/10.5281/zenodo.7992926, and the protein-level partition follows https://github.com/Kuhlman-Lab/ThermoMPNN/tree/main/dataset_splits. Double-mutation files were obtained from https://doi.org/10.5281/zenodo.13345274. Sources for the additional single-mutation datasets are listed with the [source workbooks](01_Single_Mutation/03_Testing/01_Source_Databases/README.md).

ΔΔG is defined as ΔG(mutant) − ΔG(wild type), is reported in kcal/mol, and is positive for destabilizing mutations.
