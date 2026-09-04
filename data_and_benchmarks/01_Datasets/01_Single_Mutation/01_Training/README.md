# Single-Mutation Training Set

[01_MegaScale_Training_Set.xlsx](01_MegaScale_Training_Set.xlsx) contains 215,731 mutations across 239 proteins. It is the training set used for the supervised single-mutation models.

The measurements come from the MegaScale archive (https://doi.org/10.5281/zenodo.7992926), and the protein split follows ThermoMPNN (https://github.com/Kuhlman-Lab/ThermoMPNN/tree/main/dataset_splits). ΔΔG is reported in kcal/mol using ΔG(mutant) − ΔG(wild type).

Selected compact headers: `prot_index` is the protein index, `prot_mutation_index` is the mutation index within that protein, `identifier` identifies the mutation record, and `mut_pos_seq` and `mut_pos_pdb` give sequence and PDB residue numbering.
