# Double-Mutation Training Set

[01_MegaScale_Training_Set.xlsx](01_MegaScale_Training_Set.xlsx) contains 122,278 double-mutation measurements across 116 proteins. It was used to train the SAAFEC-STAIR pair-correction model.

The file comes from the ThermoMPNN-D archive (https://doi.org/10.5281/zenodo.13345274), and the protein split follows ThermoMPNN. ΔΔG is reported in kcal/mol using ΔG(mutant) − ΔG(wild type).

Selected compact headers: `prot_index` is the protein index, `prot_mutation_index` is the mutation index within that protein, `identifier` identifies the mutation record, and `mut_pos_seq` and `mut_pos_pdb` give sequence and PDB residue numbering.
