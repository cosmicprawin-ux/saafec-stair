# Tier 2: ProteinMPNN-augmented fusion

This is the three-seed SaProt–ProteinMPNN single-mutation workflow selected for
SAAFEC-STAIR.

```bash
python run.py \
  --train-xlsx "$SAAFEC_TRAINING_DATA_ROOT/01_Single_Mutation/01_Training/01_MegaScale_Training_Set.xlsx" \
  --train-pdb-dir "$STRUCTURE_ROOT/single_mutation/training" \
  --val-xlsx "$SAAFEC_TRAINING_DATA_ROOT/01_Single_Mutation/02_Validation/03_Final_Validation_Set.xlsx" \
  --val-pdb-dir "$STRUCTURE_ROOT/single_mutation/validation" \
  --test-xlsx "$SAAFEC_TRAINING_DATA_ROOT/01_Single_Mutation/03_Testing/04_Final_Test_Set.xlsx" \
  --test-pdb-dir "$STRUCTURE_ROOT/single_mutation/test" \
  --saprot-model-dir "$SAPROT_MODEL_DIR" \
  --foldseek-bin "$FOLDSEEK_BIN" \
  --proteinmpnn-checkpoint "$PROTEINMPNN_CHECKPOINT" \
  --proteinmpnn-source "$PROTEINMPNN_SOURCE" \
  --seeds 1337,2027,3407 \
  --output-root output/tier2_saprot_proteinmpnn_augmented_fusion
```

The ProteinMPNN source must be the pinned ThermoMPNN utility file documented
in the top-level README; its checksum is verified before import.
