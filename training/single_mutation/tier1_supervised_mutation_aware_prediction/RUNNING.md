# Tier 1: supervised mutation-aware prediction

This command trains the mutation-aware head for seeds 1337, 2027, and 3407,
then averages their mutation-prediction tables.

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
  --seeds 1337,2027,3407 \
  --output-root output/tier1_saprot_supervised_mutation_aware_prediction
```

The default optimization, loss, early-stopping, and selection settings are
defined in the workflow code.
