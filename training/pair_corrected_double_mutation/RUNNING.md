# Pair-corrected double-mutation training

This is the downstream stage of SAAFEC-STAIR training. It assumes that the
three Tier 2 single-mutation predictors for seeds 1337, 2027, and 3407 have
already been trained. The workflow loads those checkpoints as a frozen
ensemble, generates the required single-mutation prediction matrices and
double-mutation feature caches, and trains only the pair-correction head.

For the complete training sequence, set `TIER2_OUTPUT_ROOT` to the output root
used for Tier 2. The default directory produced by the Tier 2 example is:

```bash
TIER2_OUTPUT_ROOT="../single_mutation/tier2_proteinmpnn_augmented_fusion/output/tier2_saprot_proteinmpnn_augmented_fusion"
```

```bash
python run.py \
  --train-xlsx "$SAAFEC_TRAINING_DATA_ROOT/02_Double_Mutation/01_Training/01_MegaScale_Training_Set.xlsx" \
  --train-pdb-dir "$STRUCTURE_ROOT/double_mutation/training" \
  --val-xlsx "$SAAFEC_TRAINING_DATA_ROOT/02_Double_Mutation/02_Validation/01_MegaScale_Validation_Set.xlsx" \
  --val-pdb-dir "$STRUCTURE_ROOT/double_mutation/validation" \
  --test-xlsx "$SAAFEC_TRAINING_DATA_ROOT/02_Double_Mutation/03_Testing/01_MegaScale_Test_Set.xlsx" \
  --test-pdb-dir "$STRUCTURE_ROOT/double_mutation/test" \
  --single-checkpoint "$TIER2_OUTPUT_ROOT/seeds/seed_1337/phase1/checkpoints/best_head.pt" \
  --single-checkpoint "$TIER2_OUTPUT_ROOT/seeds/seed_2027/phase1/checkpoints/best_head.pt" \
  --single-checkpoint "$TIER2_OUTPUT_ROOT/seeds/seed_3407/phase1/checkpoints/best_head.pt" \
  --saprot-model-dir "$SAPROT_MODEL_DIR" \
  --foldseek-bin "$FOLDSEEK_BIN" \
  --proteinmpnn-checkpoint "$PROTEINMPNN_CHECKPOINT" \
  --proteinmpnn-source "$PROTEINMPNN_SOURCE" \
  --output-root output/pair_corrected_double_mutation
```

Add `--dry-run` to inspect the staged commands without generating caches or
starting training. The double-mutation objective and its regularization terms
are defined in the training module.

To run only this downstream stage, use the three released single-mutation
checkpoints documented in the public inference repository instead. Their
simplified asset layout is addressed by `SINGLE_CHECKPOINT_DIR`:

```text
SINGLE_CHECKPOINT_DIR/
  seed_1337/best_head.pt
  seed_2027/best_head.pt
  seed_3407/best_head.pt
```
