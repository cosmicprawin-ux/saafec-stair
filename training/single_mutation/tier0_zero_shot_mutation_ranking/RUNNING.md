# Tier 0: zero-shot mutation ranking

Tier 0 evaluates SaProt masked-marginal mutation scores without a supervised
prediction head or ProteinMPNN features.

```bash
python run.py \
  --test-xlsx "$SAAFEC_TRAINING_DATA_ROOT/01_Single_Mutation/03_Testing/04_Final_Test_Set.xlsx" \
  --test-pdb-dir "$STRUCTURE_ROOT/single_mutation/test" \
  --saprot-model-dir "$SAPROT_MODEL_DIR" \
  --foldseek-bin "$FOLDSEEK_BIN" \
  --output-root output/tier0_saprot_zero_shot_mutation_ranking
```

The workbook is part of the paper support data; the structure directory and
SaProt/Foldseek assets are supplied separately.
