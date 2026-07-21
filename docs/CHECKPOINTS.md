# SAAFEC-STAIR Inference Checkpoints

The SAAFEC-STAIR inference model weights are distributed as checkpoint files.
They are project assets and are not stored in GitHub. Restore them from the
public Hugging Face model repository:

<https://huggingface.co/multiverse49/SAAFEC-STAIR-inference-checkpoints>

Download a local snapshot:

```bash
hf download multiverse49/SAAFEC-STAIR-inference-checkpoints \
  --local-dir /tmp/saafec-stair-checkpoints
```

The required layout is:

```text
assets/checkpoints/
  single_mutation/
    seeds/
      seed_1337/best_head.pt
      seed_2027/best_head.pt
      seed_3407/best_head.pt
  double_mutation/
    best_double_mutation_package.pt
```

Copy the three `best_head.pt` files from their corresponding seed directories
in the downloaded snapshot to the simplified destinations above. Copy
`best_double_mutation_package.pt` to the double-mutation destination. Do not
commit model-weight checkpoint files; all common model formats under
`assets/checkpoints/` are ignored.

When centrally managed checkpoints should remain elsewhere, set:

```bash
SINGLE_CHECKPOINT_DIR=/shared/saafec-stair/single_mutation/seeds \
DOUBLE_HEAD_CHECKPOINT=/shared/saafec-stair/double_mutation/best_double_mutation_package.pt \
MODE=double \
bash run_saafec_stair_inference.sh
```

Validate the restored files with:

```bash
python scripts/tools/check_installation.py --mode single
python scripts/tools/check_installation.py --mode double
```
