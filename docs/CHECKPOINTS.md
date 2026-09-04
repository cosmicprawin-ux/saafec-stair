# SAAFEC-STAIR Inference Checkpoints

The four SAAFEC-STAIR inference model weights are included in
[`inference_checkpoints/`](../inference_checkpoints/README.md). Cloning the
repository downloads them with the source code.

From the repository root, copy them into the default runtime destinations:

```bash
mkdir -p assets/checkpoints
cp -R inference_checkpoints/single_mutation assets/checkpoints/
cp -R inference_checkpoints/double_mutation assets/checkpoints/
```

The same checkpoint files are also available from the public Hugging Face model repository:

<https://huggingface.co/multiverse49/SAAFEC-STAIR-inference-checkpoints>

To obtain a separate Hugging Face snapshot instead:

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

For a Hugging Face snapshot, copy the three `best_head.pt` files from
`checkpoints/single_mutation/seeds/seed_<SEED>/phase1/checkpoints/` to the
corresponding destinations above. Copy `best_double_mutation_package.pt`
from `checkpoints/double_mutation/` to the double-mutation destination.
Runtime copies under `assets/checkpoints/` remain ignored by Git.

To use the bundled files directly without copying them, run from the repository root:

```bash
SINGLE_CHECKPOINT_DIR="$PWD/inference_checkpoints/single_mutation/seeds" \
DOUBLE_HEAD_CHECKPOINT="$PWD/inference_checkpoints/double_mutation/best_double_mutation_package.pt" \
MODE=double \
bash run_saafec_stair_inference.sh
```

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
