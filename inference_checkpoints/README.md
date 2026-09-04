# SAAFEC-STAIR Inference Checkpoints

This folder contains the four trained checkpoints used by SAAFEC-STAIR inference. They are also available on [Hugging Face](https://huggingface.co/multiverse49/SAAFEC-STAIR-inference-checkpoints).

| Model | Checkpoint | Size |
| --- | --- | ---: |
| Single mutation, seed 1337 | [best_head.pt](single_mutation/seeds/seed_1337/best_head.pt) | 33.33 MB |
| Single mutation, seed 2027 | [best_head.pt](single_mutation/seeds/seed_2027/best_head.pt) | 33.33 MB |
| Single mutation, seed 3407 | [best_head.pt](single_mutation/seeds/seed_3407/best_head.pt) | 33.33 MB |
| Double mutation | [best_double_mutation_package.pt](double_mutation/best_double_mutation_package.pt) | 15.48 MB |

Cloning the repository includes these files. To download one file through GitHub, open its link and select **Download raw file**. The three single-mutation checkpoints form the inference ensemble.

## Use with inference

From the repository root, copy the bundled checkpoints into the default runtime location:

```bash
mkdir -p assets/checkpoints
cp -R inference_checkpoints/single_mutation assets/checkpoints/
cp -R inference_checkpoints/double_mutation assets/checkpoints/
```

Alternatively, avoid making a second copy by setting `SINGLE_CHECKPOINT_DIR` to the absolute path of `inference_checkpoints/single_mutation/seeds` and `DOUBLE_HEAD_CHECKPOINT` to the absolute path of `inference_checkpoints/double_mutation/best_double_mutation_package.pt` when running inference. See the [checkpoint setup guide](../docs/CHECKPOINTS.md).

Third-party backbone models, Foldseek, and other external dependencies are obtained separately as described in [External assets](../docs/EXTERNAL_ASSETS.md).

## Source and integrity

The files were downloaded from Hugging Face revision `f15ebdf3eea82f5414dcff942e1740b26166d6c6` on September 3, 2026, and verified against its published SHA-256 checksums. Only directory paths were simplified; checkpoint contents are unchanged. [SOURCE.json](SOURCE.json) records the original paths and file hashes.

Verify the files from this directory on Linux:

```bash
sha256sum -c SHA256SUMS.txt
```

On macOS, use `shasum -a 256 -c SHA256SUMS.txt`.
