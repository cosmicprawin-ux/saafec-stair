#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p \
  assets/external/models/SaProt_650M_PDB \
  assets/external/models/proteinmpnn \
  assets/external/bin \
  assets/external/source/ThermoMPNN \
  assets/checkpoints/single_mutation/seeds/seed_1337 \
  assets/checkpoints/single_mutation/seeds/seed_2027 \
  assets/checkpoints/single_mutation/seeds/seed_3407 \
  assets/checkpoints/double_mutation

cat <<'MSG'
Asset directories are ready. This script intentionally does not download,
copy, or redistribute third-party code, binaries, weights, or model
checkpoints.

Restore each required file from its official source as described in:
  docs/EXTERNAL_ASSETS.md
  docs/CHECKPOINTS.md

Then validate the installation with:
  python scripts/tools/check_installation.py --mode single
  python scripts/tools/check_installation.py --mode double
MSG
