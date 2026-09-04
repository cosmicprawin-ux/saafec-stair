#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

MODE=double \
DOUBLE_MUTATION_LIST="$ROOT_DIR/examples/worked_examples/double_mutation_example/inputs/double_mutation_list.txt" \
PDB_DIR="$ROOT_DIR/examples/worked_examples/double_mutation_example/inputs/pdbs" \
RUN_NAME=worked_double_mutation \
OUT_ROOT="$ROOT_DIR/examples/worked_examples/double_mutation_example/run_outputs" \
CACHE_DEVICE="${CACHE_DEVICE:-cpu}" \
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}" \
bash "$ROOT_DIR/run_saafec_stair_inference.sh"
