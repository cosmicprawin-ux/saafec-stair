#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

display_path() {
  local path="$1"
  local package_name
  package_name="$(basename "$ROOT_DIR")"
  if [[ "$path" == "$ROOT_DIR" ]]; then
    printf "%s\n" "$package_name"
  elif [[ "$path" == "$ROOT_DIR"/* ]]; then
    printf "%s/%s\n" "$package_name" "${path#"$ROOT_DIR"/}"
  elif [[ "$path" == "$(dirname "$ROOT_DIR")"/* ]]; then
    printf "%s\n" "${path#"$(dirname "$ROOT_DIR")"/}"
  else
    printf "%s\n" "$(basename "$path")"
  fi
}

CONDA_ENV_NAME="${CONDA_ENV_NAME:-SAAFEC_STAIR_inference}"
if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_PATH"
  elif [[ -x "$CONDA_ENV_PATH/bin/python" ]]; then
    export PATH="$CONDA_ENV_PATH/bin:$PATH"
    export LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:${LD_LIBRARY_PATH:-}"
  else
    echo "CONDA_ENV_PATH does not contain a usable Python environment: $CONDA_ENV_PATH" >&2
    exit 1
  fi
elif [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
  : # Respect the environment that is already active.
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "$CONDA_ENV_NAME"; then
    conda activate "$CONDA_ENV_NAME"
  else
    echo "Conda environment '$CONDA_ENV_NAME' was not found; using the active Python on PATH." >&2
  fi
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Python was not found. Activate the SAAFEC-STAIR environment and retry." >&2
  exit 1
fi

MODE="${MODE:-single}"
CACHE_DEVICE="${CACHE_DEVICE:-cpu}"
INFERENCE_DEVICE="${INFERENCE_DEVICE:-cpu}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SAPROT_MODEL_DIR="${SAPROT_MODEL_DIR:-$ROOT_DIR/assets/external/models/SaProt_650M_PDB}"
FOLDSEEK_BIN="${FOLDSEEK_BIN:-$ROOT_DIR/assets/external/bin/foldseek}"
PROTEINMPNN_CHECKPOINT="${PROTEINMPNN_CHECKPOINT:-$ROOT_DIR/assets/external/models/proteinmpnn/v_48_020.pt}"
PROTEINMPNN_SOURCE="${PROTEINMPNN_SOURCE:-$ROOT_DIR/assets/external/source/ThermoMPNN/protein_mpnn_utils.py}"
SINGLE_BASE="${SINGLE_CHECKPOINT_DIR:-$ROOT_DIR/assets/checkpoints/single_mutation/seeds}"
SINGLE_CHECKPOINTS=(
  "$SINGLE_BASE/seed_1337/best_head.pt"
  "$SINGLE_BASE/seed_2027/best_head.pt"
  "$SINGLE_BASE/seed_3407/best_head.pt"
)
DOUBLE_HEAD_CHECKPOINT="${DOUBLE_HEAD_CHECKPOINT:-$ROOT_DIR/assets/checkpoints/double_mutation/best_double_mutation_package.pt}"

python scripts/tools/check_installation.py --mode "$MODE"

table_sheet_for_input() {
  local input_table="$1"
  local xlsx_default="$2"
  if [[ -n "${TABLE_SHEET:-}" ]]; then
    printf "%s\n" "$TABLE_SHEET"
  elif [[ "$input_table" == *.xlsx || "$input_table" == *.xlsm || "$input_table" == *.xltx || "$input_table" == *.xltm ]]; then
    printf "%s\n" "$xlsx_default"
  else
    printf "text_input\n"
  fi
}

run_single() {
  local run_name="${RUN_NAME:-single_pdb}"
  local out_root="${OUT_ROOT:-$ROOT_DIR/output/$run_name}"
  local manifest_dir="$out_root/manifests"
  local default_mutation_list="$ROOT_DIR/examples/quickstart/single_mutation/single_mutation_list.txt"
  local single_mutation_list="${SINGLE_MUTATION_LIST:-${SPECIFIED_MUTATION_TABLE:-}}"
  local input_table="${INPUT_TABLE:-${single_mutation_list:-$default_mutation_list}}"
  local pdb_dir="${PDB_DIR:-$ROOT_DIR/examples/quickstart/single_mutation/pdbs}"

  if [[ -z "${INPUT_PDB:-}" && -z "${INPUT_TABLE:-}" && -z "$single_mutation_list" ]]; then
    single_mutation_list="$default_mutation_list"
  fi

  mkdir -p "$manifest_dir"
  if [[ -n "${INPUT_PDB:-}" && -z "${INPUT_TABLE:-}" ]]; then
    pdb_dir="$manifest_dir/input_pdbs"
    input_table="$manifest_dir/single_input.csv"
    python scripts/inference/prepare_single_pdb_input.py \
      --pdb "$INPUT_PDB" \
      --chain "${CHAIN:?Single-PDB inference requires CHAIN}" \
      --output-csv "$input_table" \
      --pdb-output-dir "$pdb_dir"
  elif [[ ! -f "$input_table" || ! -d "$pdb_dir" ]]; then
    echo "Single mode needs INPUT_PDB plus CHAIN, or SINGLE_MUTATION_LIST plus PDB_DIR." >&2
    echo "Mutation list: $input_table" >&2
    echo "Default PDB_DIR:     $pdb_dir" >&2
    exit 2
  fi

  python scripts/inference/validate_pdb_inputs.py --pdb-dir "$pdb_dir"

  local table_sheet
  table_sheet="$(table_sheet_for_input "$input_table" "refined_sorted_clean")"

  local features_root="$out_root/features"
  local embeddings_root="$features_root/saprot_embeddings"
  local proteinmpnn_root="$features_root/proteinmpnn_logits"
  local prediction_root="$out_root/predictions"

  echo "[single 1/4] Generating SaProt structure-aware embeddings"
  python scripts/cache/generate_saprot_structure_aware_cache.py \
    --dataset-csv "$input_table" \
    --pdb-dir "$pdb_dir" \
    --output-dir "$embeddings_root" \
    --saprot-model-dir "$SAPROT_MODEL_DIR" \
    --foldseek-bin "$FOLDSEEK_BIN" \
    --model-name SaProt \
    --table-sheet "$table_sheet" \
    --device "$CACHE_DEVICE"

  echo "[single 2/4] Generating ProteinMPNN logits"
  python scripts/cache/generate_proteinmpnn_logits_cache.py \
    --dataset-csv "$input_table" \
    --pdb-dir "$pdb_dir" \
    --output-dir "$proteinmpnn_root" \
    --reference-cache-dir "$embeddings_root/by_protein" \
    --proteinmpnn-checkpoint "$PROTEINMPNN_CHECKPOINT" \
    --proteinmpnn-source "$PROTEINMPNN_SOURCE" \
    --table-sheet "$table_sheet" \
    --device "$CACHE_DEVICE"

  echo "[single 3/4] Running three-seed single-mutation fusion inference"
  python scripts/inference/run_fusion_3seed_inference.py \
    --input-table "$input_table" \
    --table-sheet "$table_sheet" \
    --saprot-embeddings-dir "$embeddings_root/by_protein" \
    --proteinmpnn-cache-dir "$proteinmpnn_root/by_protein" \
    --checkpoint-dir "$SINGLE_BASE" \
    --output-dir "$prediction_root" \
    --request-name "$run_name" \
    --pdb-dir "$pdb_dir" \
    --write-visualizations \
    --device "$INFERENCE_DEVICE" \
    --num-workers "$NUM_WORKERS"

  if [[ -n "$single_mutation_list" ]]; then
    echo "[single specified] Writing specified single-mutation DDG predictions"
    python scripts/inference/write_specified_single_mutation_predictions.py \
      --specified-table "$single_mutation_list" \
      --matrix-dir "$prediction_root/predicted_DDG" \
      --output-dir "$out_root/specified_DDG_predictions"
  fi

  echo "[single 4/4] Single-mutation outputs"
  echo "  Feature caches:  $(display_path "$features_root")/"
  echo "  Matrix CSV/HTML: $(display_path "$prediction_root")/visualizations/<protein>/"
  echo "  Predicted DDG:   $(display_path "$prediction_root")/predicted_DDG/"
  if [[ -n "$single_mutation_list" ]]; then
    echo "  Specified DDG:   $(display_path "$out_root")/specified_DDG_predictions/"
  fi
}

run_double() {
  local input_table="${DOUBLE_MUTATION_LIST:-${INPUT_TABLE:-$ROOT_DIR/examples/quickstart/double_mutation/double_mutation_list.txt}}"
  local table_sheet
  table_sheet="$(table_sheet_for_input "$input_table" "refined_sorted")"
  local pdb_dir="${PDB_DIR:-$ROOT_DIR/examples/quickstart/double_mutation/pdbs}"
  local run_name="${RUN_NAME:-specified_double_mutations}"
  local out_root="${OUT_ROOT:-$ROOT_DIR/output/$run_name}"

  local manifest_dir="$out_root/manifests"
  local manifest_path="$manifest_dir/double_mutation_inference_manifest.csv"
  local pdb_link_dir="$manifest_dir/pdb_links"
  local features_root="$out_root/features"
  local embeddings_root="$features_root/saprot_embeddings"
  local proteinmpnn_root="$features_root/proteinmpnn_logits"
  local single_ddg_dir="$out_root/single_mutation_priors"
  local prediction_dir="$out_root/predictions"

  mkdir -p "$out_root"
  if [[ ! -f "$input_table" || ! -d "$pdb_dir" ]]; then
    echo "Double mode needs DOUBLE_MUTATION_LIST plus PDB_DIR." >&2
    echo "Mutation list: $input_table" >&2
    echo "Default PDB_DIR:     $pdb_dir" >&2
    exit 2
  fi

  python scripts/inference/validate_pdb_inputs.py --pdb-dir "$pdb_dir"

  echo "[double 1/5] Preparing specified-mutation manifest"
  python scripts/inference/prepare_double_mutation_cache_manifest.py \
    --input-table "$input_table" \
    --table-sheet "$table_sheet" \
    --pdb-dir "$pdb_dir" \
    --manifest-path "$manifest_path" \
    --pdb-link-dir "$pdb_link_dir"

  echo "[double 2/5] Generating SaProt structure-aware embeddings"
  python scripts/cache/generate_saprot_structure_aware_cache.py \
    --dataset-csv "$manifest_path" \
    --pdb-dir "$pdb_link_dir" \
    --output-dir "$embeddings_root" \
    --saprot-model-dir "$SAPROT_MODEL_DIR" \
    --foldseek-bin "$FOLDSEEK_BIN" \
    --model-name SaProt \
    --table-sheet "$table_sheet" \
    --device "$CACHE_DEVICE"

  echo "[double 3/5] Generating ProteinMPNN logits"
  python scripts/cache/generate_proteinmpnn_logits_cache.py \
    --dataset-csv "$manifest_path" \
    --pdb-dir "$pdb_link_dir" \
    --output-dir "$proteinmpnn_root" \
    --reference-cache-dir "$embeddings_root/by_protein" \
    --proteinmpnn-checkpoint "$PROTEINMPNN_CHECKPOINT" \
    --proteinmpnn-source "$PROTEINMPNN_SOURCE" \
    --table-sheet "$table_sheet" \
    --device "$CACHE_DEVICE"

  local single_checkpoint_args=()
  for checkpoint in "${SINGLE_CHECKPOINTS[@]}"; do
    single_checkpoint_args+=(--single-checkpoint "$checkpoint")
  done

  echo "[double 4/5] Exporting implicit single-mutation prior matrices"
  python scripts/inference/export_saprot_local_contact_single_ddg_for_double_mutation.py \
    "${single_checkpoint_args[@]}" \
    --embeddings-dir "$embeddings_root/by_protein" \
    --proteinmpnn-cache-dir "$proteinmpnn_root/by_protein" \
    --output-dir "$single_ddg_dir" \
    --baseline-name saafec_stair_single_mutation_3seed_ensemble \
    --input-table "$input_table" \
    --table-sheet "$table_sheet" \
    --device "$INFERENCE_DEVICE"

  echo "[double 5/5] Running double-mutation inference head"
  python scripts/inference/run_double_mutation_inference.py \
    --input-table "$input_table" \
    --table-sheet "$table_sheet" \
    --embeddings-dir "$embeddings_root/by_protein" \
    --single-ddg-dir "$single_ddg_dir" \
    --proteinmpnn-cache-dir "$proteinmpnn_root/by_protein" \
    --checkpoint "$DOUBLE_HEAD_CHECKPOINT" \
    --output-dir "$prediction_dir" \
    --device "$INFERENCE_DEVICE" \
    --num-workers "$NUM_WORKERS"

  echo "Feature caches: $(display_path "$features_root")/"
  echo "Double-mutation predictions: $(display_path "$prediction_dir")/mutation_DDG_predictions.csv"
  echo "Implicit single-mutation prior matrices: $(display_path "$single_ddg_dir")"
}

case "$MODE" in
  single) run_single ;;
  double) run_double ;;
  *) echo "MODE must be single or double, got: $MODE" >&2; exit 2 ;;
esac
