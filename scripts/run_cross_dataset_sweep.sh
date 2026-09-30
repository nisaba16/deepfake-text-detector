#!/usr/bin/env bash
# Cross-dataset sweep:
# - Train on Human vs AI (~4k rows) with various backbones/layers/pooling/classifiers
# - Evaluate each saved model on Mercor AI (train.csv) where labels are known
# - Results are saved under evaluation_results/cross_dataset
#
# Usage examples:
#   bash scripts/run_cross_dataset_sweep.sh
#   PYTHON=/path/to/env/bin/python bash scripts/run_cross_dataset_sweep.sh
#   (inside Slurm batch) srun bash scripts/run_cross_dataset_sweep.sh
set -euo pipefail

# Allow overriding python interpreter
PYTHON_BIN="${PYTHON:-python}"

# Paths
HUMAN_AI_CSV=${HUMAN_AI_CSV:-data/data_human/AI_Human.csv}
MERCOR_CSV=${MERCOR_CSV:-data/mercor-ai/train.csv}
SAVE_DIR=${SAVE_DIR:-saved_models}
EVAL_DIR=${EVAL_DIR:-evaluation_results/cross_dataset}
DEVICE=${DEVICE:-cuda:0}
# Default to a smaller batch to reduce peak memory; override with BATCH_SIZE env
BATCH_SIZE=${BATCH_SIZE:-4}
N_ROWS=${N_ROWS:-4000}   # approx 4k texts
MEM_EFF=${MEM_EFF:---memory_efficient}    # default to memory-efficient extraction (set to empty to disable)
DELETE_AFTER_EVAL=${DELETE_AFTER_EVAL:-}  # set to any non-empty value to delete saved models after evaluation

mkdir -p "$SAVE_DIR" "$EVAL_DIR"

# Search space (edit as needed)
MODELS=(
  "sentence-transformers/all-MiniLM-L6-v2"
)

# Fallback layers if no per-model override provided
LAYER_DEFAULTS=(3 4 5)

# ["Qwen/Qwen2.5-0.5B"]="16 20 22 24"
#  ["Qwen/Qwen3-Embedding-0.6B"]="10 20"
#  ["Qwen/Qwen3-8B"]="10 15 20 25 30 -1"

POOLINGS=(mean last mean_std)
#POOLINGS=(statistical)
#CLFS=(lr svm)
CLFS=(ocsvm elliptic)
# Normalization sweep: 0 = no normalization, 1 = L2 normalize token embeddings before pooling
NORMALIZES=(0 1)
# Per-model layer overrides (edit entries as needed)
declare -A LAYERS_MAP=(
 ["sentence-transformers/all-MiniLM-L6-v2"]="0 1 2 3 4 5"
)

#  ["sentence-transformers/use-cmlm-multilingual"]="0 1 2 3 4 5 6 7 8 9 10 11"
# ["Qwen/Qwen3-1.7B"]="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23 24 25 26 27"
# ["sentence-transformers/all-MiniLM-L6-v2"]="0 1 2 3 4 5 6 7 8 9 10 11"
#["sentence-transformers/paraphrase-multilingual-mpnet-base-v2"]="1 2 3 4 5 6 7 8 9 10 11"
#["sentence-transformers/all-mpnet-base-v2"]="1 2 3 4 5 6 7 8 9 10 11"
#  ["microsoft/deberta-v3-large"]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
# ["Qwen/Qwen2.5-0.5B"]="5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 23"
#["Qwen/Qwen3-Embedding-4B"]="1 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33"

# ["microsoft/deberta-v3-large"]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
#  ["Qwen/Qwen3-Embedding-0.6B"]="10 20"
#  ["Qwen/Qwen3-8B"]="10 15 20 25 30 -1"
#
#["sentence-transformers/all-distilroberta-v1"]
#
# Helper: sanitize model name to match saved filename convention
sanitize() {
  echo "$1" | sed 's#[/ ]#_#g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

echo "Python: $($PYTHON_BIN --version 2>/dev/null || echo not-found)"
echo "Device: ${DEVICE} | BatchSize: ${BATCH_SIZE} | N_ROWS: ${N_ROWS}"

echo "==> Training on Human vs AI (~${N_ROWS}) and evaluating on Mercor AI"

for model in "${MODELS[@]}"; do
  model_tag="$(sanitize "$model")"
  # Resolve layer list for this model (override if present)
  if [[ -n "${LAYERS_MAP[$model]+set}" && -n "${LAYERS_MAP[$model]}" ]]; then
    IFS=' ' read -r -a LAYERS_THIS <<< "${LAYERS_MAP[$model]}"
  else
    LAYERS_THIS=("${LAYER_DEFAULTS[@]}")
  fi

  for layer in "${LAYERS_THIS[@]}"; do
    for pool in "${POOLINGS[@]}"; do
      for norm in "${NORMALIZES[@]}"; do
        for clf in "${CLFS[@]}"; do
        echo "\n----\nModel: ${model} | layer=${layer} | pool=${pool} | norm=${norm} | clf=${clf}"

        # Construct expected artifact names upfront
        BASE_NAME="human_ai_${model_tag}_embedding_layer${layer}_${pool}"
        if [[ "$norm" == "1" ]]; then
          BASE_NAME+="_l2norm"
        fi
        BASE_NAME+="_${clf}"
        SUMMARY_PATH="${EVAL_DIR}/cross_dataset_summary_${BASE_NAME}.csv"
        MODEL_PATH="${SAVE_DIR}/${BASE_NAME}.pkl"

        # Skip entirely if the summary already exists
        if [ -f "$SUMMARY_PATH" ]; then
          echo "⏭️  Skipping: summary already exists at ${SUMMARY_PATH}"
          continue
        fi

        # For statistical pooling, constrain covariance dimensionality to avoid RAM OOM.
        # We set COV_MAX_HIDDEN override to 0 by default for 'statistical' so the extractor
        # uses diagonal variance only (feature dim ~= hidden size). You can override by exporting
        # COV_MAX_HIDDEN_OVERRIDE to a small value (e.g., 256/384) if you want upper-tri flattening.
        EXTRA_ENV=""
        if [[ "$pool" == "statistical" ]]; then
          EXTRA_ENV="COV_MAX_HIDDEN=${COV_MAX_HIDDEN_OVERRIDE:-0}"
          echo "Applying ${EXTRA_ENV} for statistical pooling to prevent large covariance features."
        fi

        # Build env prefix for commands (use env so name=value isn't treated as a command)
        ENV_PREFIX=(env)
        if [[ -n "$EXTRA_ENV" ]]; then
          ENV_PREFIX+=("$EXTRA_ENV")
        fi

        # Train only if the saved model doesn't already exist
        if [ ! -f "$MODEL_PATH" ]; then
          echo "Training (no existing model found at ${MODEL_PATH})"
          "${ENV_PREFIX[@]}" "$PYTHON_BIN" scripts/train_and_save_detector.py \
            --model_name "$model" \
            --analysis_type embedding \
            --classifier_type "$clf" \
            --layer "$layer" \
            --pooling "$pool" \
            --dataset_name human_ai \
            --train_data_path "$HUMAN_AI_CSV" \
            --text_column text \
            --label_column generated \
            --n_rows "$N_ROWS" \
            --batch_size "$BATCH_SIZE" \
            --device "$DEVICE" \
            --stratified_sample \
            ${MEM_EFF} \
            $( [[ "$norm" == "1" ]] && echo "--normalize" )
        else
          echo "Reusing existing saved model: ${MODEL_PATH}"
        fi

        # Resolve actual model path (support slight naming variations via glob)
        if [ ! -f "$MODEL_PATH" ]; then
          echo "Saved model not found at ${MODEL_PATH}, trying glob..."
          MODEL_PATH=$(ls -t ${SAVE_DIR}/${BASE_NAME}.pkl ${SAVE_DIR}/${BASE_NAME}_*.pkl 2>/dev/null | grep -v '_metadata' | head -n 1 || true)
        fi
        if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
          echo "!! Skipping evaluation: could not find saved model for ${model} layer=${layer} pool=${pool} clf=${clf}" >&2
          continue
        fi
        echo "Using model: $MODEL_PATH"

        # Evaluate on the 'select' half of Mercor AI train (will write summary to $SUMMARY_PATH);
        # the 'test' half stays untouched for the final score of the chosen config.
        # Use threshold optimization to fix domain shift issues with SVM
        "${ENV_PREFIX[@]}" "$PYTHON_BIN" scripts/cross_dataset_evaluation.py \
          --model_path "$MODEL_PATH" \
          --datasets mercor_ai:"$MERCOR_CSV" \
          --device "$DEVICE" \
          --eval_part select \
          --optimize_threshold f1 \
          --optimize_split 0.2 \
          --save_summary \
          --output_dir "$EVAL_DIR"

        # Optionally delete saved model after evaluation to save disk space
        if [[ -n "$DELETE_AFTER_EVAL" ]]; then
          if rm -f -- "$MODEL_PATH"; then
            echo "Deleted saved model: $MODEL_PATH"
          else
            echo "Warning: failed to delete $MODEL_PATH" >&2
          fi
        fi
        done
      done
    done
  done
done

echo "\n✅ Cross-dataset sweep completed. Summaries in: $EVAL_DIR"
