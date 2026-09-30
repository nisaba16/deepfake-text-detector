#!/bin/bash
#SBATCH --job-name=textdet
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# textdet jobs. MODE picks the job, the other variables override its defaults.
#
#   MODE=eval    every detector on every dataset, same rows and metrics  -> evaluation_results/textdet
#   MODE=laya    the Laya Jev-like reference on the same rows (needs: pip install laya)
#   MODE=export  browser model folders -> $MODELS_DIR (copy them to deepfake-detection-38502/public/models)
#
#   sbatch --export=ALL,MODE=eval slurm_textdet.sh
#   sbatch --export=ALL,MODE=eval,EVAL_PART=test slurm_textdet.sh                    # the final numbers, once
#   sbatch --export=ALL,MODE=eval,DETECTORS="hf:tmr-roberta-raid julia" slurm_textdet.sh
#   sbatch --export=ALL,MODE=export,EXPORT_PYTHON=$PWD/.venv-export/bin/python slurm_textdet.sh

set -euo pipefail

PYTHON="${PYTHON:-$HOME/miniconda3/envs/env_esa/bin/python}"
if [ ! -x "$PYTHON" ]; then
  echo "Python not found at: $PYTHON" >&2
  exit 1
fi
MODE="${MODE:-eval}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-evaluation_results/textdet}"
EVAL_PART="${EVAL_PART:-select}"        # select: compare and choose; test: final numbers only
N_ROWS="${N_ROWS:-2000}"                # per dataset, balanced classes (and generators when a column exists)

# Saved detectors trained here (edit the paths to where the .pkl files are on the cluster)
SAVED="${SAVED:-saved_models/human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_last_ocsvm.pkl saved_models/human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_mean_std_l2norm_ocsvm.pkl saved_models/human_ai_microsoft_deberta-v3-large_embedding_layer23_mean_std_l2norm_lr.pkl saved_models/human_ai_Qwen_Qwen2.5-0.5B_embedding_layer16_last_l2norm_lr.pkl}"
SAVED_SPECS=""
for p in $SAVED; do
  if [ -f "$p" ]; then SAVED_SPECS="$SAVED_SPECS saved:$p"; else echo "Skipping missing saved detector: $p" >&2; fi
done

DETECTORS="${DETECTORS:-julia hf:bert-tiny-raid hf:e5-small-raid hf:tmr-roberta-raid hf:modernbert-raid-mage fastdetectgpt:Qwen/Qwen2.5-0.5B binoculars:Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-0.5B-Instruct ens:hf:tmr-roberta-raid+hf:e5-small-raid ens:hf:tmr-roberta-raid+fastdetectgpt:Qwen/Qwen2.5-0.5B $SAVED_SPECS}"
# mage_test is in distribution for modernbert-raid-mage, raid_sample for every hf:*-raid detector (see the catalog)
DATASETS="${DATASETS:-mercor_ai:data/mercor-ai/train.csv human_ai_sample mage_ood_gpt4 mage_ood_gpt4_para mage_test raid_sample}"

echo "Node: $(hostname) | mode: $MODE"
"$PYTHON" --version
"$PYTHON" -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| cuda', torch.cuda.is_available())"

case "$MODE" in
  eval)
    # Thresholds: set on the human texts of a 20% split for a 5% false-positive rate (no AI example needed);
    # AUROC and TPR@1%/5%FPR do not depend on it.
    "$PYTHON" scripts/evaluate_detectors.py \
      --detectors $DETECTORS \
      --datasets $DATASETS \
      --eval_part "$EVAL_PART" --n_rows "$N_ROWS" --stratified_sample \
      --optimize_threshold fpr0.05 --optimize_split 0.2 \
      --questions ai_generated human_written \
      --device "$DEVICE" --batch_size "${BATCH_SIZE:-8}" \
      --save_summary --save_predictions --output_dir "$OUT/$EVAL_PART"
    "$PYTHON" scripts/analyze_cross_dataset.py --input_dir "$OUT/$EVAL_PART" --metric "${METRIC:-tpr_at_0.01fpr}"
    ;;
  laya)
    "$PYTHON" scripts/evaluate_jev_baseline.py --backend laya \
      --datasets $DATASETS \
      --questions ai_generated ai_generated_ab human_written \
      --eval_part "$EVAL_PART" --n_rows "$N_ROWS" --stratified_sample \
      --optimize_threshold fpr0.05 --device "$DEVICE" --batch_size "${BATCH_SIZE:-16}" \
      --save_summary --save_predictions --output_dir "$OUT/$EVAL_PART"
    ;;
  export)
    # hf: and saved: exports use torch.onnx.export, which needs transformers<5 (requirements-export.txt)
    EXPORT_PYTHON="${EXPORT_PYTHON:-$PYTHON}"
    MODELS_DIR="${MODELS_DIR:-web_models}"
    mkdir -p "$MODELS_DIR"
    [ -f "$MODELS_DIR/index.json" ] || echo '[]' > "$MODELS_DIR/index.json"
    "$PYTHON" scripts/export_web_model.py julia --id julia1-jev --models_dir "$MODELS_DIR" \
      --name "Julia 1 · Jev-like zero-shot (demo)"
    for m in tmr-roberta-raid e5-small-raid bert-tiny-raid; do
      "$EXPORT_PYTHON" scripts/export_web_model.py "hf:$m" --id "$m" --models_dir "$MODELS_DIR" \
        --calibrate_fpr 0.05 --calibration_dataset mage_test
    done
    for p in $SAVED; do
      case "$p" in *MiniLM*) [ -f "$p" ] && "$EXPORT_PYTHON" scripts/export_web_model.py "saved:$p" \
        --models_dir "$MODELS_DIR" --calibrate_fpr 0.05 --calibration_dataset mage_test ;; esac
    done
    ls -la "$MODELS_DIR"/*
    ;;
  *)
    echo "Unknown MODE=$MODE (eval, laya, export)" >&2
    exit 1
    ;;
esac
