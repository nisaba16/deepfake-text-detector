# Train & Evaluate Guide for Deepfake Text Detectors

This guide focuses on training classifiers on a dataset and loading/evaluating them later. A short Mercor challenge section is kept for convenience.

## 📁 Project Structure

The main scripts are located in the `scripts/` directory:

- **`main_submission_mercor.py`**: Mercor AI challenge submission pipeline  
- **`parameter_sweep_mercor.py`**: Hyperparameter optimization with k-fold CV
- **`train_and_save_detector.py`**: Train models and save for later use
- **`load_and_evaluate.py`**: Load saved models and evaluate on datasets
- **`cross_dataset_evaluation.py`**: Cross-dataset generalization evaluation

## 🚀 Core Analysis Types

All scripts support three main analysis types:

1. **`embedding`**: Layer-wise embeddings from transformer models (requires `--layer` and `--pooling`)
2. **`perplexity`**: Text perplexity scores
3. **`phd`**: Persistent homological dimensions (requires `--layer`)

## 📊 Classifier Types

- **`svm`**: Support Vector Machine (fast, good baseline)
- **`lr`**: Logistic Regression (fastest, interpretable)
- **`xgb`**: XGBoost (ensemble method)
- **`neural`**: Neural network (requires CUDA, most powerful)

---

## 🔄 Pooling strategies (embedding)

When using `--analysis_type embedding`, set `--pooling` to control how token embeddings are aggregated per layer:

- mean: Average over tokens. Stable baseline.
- max: Element-wise maximum over tokens. Highlights salient features.
- last: Use the last token’s vector. Works well for encoder-style sentence embeddings or causal models with EOS.
- attn_mean: Attention-weighted mean; gracefully falls back to uniform mean if attentions aren’t available (e.g., flash attention).
- mean_std: Concatenate mean and standard deviation over tokens. Doubles dimensionality (2 × hidden_size) and often improves robustness by capturing dispersion.
- statistical (aliases: covariance, cov): Covariance pooling. Flattens the upper triangle of the token-embedding covariance matrix for the chosen layer, capturing style/coherence patterns (useful for authorship-like signals).
  - Dimensionality warning: hidden_size × (hidden_size + 1) / 2. To keep things practical on large models, we cap via env var `COV_MAX_HIDDEN` (default 1024). If hidden_size > cap, it falls back to the diagonal (per-dimension variances).
  - Recommended layers: mid-to-deep layers to balance syntax/semantics (e.g., for 32-layer backbones, try 20–26 first).

Notes:
- All pooling works per-layer. Select the layer with `--layer`.
- For embeddings, the downstream detector applies StandardScaler + PCA(0.95) by default, so higher-dimensional poolings are reduced automatically.

---

## 2. Mercor AI Challenge Submission (`main_submission_mercor.py`)

Generate submissions for Mercor AI cheating detection challenge.

### Basic Usage

```bash
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 21 \
  --pooling mean \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_mercor.csv \
  --device cuda:0
```

### Examples by Model Size

#### Small Models (Fast Training)
```bash
# DistilRoBERTa
python scripts/main_submission_mercor.py \
  --model_name "sentence-transformers/all-distilroberta-v1" \
  --analysis_type embedding \
  --layer 5 \
  --pooling mean \
  --classifier_type neural \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_distilroberta.csv \
  --device cuda:0 \
  --batch_size 16

# Multilingual model
python scripts/main_submission_mercor.py \
  --model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
  --analysis_type embedding \
  --layer 4 \
  --pooling max \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_multilingual.csv \
  --device cuda:0 \
  --batch_size 12
```

#### Large Models (Better Performance)
```bash
# Qwen 8B model
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --layer 30 \
  --pooling attn_mean \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_qwen8b_attnmean.csv \
  --device cuda:0 \
  --batch_size 4 \
  --memory_efficient

# Qwen 8B with mean+std pooling
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --layer 26 \
  --pooling mean_std \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_qwen8b_meanstd.csv \
  --device cuda:0 \
  --batch_size 4 \
  --memory_efficient

# Qwen 8B with covariance pooling (consider mid/deep layers)
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --layer 22 \
  --pooling statistical \
  --classifier_type lr \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_qwen8b_cov.csv \
  --device cuda:0 \
  --batch_size 4 \
  --memory_efficient

# Llama 8B model
python scripts/main_submission_mercor.py \
  --model_name "meta-llama/Llama-3.1-8B" \
  --analysis_type embedding \
  --layer 26 \
  --pooling attn_mean \
  --classifier_type svm \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_llama8b.csv \
  --device cuda:1 \
  --batch_size 4
```

#### Different Analysis Types
```bash
# Perplexity-based detection
python scripts/main_submission_mercor.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type perplexity \
  --classifier_type lr \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_perplexity.csv \
  --device cuda:1 \
  --batch_size 8

# PHD-based detection
python scripts/main_submission_mercor.py \
  --model_name "FacebookAI/roberta-base" \
  --analysis_type phd \
  --layer 8 \
  --classifier_type lr \
  --train_csv data/mercor-ai/train.csv \
  --test_csv data/mercor-ai/test.csv \
  --output_path submission_phd.csv \
  --device cuda:0 \
  --batch_size 8
```

---

## 3. Parameter Sweep (`parameter_sweep_mercor.py`)

Systematic hyperparameter optimization using k-fold cross-validation.

### Quick Testing Sweep
```bash
python scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "sentence-transformers/all-distilroberta-v1" \
  --layers 3 5 \
  --pooling_types "mean" "mean_std" "statistical" \
  --use_pca_options true \
  --normalize_options true \
  --classifier_types svm \
  --cv_folds 3 \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/quick_sweep.json
```

### Comprehensive Sweep (Multiple Models)
```bash
nohup python > embed06B_binary.out scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-Embedding-0.6B" \
  --layers 1 2 5 10 15 20 25 26 27 \
  --pooling_types "mean" "max" "last" \
  --use_pca_options true false \
  --normalize_options true false \
  --classifier_types lr svm xgb neural \
  --cv_folds 5 \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/comprehensive_sweep_all06b_binary.json \
  --memory_efficient &

nohup python > embed06B_binary.out scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-Embedding-0.6B" \
  --layers 1 2 5 10 15 20 25 26 27 \
  --pooling_types "mean" "max" "last"  \
  --use_pca_options false \
  --normalize_options false \
  --classifier_types ocsvm iforest \
  --cv_folds 5 \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/sweep_binary.json &
```

### Background Execution (Long Sweeps)
```bash
nohup python scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-8B" "meta-llama/Llama-3.1-8B" \
  --layers 1 5 10 15 20 25 30 -1 -2 -3 \
  --pooling_types "mean" "max" \
  --use_pca_options true false \
  --normalize_options true false \
  --classifier_types lr svm xgb neural \
  --cv_folds 5 \
  --device cuda:1 \
  --batch_size 4 \
  --output_path results/long_sweep.json > sweep.log 2>&1 &
```

### Focused Layer Search
```bash
# Find best layer for specific model
python scripts/parameter_sweep_mercor.py \
  --train_csv data/mercor-ai/train.csv \
  --models "Qwen/Qwen3-Embedding-8B" \
  --layers 1 5 10 15 20 25 30 32 34 35 -1 -2 -3 \
  --pooling_types "mean" \
  --use_pca_options true \
  --normalize_options false \
  --classifier_types lr \
  --cv_folds 5 \
  --device cuda:0 \
  --batch_size 4 \
  --output_path results/layer_search_qwen.json
```

---

## 4. Train and Save Models (`train_and_save_detector.py`)

Train models on one dataset and save for later cross-dataset evaluation.

### Train on Human vs AI Dataset
sentence-transformers/all-distilroberta-v1
Qwen/Qwen2.5-0.5B
Qwen/Qwen3-8B

```bash
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 16 \
  --pooling mean_std \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --n_rows 8000 \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient \
  --stratified_sample \
  --log_memory
```

```bash
# Mean+Std pooling
python scripts/train_and_save_detector.py \
  --model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 20 \
  --pooling mean_std \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient
```

```bash
# Covariance pooling (second-order); recommended mid/deep layers
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 22 \
  --pooling statistical \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient
```

```bash
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type iforest \
  --layer 22 \
  --pooling mean \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient \
  --sample_frac 0.01 \
  --log_memory


python scripts/train_and_save_detector.py \
  --analysis_type tfidf \
  --classifier_type svm \
  --dataset_name human_ai \
  --train_data_path data/data_human/AI_Human.csv \
  --text_column text \
  --label_column generated \
  --tfidf_max_features 50000 \
  --tfidf_ngram_min 1 \
  --tfidf_ngram_max 2 \
  --tfidf_min_df 2 \
  --tfidf_max_df 0.9 \
  --tfidf_stop_words english \
  --svd_components 500 \
  --validation_split 0.2 \
  --sample_frac 0.1
```

### Train on Mercor AI Dataset
```bash
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 22 \
  --pooling mean \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --batch_size 8 \
  --device cuda:0
```

### Train on DAIGT v2 Dataset

You can train directly on the DAIGT v2 dataset stored under `data/daigt_v2/`. The loader accepts either a directory (it will auto-pick a `train_v2_*.csv`) or a direct path to a CSV. Expected columns are `text` and `label` (0=human, 1=AI).

```bash
# Directory input (auto-detect CSV inside)
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type lr \
  --layer 20 \
  --pooling mean \
  --dataset_name daigtv2 \
  --train_data_path data/daigt_v2 \
  --batch_size 8 \
  --device cuda:0 \
  --memory_efficient \
  --sample_frac 0.05

# Explicit CSV input
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type embedding \
  --classifier_type svm \
  --layer 22 \
  --pooling mean \
  --dataset_name daigtv2 \
  --train_data_path data/daigt_v2/train_v2_drcat_02.csv \
  --batch_size 8 \
  --device cuda:0
```

Notes:
- DAIGT v2 loader uses fixed columns (`text`, `label`). You do not need to set `--text_column` or `--label_column` for this dataset.
- If you pass a generic CSV with `--dataset_name generic`, you can still specify `--text_column`/`--label_column` manually.


### Training with Sampling (Large Datasets)
```bash
# Train on 50% of data for faster experiments
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen3-8B" \
  --analysis_type embedding \
  --classifier_type neural \
  --layer 30 \
  --pooling mean \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --sample_frac 0.5 \
  --batch_size 4 \
  --device cuda:1
```

### Different Analysis Types
```bash
# Perplexity-based model
python scripts/train_and_save_detector.py \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --analysis_type perplexity \
  --classifier_type lr \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --device cuda:0

# PHD-based model
python scripts/train_and_save_detector.py \
  --model_name "FacebookAI/roberta-base" \
  --analysis_type phd \
  --layer 6 \
  --classifier_type lr \
  --dataset_name mercor_ai \
  --train_data_path data/mercor-ai/train.csv \
  --text_column answer \
  --label_column is_cheating \
  --device cuda:0
```

---

## 5. Load and Evaluate Models (`load_and_evaluate.py`)

Evaluate saved models on different datasets.

### Basic Evaluation
```bash
python scripts/load_and_evaluate.py \
  --model_path saved_models/human_ai_Qwen_Qwen2.5-0.5B_embedding_layer22_mean_svm_metadata.pkl \
  --dataset_name mercor_ai \
  --data_path data/mercor-ai/train.csv \
  --device cuda:0 \
  --save_predictions \
  --output_dir evaluation_results
```

### Evaluate Multiple Models (Batch)
```bash
# Evaluate all saved models on mercor dataset
for model in saved_models/*.pkl; do
    echo "Evaluating $model"
    python scripts/load_and_evaluate.py \
        --model_path "$model" \
        --dataset_name mercor_ai \
        --data_path data/mercor-ai/train.csv \
        --device cuda:0 \
        --save_predictions \
        --output_dir evaluation_results
done
```

### Generic Dataset Evaluation
```bash
python scripts/load_and_evaluate.py \
  --model_path saved_models/custom_model.pkl \
  --dataset_name generic \
  --data_path data/custom_dataset.csv \
  --text_column "text_content" \
  --label_column "is_fake" \
  --device cuda:0
```

---

## 6. Cross-Dataset Evaluation (`cross_dataset_evaluation.py`)

Test model generalization across different datasets.

### Single Model, Multiple Datasets
```bash
python scripts/cross_dataset_evaluation.py \
  --model_path saved_models/daigtv2_Qwen_Qwen2.5-0.5B_embedding_layer20_mean_lr.pkl \
  --datasets mercor_ai:data/mercor-ai/train.csv \
  --device cuda:0 \
  --save_summary \
  --output_dir evaluation_results
```

### Multiple Models, Cross-Dataset
```bash
# Compare different models across datasets
for model in saved_models/human_ai_*.pkl; do
    echo "Cross-evaluating $model"
    python scripts/cross_dataset_evaluation.py \
        --model_path "$model" \
        --datasets mercor_ai:data/mercor-ai/train.csv \
        --device cuda:0 \
        --save_summary \
        --output_dir evaluation_results/cross_dataset
done
```

### Selecting configs without inflating the final score
Picking the best of many configs on the rows you report on overstates the result. `--eval_part` splits each evaluation set in two fixed, stratified halves (`--eval_test_frac`, default 0.5): compare configs on `select` (what `run_cross_dataset_sweep.sh` does), then score only the chosen config on `test`, in its own output directory:

```bash
python scripts/cross_dataset_evaluation.py \
  --model_path saved_models/<chosen_config>.pkl \
  --datasets mercor_ai:data/mercor-ai/train.csv \
  --eval_part test \
  --optimize_threshold f1 \
  --save_summary \
  --output_dir evaluation_results/final_test
```

With `--optimize_threshold`, the threshold is chosen on `--optimize_split` of the rows and the metrics are computed on the other rows only (`n_scored` in the summary). ROC-AUC does not depend on the threshold, which is why `analyze_cross_dataset.py` ranks by it by default.

---

## 7. Jev-like Zero-Shot Baseline (`evaluate_jev_baseline.py`)

A reference point that needs no training. A "System 1" decision model, the Jev family (TypeSafe's hosted Jev and
its open counterparts), is asked a yes/no (`noul`) question per text: *"Was this text generated by an AI language
model rather than written by a human?"*. Its P(true) is used as P(fake). It takes one encoder forward pass per
text and question, with no generation. Two backends:

| `--backend` | Model | Size | Notes |
|---|---|---|---|
| `julia` (default) | [Julia 1](https://huggingface.co/SupersonicLabs/Julia-1), mmBERT-small | 144M, 550 MiB fp32 | Runs natively in `textdet` (no extra package, transformers>=5); reads up to 8192 tokens (default 1024). It is also shipped to the browser (`export_web_model.py julia`). |
| `laya` | [Laya](https://github.com/NandhaKishorM/laya), ModernBERT-large | 421M, 804 MiB fp16 | `pip install laya`; reads 512 tokens (`--subfolder multilingual` for mmBERT-base, up to 8192). Kept for comparison. |

The script is `evaluate_detectors.py` with one detector. Any dataset spec works (see `textdet/data/catalog.py`):
`name:path` for your CSVs, a catalog name such as `mage_ood_gpt4` or `raid_sample`, or `name:any/table.jsonl`
with `--text_column/--label_column/--human_values`. It applies the same `--threshold` / `--optimize_threshold`
code (`textdet/evaluation/metrics.py`) as the other scripts. It writes
`cross_dataset_summary_zeroshot_<checkpoint>_jev_<question>.csv`, so the trained detectors and the baseline are
scored identically.

```bash
# Same rows and protocol as run_cross_dataset_sweep.sh (Mercor 'select' half, F1 threshold on a 20% split)
python scripts/evaluate_jev_baseline.py \
  --datasets mercor_ai:data/mercor-ai/train.csv mage_ood_gpt4 \
  --questions ai_generated human_written \
  --device cuda:0 \
  --eval_part select \
  --optimize_threshold f1 \
  --optimize_split 0.2 \
  --save_summary \
  --save_predictions \
  --output_dir evaluation_results/cross_dataset

# The Laya reference on the same rows
python scripts/evaluate_jev_baseline.py --backend laya --datasets mercor_ai:data/mercor-ai/train.csv mage_ood_gpt4 \
  --questions ai_generated ai_generated_ab human_written --eval_part select --save_summary \
  --output_dir evaluation_results/cross_dataset

# Rank it next to the sweep results (family "zeroshot"; ranked by roc_auc)
python scripts/analyze_cross_dataset.py \
  --input_dir evaluation_results/cross_dataset
```

For the final comparison, score the baseline on `--eval_part test` like the chosen detector (see "Selecting configs without inflating the final score").

Question presets (`--questions`, several are also averaged into `ensemble_mean`):
- `ai_generated`: the question above, options described as "generated by an AI model such as ChatGPT" / "written by a human".
- `ai_generated_ab`: the same question with the options shown as `A`/`B`, because Laya's words `true`/`false` can make the model follow the label rather than the text (Laya only; Julia shows the descriptions and ignores labels).
- `human_written`: the opposite polarity (P(fake) = 1 - P(true)), which cancels a bias towards answering "true".
- Your own wording: `--instructions "..." --true_criterion "..." --false_criterion "..."` (added as `custom`; its `true` answer must mean AI-generated).

Notes:
- ROC-AUC and TPR@FPR are computed on unrounded log-odds (Laya rounds P(true) to 4 decimals, which ties confident answers); thresholded metrics use P(true).
- Julia 1 zero-shot is at or below chance on MAGE's GPT-4 sets and flags most human texts (see `docs/TEXT_DETECTION_REVIEW.md`): it is a sanity baseline, not a detector.
- Quick check before a full run: add `--n_rows 200 --stratified_sample`.
- Zero-shot numbers depend on the wording, so each summary stores the checkpoint, its revision and the exact question.

## 8. Any detector on any dataset (`evaluate_detectors.py`)

```bash
python scripts/evaluate_detectors.py \
  --detectors julia hf:e5-small-raid hf:tmr-roberta-raid fastdetectgpt:HuggingFaceTB/SmolLM2-135M \
              saved:saved_models/human_ai_sentence-transformers_all-MiniLM-L6-v2_embedding_layer2_last_ocsvm.pkl \
              "ens:hf:e5-small-raid+hf:tmr-roberta-raid" \
  --datasets mage_ood_gpt4 mage_ood_gpt4_para mercor_ai:data/mercor-ai/train.csv raid_sample \
  --n_rows 1000 --stratified_sample --balance_column model \
  --optimize_threshold fpr0.05 --save_summary --save_predictions --output_dir evaluation_results/cross_dataset
```

- `--optimize_threshold fpr0.05` sets the threshold on the human texts of a 20% split for a 5% false-positive
  rate, and scores the other 80%. That is the realistic deployment: calibrate on human texts of your domain.
- Datasets with a generator/domain column also get `breakdown_*.csv` (TPR per generator at a global 5% FPR).
- `raid_sample` streams RAID's 11 GB train.csv: RAID-trained detectors (`hf:*-raid`) have seen those rows.

## 9. Browser models (`export_web_model.py`)

The text counterpart of the image pipeline: one folder per model in `deepfake-detection-38502/public/models/`
(`config.json`, `model.onnx`, `tokenizer.json`), added to `index.json`, with a parity check. The folder is run
exactly as the page runs it, and compared with the Python detector.

```bash
python scripts/export_web_model.py julia --id julia1-jev                  # Julia 1, question baked into the graph
.venv-export/bin/python scripts/export_web_model.py hf:e5-small-raid       # transformers<5: requirements-export.txt
.venv-export/bin/python scripts/export_web_model.py saved:saved_models/<detector>.pkl   # embedding + LR / OCSVM
python scripts/evaluate_detectors.py --detectors web:../deepfake-detection-38502/public/models/e5-small-raid \
  --datasets mage_ood_gpt4                                                   # evaluate the shipped file itself
```

---

## 🔧 Common Parameters

### Model Selection
```bash
# Small/Fast models
--model_name "microsoft/deberta-v3-large"
--model_name "sentence-transformers/all-distilroberta-v1"
--model_name "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
--model_name "Qwen/Qwen2.5-0.5B"

# Large/Powerful models  
--model_name "Qwen/Qwen3-8B"
--model_name "meta-llama/Llama-3.1-8B"
--model_name "Qwen/Qwen3-Embedding-8B"
--model_name "sentence-transformers/all-mpnet-base-v2"

# Specialized models
--model_name "FacebookAI/roberta-base"  # Good for PHD analysis
```

### Layer Selection Guidelines
```bash
# Small models (0.5B-1B parameters): layers 1-24
--layer 1     # Early features
--layer 12    # Middle layer
--layer 22    # Late layer
--layer -1    # Final layer

# Large models (7B-8B parameters): layers 1-32
--layer 1     # Early features  
--layer 15    # Middle layer
--layer 30    # Late layer
--layer -1    # Final layer
```

### Hardware Optimization
```bash
# GPU memory limited
--batch_size 4 --device cuda:0

# Multiple GPUs
--device cuda:1  # Use second GPU

# CPU only
--device cpu --batch_size 1

# Large memory available
--batch_size 16 --device cuda:0
```

### Output Management
```bash
# Background execution with logging
nohup python script.py [args] > output.log 2>&1 &

# Check background job
tail -f output.log

# Monitor GPU usage
nvidia-smi -l 1
```

---

## 📈 Results Analysis

### Parameter Sweep Results
```bash
# View best configurations
python -c "
import json
import pandas as pd
with open('results/sweep.json', 'r') as f:
    data = json.load(f)
df = pd.DataFrame(data)
print(df.nlargest(10, 'roc_auc_mean')[['model_name', 'layer', 'pooling', 'classifier_type', 'roc_auc_mean']])
"
```

### Submission Files
All submission scripts generate CSV files with the format:
- `submission_*.csv` for competition submissions
- Contains pair predictions in the required format

### Evaluation Results
- Saved in `evaluation_results/` directory
- Includes metrics (accuracy, F1, ROC-AUC) and predictions
- Cross-dataset summaries in CSV format

---

## 💡 Tips and Best Practices

### 1. Start Small
```bash
# Test with small model first
--model_name "sentence-transformers/all-distilroberta-v1" --batch_size 16
```

### 2. Parameter Sweeps
```bash
# Use fewer CV folds for initial exploration
--cv_folds 3

# Use more folds for final evaluation
--cv_folds 5
```

### 3. GPU Memory Management
```bash
# For large models, reduce batch size
--batch_size 4

# Clear memory between runs
python -c "import torch; torch.cuda.empty_cache()"
```

### 4. Reproducibility
```bash
# Always set random state
--random_state 42
```

### 5. Monitoring Long Jobs
```bash
# Use tmux/screen for long jobs
tmux new-session -d 'python script.py [args]'

# Monitor with tail
tail -f nohup.out
```