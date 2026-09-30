# AI-text detection: critical review, better data, and a lightweight plan

*September 2026. It covers this repository's baselines, the published state of the art, the datasets to use,
and what can ship client-side now without training. Local numbers come from `scripts/evaluate_detectors.py`
on a CPU (400 balanced texts per set, 4 detectors so far). They are small samples, meant for ranking detectors,
not for publication. The full comparison is `slurm_textdet.sh` (`MODE=eval`).*

## TL;DR

1. **The historical protocol cannot show whether a detector generalizes.** Detectors were trained on one Kaggle
   essay corpus (`AI_Human.csv`, 10k rows, 10% AI), selected on one target (Mercor), and ranked by ROC-AUC. That
   covers two domains, a few old generators and no attacks, and it misses the failure that matters: flagging
   humans.
2. **The Jev-like zero-shot baseline is not a detector.** Julia 1 scores at or below chance on MAGE's GPT-4 sets
   (AUROC 0.44–0.53) and flags 64–74% of human texts; in spot checks it also answered "yes" to "is it
   human?". Keep it as a sanity floor.
   Laya can still be run for comparison.
3. **Off-the-shelf RAID-trained classifiers are the easy win.** They rank unseen GPT-4 text well (tmr RoBERTa-base
   AUROC 0.975, e5-small 0.955). Out of the box they are overconfident on human text from new domains: at
   P = 0.5 they flag 39–46% of MAGE's humans, and tmr flags 98.5% of the human student essays.
4. **Calibration fixes most of that, with no training.** Setting the threshold on human texts brings e5-small's
   false-positive rate on the GPT-4 set from 46% to 2.5%. The browser export bakes it in with
   `--calibrate_fpr 0.05`.
5. **Next step (training, later):** a MELD-style small encoder (Ettin-32M/68M) trained on RAID + MAGE + M4GT +
   DetectRL with attack augmentation, and distilled from MELD. Evaluate it at TPR@1%FPR on held-out domains and
   generators.

## 1. Critical review of the baselines in this repository

### 1.1 Data

| Issue | Why it matters |
|---|---|
| **One training corpus.** `AI_Human.csv` (Kaggle, 487k essays) is mostly student essays on a few prompts (DAIGT-style), and its AI essays come from a handful of 2023 models. | Topic, prompt and register are confounded with the label. A detector can learn "argumentative essay on prompt X" rather than "machine-written". |
| **The saved detectors saw 10% AI.** Their metadata reads `train_label_counts: [8964, 1036]` on 10,000 rows, although `stratified_sample: True`. | The OCSVM and LR operating points are fitted to a 9:1 prior, and the 0.5 threshold means little elsewhere. |
| **One target domain (Mercor answers)** for selection and final scores. | Two corpora cannot estimate generalization across generators, domains, decoding settings or attacks. |
| **No adversarial or edited text** (paraphrase, homoglyphs, whitespace, human-polished AI text). | On RAID, attacks cut many detectors' TPR by half (e.g. e5-small: 0.83 → 0.73 at 1% FPR). |
| Texts truncated at 512 tokens, no length stratification. | Short texts are much harder; results mix lengths. |

### 1.2 Features and classifiers

- **Frozen mid-layer embeddings + LR / SVM / OCSVM.** This is a reasonable probing baseline, and similar to
  published "intermediate layer" detectors. But the frozen features encode topic and style of the training corpus
  as much as generation artifacts. Nothing forces invariance to domain.
- **One-class detectors (OCSVM, elliptic envelope)** model "human essays". Any human text that is not an essay
  (a recipe, a tweet, a lab report) is an outlier, so it is flagged as AI. That is the domain-shift failure by
  construction. They also turn `decision_function` into a probability with a fixed sigmoid, which is not
  calibrated.
- **`last` pooling on BERT-style encoders reads the `[SEP]` vector** (the extractor keeps special tokens). It
  is not the last word of the text. It works, but it is not what the name suggests.
- **Perplexity and PHD are single features.** Perplexity is length- and domain-sensitive, and small scorer
  models are poor judges of frontier-model text. PHD (Tulchinskii et al., 2023) resists paraphrase but is weak
  in absolute terms and slow.
- **Pickled sklearn objects** tie deployment to library versions (`models.classifiers` must stay importable, and
  `xgboost` must be installed to load an LR). The ONNX export (`textdet/export/saved_onnx.py`) removes that
  dependency for the browser.

### 1.3 Evaluation protocol

What is already right: the fixed `select`/`test` split, thresholds tuned on a held-out 20%, and ROC-AUC ranking
(threshold-free). What was missing, now in `textdet/evaluation`:

- **TPR at 1% and 5% FPR.** This is RAID's headline metric. AUROC hides the case that matters, where a
  detector ranks well but flags a large share of humans with high confidence. The 2026 study below shows
  60% of unseen-domain human texts receiving p ≥ 0.95.
- **The false-positive rate at the operating threshold**, reported separately from accuracy.
- **Per-generator / per-domain / per-attack breakdowns** (`breakdown_*.csv`) when a dataset has those columns.
- **Human-only threshold calibration** (`--optimize_threshold fpr0.05`). A deployment can collect human texts
  of its domain, but rarely labelled AI texts from unknown generators.
- **Selection bias.** Sweeping hundreds of configs (models × layers × poolings × classifiers) against the same
  Mercor rows inflates the best score. The `select`/`test` split helps; keep the `test` part for the one final
  number.

### 1.4 The Jev-like zero-shot baseline

Jev-style decision models (TypeSafe's Jev, Laya, Julia 1) are trained to pick among options grounded in the
supplied context: classification, routing, yes/no. Julia's own card says it "compares the answers you give it
and cannot supply missing facts". Authorship is not visible from the content, so it is not that kind of
question. Measured here:

- Julia 1 scores AUROC 0.47 on MAGE GPT-4, 0.44 on its paraphrased version, and 0.53 on `AI_Human.csv`.
- Its false-positive rate at 0.5 is 64–74%. In spot checks, the reversed question (`human_written`) also got
  "yes" (P(human) ≈ 1 on AI text), so the two polarities disagree. `MODE=eval` asks both and scores their mean.

That makes it useful as a floor that trained detectors must beat, and as a demo of the Jev approach in the
browser (`julia1-jev`, 196 MiB). It is not a detector. Laya (ModernBERT-large) is 4× bigger. It was not re-run
here, but `--backend laya` scores it on the same rows for comparison.

## 2. What the literature says (2024–2026)

| Family | Representative | What to expect |
|---|---|---|
| Zero-shot likelihood statistics | DetectGPT → **Fast-DetectGPT** (Bao et al., ICLR 2024), **Binoculars** (Hans et al., ICML 2024) | Domain-agnostic and strong with 7B scorers on unattacked ChatGPT-era text. Weak with small scorers and under attacks. On RAID, Binoculars (Falcon-7B pair) reaches TPR 0.79 at 5% FPR without attacks. |
| Fine-tuned encoders | RoBERTa / DeBERTa / ModernBERT classifiers; RADAR (adversarial) | Near-perfect in distribution; fragile under shift. |
| Large multi-source, multi-task encoders | **MELD** (arXiv 2605.06903, 2026): Ettin encoder, 6.6M rows (RAID, MAGE, M4GT, DetectRL, Ghostbuster, FineWeb, WildChat), auxiliary generator/attack/domain heads, attack-augmented distillation, hard-negative ranking loss | Strongest open detector on RAID: TPR 0.992 at 1% FPR with attacks, competitive with commercial detectors. |
| Discrepancy learning | **DetectAnyLLM** (DDL, arXiv 2509.14268) + the MIRAGE benchmark | Trains the Fast-DetectGPT statistic directly. Large gains on polished/rewritten text from 17 frontier LLMs. |
| Sobering evidence | **"Rethinking AI-Generated Text Detection"** (arXiv 2607.03680, 2026) | A plain fine-tuned RoBERTa matches the specialized detectors on their own benchmarks (MAGE, MIRAGE, FAID), but leave-one-domain-out TPR@1%FPR falls to 0.59 and cross-benchmark transfer nears 0. The cause is human text of new domains flagged with near certainty. |

### RAID leaderboard, open detectors by size

TPR at 5% / 1% FPR on RAID's hidden test set (per-domain thresholds), from `leaderboard/submissions/*/results.json`
of [liamdugan/raid](https://github.com/liamdugan/raid). All of these were trained on RAID's train split, so this
is in-distribution for them.

| Detector | Params | No attack | All attacks | AUROC (all) |
|---|---|---|---|---|
| MELD (`anon-review-meld-2026/meld`) | 1.0B (Ettin-1B) | 0.998 / 0.994 | 0.998 / 0.992 | 0.998 |
| tmr (`Oxidane/tmr-ai-text-detector`) | 125M (RoBERTa-base) | 0.997 / 0.986 | 0.958 / 0.902 | 0.993 |
| ModernBERT (`GeorgeDrayson/modernbert-ai-detection-raid-mage`) | 150M | 0.991 / 0.982 | 0.941 / 0.882 | 0.976 |
| desklib v1.01 | 434M (DeBERTa-v3-large) | 0.949 / 0.891 | 0.912 / 0.765 | 0.948 |
| e5-small-lora (`MayZhou/e5-small-lora-ai-generated-detector`) | 33M | 0.939 / 0.828 | 0.857 / 0.731 | 0.968 |
| BERT-tiny (`ShantanuT01/BERT-tiny-RAID`) | 4.4M | 0.911 / 0.808 | 0.842 / 0.719 | 0.967 |
| Binoculars (zero-shot, Falcon-7B ×2) | 14B | 0.790 / 0.695 | n/a | n/a |
| RADAR | 355M | 0.656 / 0.481 | 0.639 / 0.431 | 0.819 |
| OpenAI RoBERTa-base GPT-2 detector | 125M | 0.592 / 0.398 | 0.518 / 0.346 | 0.723 |

## 3. Better datasets

| Dataset | Content | Use it for | In `textdet` |
|---|---|---|---|
| **RAID** (Dugan et al., ACL 2024) | 6M+ texts; 11 generators × 8 domains × 4 decoding settings × 11 attacks; labelled train (11 GB), hidden-label test with a leaderboard | Training (sample it), per-cell breakdowns, the leaderboard for final numbers | `raid_sample`, `raid_sample_clean` (streamed, N rows per cell) |
| **MAGE** (Li et al., ACL 2024) | 447k texts; 27 LLMs, 10 domains; two small wild sets: unseen domains with GPT-4 generations, and the same paraphrased | Training; the wild sets are the cheapest honest out-of-distribution test (CPU, minutes) | `mage_test`, `mage_ood_gpt4`, `mage_ood_gpt4_para` |
| **M4GT-Bench** / SemEval-2024 Task 8 | Multilingual, multi-domain, multi-generator; also generator attribution | Multilingual coverage (Julia/mmBERT and e5 are multilingual-capable) | any `name:path` table |
| **DetectRL** (NeurIPS 2024 D&B; `WUJUNCHAO/DetectRL-X` on HF) | Real-world scenarios: attacks, human edits, varied lengths | Robustness training and testing | any `name:path` table |
| **MIRAGE** (DetectAnyLLM, [project page](https://fjc2005.github.io/detectanyllm)) | 17 frontier LLMs (GPT-4o class), generate / polish / rewrite tasks | The most recent generators; polished human text is the hard case | any `name:path` table |
| Your own data (Mercor, AI_Human) | Target domains | Calibration (human texts) and the final `test` number | `mercor_ai:path`, `human_ai_sample:path.zip` |

Recommended protocol:
- **Train** on RAID train + MAGE train (+ M4GT/DetectRL for breadth), plus human text from your target domain.
- **Select** on held-out RAID cells (unseen generator or domain) and MAGE's wild sets.
- **Report** TPR@1%FPR and TPR@5%FPR, the FPR at the deployed threshold, and a leave-one-domain-out score.
- **Final test** once on untouched sets: the RAID leaderboard, MIRAGE, and your `test` part.

## 4. Measured here (CPU, 400 balanced texts per set; partial)

`evaluate_detectors.py`, same rows for every detector (seed 42). Threshold 0.5, uncalibrated. Each cell is
AUROC / TPR at 1% FPR / FPR at P(AI) = 0.5.

| Detector | Params | MAGE GPT-4 (unseen domains) | same, paraphrased | AI_Human essays* |
|---|---|---|---|---|
| Julia 1, zero-shot Jev-like | 144M | 0.467 / 0.000 / 0.74 | 0.441 / 0.000 / 0.74 | 0.533 / 0.005 / 0.64 |
| BERT-tiny (RAID) | 4.4M | 0.877 / 0.475 / 0.45 | 0.790 / 0.300 / 0.45 | 0.871 / 0.355 / 0.58 |
| e5-small-lora (RAID) | 33M | 0.955 / 0.530 / 0.46 | 0.847 / 0.330 / 0.46 | 0.849 / 0.250 / 0.68 |
| tmr RoBERTa-base (RAID) | 125M | **0.975 / 0.805** / 0.39 | 0.810 / 0.380 / 0.39 | 0.885 / 0.550 / **0.985** |

\* `human_ai_sample`, 200 essays per class. The saved `human_ai_*` detectors were trained on this file.

What it shows:
- **Ranking is good; operating points are not.** tmr flags 98.5% of the human student essays at 0.5
  (AUROC 0.885). Essays are a domain RAID does not cover.
- **Calibration on human text is cheap and transfers.** e5-small's output was shifted so 50% flags 5% of 2,000
  MAGE-test human texts (other domains than the test set). On the GPT-4 set its false-positive rate then went
  from 46% to 2.5%, with AUROC unchanged (0.956) and recall 0.55.
- **Paraphrasing is the main open weakness:** every detector loses 0.08–0.17 AUROC on the paraphrased set.
- **Pending (SLURM `MODE=eval`):** ModernBERT RAID+MAGE, Fast-DetectGPT and Binoculars (Qwen2.5-0.5B), the
  saved MiniLM / DeBERTa / Qwen detectors, Laya, the ensembles, and Mercor and RAID.

Browser exports (`export_web_model.py`), checked against the Python detector on 40 MAGE texts, and in the app's
own runtime (`deepfake-detection-38502/scripts/run-text-model.mjs`: onnxruntime-web WebAssembly + tokenizers.js):

| Folder | Size | Rank correlation with Python | Same decision | App runtime vs Python |
|---|---|---|---|---|
| `julia1-jev` (int8 embeddings + fp16 matrices) | 196 MiB | 0.9988 (120 texts) | 99.2% | identical tokens, ΔP < 4e-6 |
| `e5-small-raid` (int8) | 37 MiB | 0.9991 | 100% | identical tokens, ΔP < 4e-7 |
| `minilm-l2-last-ocsvm` (your saved detector, int8) | 18 MiB | 0.998 | 100% | identical tokens, ΔP < 7e-6 |

Quantization note: onnxruntime's `quantize_dynamic` (140 MiB for Julia) also quantizes activations, and it
destroys BERT-family encoders (rank correlation 0.02). Weight-only int8 with one scale per 32 weights
(`textdet/export/quantize.py`) keeps them. mmBERT's matrices still prefer fp16, hence Julia's mixed format.

## 5. The lightweight plan

**Now, without training (implemented):**

1. **Detector:** an off-the-shelf RAID-trained small classifier, exported for the browser with weight-only int8:
   tmr RoBERTa-base (best ranking) or e5-small (37 MiB, fastest).
2. **Calibration:** `export_web_model.py ... --calibrate_fpr 0.05` shifts the output so the page's 50% line
   flags 5% of reference human texts (MAGE by default; your own domain's human texts are better). It is one
   quantile, not training.
3. **Evaluation:** every candidate goes through `evaluate_detectors.py` on the same rows, reporting TPR@1%FPR
   and FPR. The exported folder itself can be evaluated (`web:<folder>`), so what ships is what was measured.

**Next, with training (not run):** a MELD-style student small enough for the browser.
- Backbone: Ettin-32M or Ettin-68M (ModernBERT family, 8k context, MIT). int8 gives ~35–75 MiB.
- Data: RAID + MAGE + M4GT + DetectRL + human web text (FineWeb/WildChat-style), with character and paraphrase
  attack augmentation.
- Losses: binary cross-entropy, plus auxiliary generator/domain/attack heads (dropped at inference), plus a
  hard-negative ranking loss on the most confusable human texts.
- Distillation: from MELD (open weights) or tmr as the teacher.
- Selection by leave-one-domain-out TPR@1%FPR, not AUROC.

`scripts/train_and_save_detector.py` and the embedding probes remain useful as cheap baselines and for studying
layers. They are not the path to a robust detector.
