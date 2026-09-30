#!/usr/bin/env python3
"""
Download what the textdet SLURM jobs need, on a node with internet (the login node), so compute nodes
run offline from the Hugging Face cache and data/hf/ (the catalog uses data/hf/<org>/<name>/<file> first).

    python scripts/prefetch_textdet.py                 # models + MAGE files (~0.5 GB of data)
    python scripts/prefetch_textdet.py --raid          # + RAID train.csv (11 GB)
    python scripts/prefetch_textdet.py --laya          # + the Laya checkpoint (0.8 GB)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODELS = [
    "SupersonicLabs/Julia-1", "SupersonicLabs/Julia-1-ONNX",
    "ShantanuT01/BERT-tiny-RAID", "MayZhou/e5-small-lora-ai-generated-detector",
    "Oxidane/tmr-ai-text-detector", "GeorgeDrayson/modernbert-ai-detection-raid-mage",
    "Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct",
    "sentence-transformers/all-MiniLM-L6-v2",
]
MAGE_FILES = ["test.csv", "test_ood_set_gpt.csv", "test_ood_set_gpt_para.csv"]


def main():
    parser = argparse.ArgumentParser(description="Prefetch models and datasets for the textdet jobs")
    parser.add_argument("--raid", action="store_true", help="Also download RAID train.csv (11 GB)")
    parser.add_argument("--laya", action="store_true", help="Also download the Laya checkpoint")
    parser.add_argument("--no_models", action="store_true", help="Datasets only")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download
    from textdet.data.catalog import MIRROR

    models = [] if args.no_models else MODELS + (["convaiinnovations/laya"] if args.laya else [])
    for repo in models:
        print(f"model {repo}")
        # Weights, configs and tokenizers only (some repos also hold .bin / .pt duplicates)
        snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.onnx",
                                                "*.onnx.data", "encoder/*", "tokenizer/*", "merges.txt"])
    files = [("yaful/MAGE", f) for f in MAGE_FILES] + ([("liamdugan/raid", "train.csv")] if args.raid else [])
    for repo, name in files:
        target = MIRROR / repo
        print(f"dataset {repo}/{name} -> {target}")
        hf_hub_download(repo, name, repo_type="dataset", local_dir=str(target))
    print("Done. On the compute nodes: export HF_HUB_OFFLINE=1 (the catalog reads data/hf/ first).")


if __name__ == "__main__":
    main()
