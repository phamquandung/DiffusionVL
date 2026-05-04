#!/usr/bin/env python3
"""
Show NAVIDA -> DiffusionVL training pipeline stages for one sample:
1) pre-tokenization synthetic conversation
2) tokenized input_ids / labels (from preprocess path)
3) final batch dict produced by data collator (model training input)
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoTokenizer


def shorten(text, max_chars):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ... [truncated]"


def read_jsonl_row(path: Path, idx: int):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"sample_idx={idx} out of range")


def main():
    parser = argparse.ArgumentParser(description="Inspect NAVIDA -> DiffusionVL training inputs.")
    parser.add_argument(
        "--dataset",
        default="/mnt/samsung/Project/CoRL-ICRA/navida_train_data_r2r.jsonl",
        help="Path to NAVIDA jsonl.",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index in jsonl.")
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model path/name for tokenizer+processor.",
    )
    parser.add_argument("--max_history_frames", type=int, default=8)
    parser.add_argument("--max_text_chars", type=int, default=600)
    parser.add_argument("--max_tokens_preview", type=int, default=256)
    args = parser.parse_args()

    # Make sure llava package is importable.
    this_script = Path(__file__).resolve()
    diffusion_train_root = this_script.parent.parent  # DiffusionVL/train
    if str(diffusion_train_root) not in sys.path:
        sys.path.insert(0, str(diffusion_train_root))

    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.train.navida_jsonl_dataset import LazyNavidaJsonlDataset
    from llava.train.train import DataArguments, DataCollatorForSupervisedDataset, preprocess, preprocess_multimodal

    dataset_path = Path(args.dataset).expanduser().resolve()
    raw = read_jsonl_row(dataset_path, args.sample_idx)

    # ---------- Stage 1: before tokenize ----------
    converter = object.__new__(LazyNavidaJsonlDataset)
    converter.data_args = type("DA", (), {"navida_max_history_frames": args.max_history_frames})()
    synthetic = LazyNavidaJsonlDataset._raw_to_synthetic(converter, raw)

    print("=== Stage 1: Pre-tokenization synthetic sample ===")
    print(f"sample_idx: {args.sample_idx}")
    print(f"task type: {raw.get('task type', raw.get('task_type'))}")
    print(f"selected image count: {len(synthetic['image'])}")
    print(f"<image> token count in prompt: {synthetic['conversations'][0]['value'].count('<image>')}")
    print("\n[human prompt preview]")
    print(shorten(synthetic["conversations"][0]["value"], args.max_text_chars))
    print("\n[assistant target preview]")
    print(shorten(synthetic["conversations"][1]["value"], args.max_text_chars))
    print("\n[first 5 selected image paths]")
    for p in synthetic["image"][:5]:
        print(" -", p)

    # ---------- Stage 2: tokenize ----------
    print("\n=== Stage 2: Tokenization output (preprocess) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    tokenizer.model_max_length = 8192

    # Mimic train pipeline: preprocess_multimodal then preprocess
    data_args_token = DataArguments(
        data_path=str(dataset_path),
        is_multimodal=True,
        image_folder=".",
        image_aspect_ratio="pad",
    )
    # In DiffusionVL train flow this is injected from model args.
    data_args_token.mm_use_im_start_end = False
    sources = preprocess_multimodal([synthetic["conversations"]], data_args_token)
    tokenized = preprocess(sources, tokenizer, has_image=True)
    input_ids = tokenized["input_ids"][0]
    labels = tokenized["labels"][0]
    target_count = int((labels != -100).sum().item())
    print(f"input_ids length: {input_ids.numel()}")
    print(f"labels length   : {labels.numel()}")
    print(f"target tokens   : {target_count}")
    print(f"image token id occurrences (IMAGE_TOKEN_INDEX): {(input_ids == IMAGE_TOKEN_INDEX).sum().item()}")
    print("\n[input_ids preview]")
    print(input_ids[: args.max_tokens_preview].tolist())
    print("\n[labels preview]")
    print(labels[: args.max_tokens_preview].tolist())

    # ---------- Stage 3: final model training batch ----------
    print("\n=== Stage 3: Final batch for model training (dataset + collator) ===")
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    data_args_full = DataArguments(
        data_path=str(dataset_path),
        is_multimodal=True,
        image_folder=".",
        image_aspect_ratio="pad",
        navida_max_history_frames=args.max_history_frames,
    )
    data_args_full.mm_use_im_start_end = False
    data_args_full.image_processor = processor
    ds = LazyNavidaJsonlDataset(data_path=str(dataset_path), tokenizer=tokenizer, data_args=data_args_full)
    sample = ds[args.sample_idx]
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, training_args=None)
    batch = collator([sample])

    print("batch keys:", list(batch.keys()))
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        elif isinstance(v, list):
            print(f"{k}: list(len={len(v)})")
        else:
            print(f"{k}: type={type(v)}")

    print("\nDone.")


if __name__ == "__main__":
    main()