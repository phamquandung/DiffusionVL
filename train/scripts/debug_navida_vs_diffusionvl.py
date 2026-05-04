#!/usr/bin/env python3
"""
Compare NAVIDA reference conversion vs DiffusionVL NAVIDA dataset conversion.

This script validates that DiffusionVL's LazyNavidaJsonlDataset produces
pre-tokenization structure consistent with NAVIDA convert logic.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def uniform_sample_with_ends(data, n):
    if len(data) <= n:
        return list(data)
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def navida_expected_from_raw(raw, max_history_frames=8):
    """Build expected invariants from NAVIDA convert logic."""
    conv = raw["conversations"]
    user = conv[0]
    assistant = conv[1]["value"]
    task = raw.get("task type", raw.get("task_type", "vln"))
    image_paths = user["image"] if isinstance(user["image"], list) else [user["image"]]
    user_value = user.get("value", "")

    if task == "vln":
        if len(image_paths) > 1:
            history = uniform_sample_with_ends(image_paths[:-1], max_history_frames)
            selected = history + [image_paths[-1]]
        else:
            selected = [image_paths[-1], image_paths[-1]]
        split_key = "current observation"
    elif task == "trajectory summarization":
        selected = uniform_sample_with_ends(image_paths, max_history_frames)
        split_key = "images sequences"
    elif task == "idm":
        if len(image_paths) < 2:
            raise ValueError("idm expects at least two images")
        selected = [image_paths[0], image_paths[1]]
        split_key = "goal view. "
    else:
        raise NotImplementedError(f"Unsupported task type: {task}")

    tail = user_value.split(split_key, 1)[1] if split_key in user_value else user_value
    return {
        "task": task,
        "assistant": assistant,
        "selected_images": selected,
        "tail": tail.strip(),
        "system": str(raw.get("system", "")).strip(),
    }


def load_jsonl_indices(path, indices):
    max_idx = max(indices)
    wanted = set(indices)
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > max_idx:
                break
            if i in wanted:
                rows[i] = json.loads(line)
    missing = [i for i in indices if i not in rows]
    if missing:
        raise IndexError(f"indices out of range or unreadable: {missing}")
    return rows


def count_jsonl_lines(path):
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            total += 1
    return total


def build_diffusion_converter():
    # Ensure "llava" package import works regardless of current cwd.
    script_path = Path(__file__).resolve()
    diffusion_train_root = script_path.parent.parent  # DiffusionVL/train
    if str(diffusion_train_root) not in sys.path:
        sys.path.insert(0, str(diffusion_train_root))

    from llava.train.navida_jsonl_dataset import LazyNavidaJsonlDataset

    obj = object.__new__(LazyNavidaJsonlDataset)
    obj.data_args = SimpleNamespace(navida_max_history_frames=8)
    return obj, LazyNavidaJsonlDataset


def compare_sample(idx, raw, converter_obj, dataset_cls, max_history_frames=8):
    expected = navida_expected_from_raw(raw, max_history_frames=max_history_frames)
    converted = dataset_cls._raw_to_synthetic(converter_obj, raw)
    user_text = converted["conversations"][0]["value"]
    assistant_text = converted["conversations"][1]["value"]
    selected = converted["image"]

    checks = {
        "assistant_equal": assistant_text == expected["assistant"],
        "image_selection_equal": selected == expected["selected_images"],
        "image_token_count_equal": user_text.count("<image>") == len(selected),
        "tail_in_prompt": expected["tail"] in user_text if expected["tail"] else True,
        "system_prefix_present": expected["system"] in user_text if expected["system"] else True,
        "roles_human_gpt": (
            converted["conversations"][0]["from"] == "human"
            and converted["conversations"][1]["from"] == "gpt"
        ),
    }
    ok = all(checks.values())
    return ok, checks, expected, converted


def main():
    parser = argparse.ArgumentParser(description="Compare NAVIDA vs DiffusionVL conversion.")
    parser.add_argument(
        "--dataset",
        default="/mnt/samsung/Project/CoRL-ICRA/navida_train_data_r2r.jsonl",
        help="Path to NAVIDA jsonl file.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        nargs="*",
        default=[],
        help="Specific indices to check.",
    )
    parser.add_argument(
        "--num_random",
        type=int,
        default=20,
        help="Number of random samples to check when --sample_idx is empty.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max_history_frames",
        type=int,
        default=8,
        help="Expected history frame sampling count.",
    )
    parser.add_argument(
        "--show_ok_samples",
        action="store_true",
        help="Print per-sample details even when all checks pass.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    total = count_jsonl_lines(dataset_path)
    if total == 0:
        raise ValueError("Dataset is empty.")

    if args.sample_idx:
        indices = sorted(set(args.sample_idx))
    else:
        random.seed(args.seed)
        n = min(args.num_random, total)
        indices = sorted(random.sample(range(total), n))

    converter_obj, dataset_cls = build_diffusion_converter()
    converter_obj.data_args.navida_max_history_frames = args.max_history_frames

    rows = load_jsonl_indices(dataset_path, indices)

    passed = 0
    failed = 0
    print(f"Dataset: {dataset_path}")
    print(f"Total rows: {total}")
    print(f"Checking indices: {indices[:10]}{' ...' if len(indices) > 10 else ''}")
    print()

    for idx in indices:
        ok, checks, expected, converted = compare_sample(
            idx,
            rows[idx],
            converter_obj,
            dataset_cls,
            max_history_frames=args.max_history_frames,
        )
        if ok:
            passed += 1
            if args.show_ok_samples:
                print(f"[OK] idx={idx} task={expected['task']} images={len(expected['selected_images'])}")
            continue

        failed += 1
        print(f"[FAIL] idx={idx} task={expected['task']}")
        for key, value in checks.items():
            print(f"  - {key}: {value}")
        print(f"  - expected images ({len(expected['selected_images'])}): {expected['selected_images'][:6]}")
        print(f"  - actual   images ({len(converted['image'])}): {converted['image'][:6]}")
        print(f"  - expected assistant: {expected['assistant'][:200]}")
        print(f"  - actual   assistant: {converted['conversations'][1]['value'][:200]}")
        print(f"  - actual user text head: {converted['conversations'][0]['value'][:300]}")
        print()

    print("=== Summary ===")
    print(f"checked: {len(indices)}")
    print(f"passed : {passed}")
    print(f"failed : {failed}")
    if failed == 0:
        print("All checks passed.")
    else:
        print("Found mismatches. See failed samples above.")


if __name__ == "__main__":
    main()
