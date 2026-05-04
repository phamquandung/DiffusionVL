"""
NAVIDA-style JSONL dataset for DiffusionVL training.

Builds LLaVA-compatible rows (human/gpt + explicit <image> placeholders) from
NAVIDA jsonl, then delegates tensorization to LazySupervisedDataset._get_item
via a lightweight LazySupervisedDataset instance (no train.py import at module load).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

from torch.utils.data import Dataset

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.utils import rank0_print


def uniform_sample_with_ends(data: List[Any], n: int) -> List[Any]:
    """Same indexing as NAVIDA/src/train/train.py uniform_sample_with_ends."""
    if len(data) <= n:
        return list(data)
    indices = [round(i * (len(data) - 1) / (n - 1)) for i in range(n)]
    return [data[i] for i in indices]


def _load_list_data_dict(data_path: str) -> List[Dict[str, Any]]:
    """Load JSON array or JSONL (NaVILA LazyVLNCEDataset style)."""
    try:
        with open(data_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        with open(data_path, "r", encoding="utf-8") as fp:
            return [json.loads(line) for line in fp if line.strip()]


def _get_user_assistant(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    conv = raw["conversations"]
    if len(conv) < 2:
        raise ValueError("NAVIDA sample needs at least one user and one assistant turn.")
    return conv[0], conv[1]


def _coerce_image_list(user_turn: Dict[str, Any]) -> List[str]:
    img = user_turn.get("image")
    if img is None:
        raise ValueError("NAVIDA user turn missing 'image'.")
    if isinstance(img, str):
        return [img]
    if isinstance(img, list):
        return list(img)
    raise TypeError(f"Unexpected image type: {type(img)}")


class LazyNavidaJsonlDataset(Dataset):
    """
    Loads NAVIDA jsonl / json array; each ``_get_item`` builds a synthetic LLaVA row
    (top-level ``image`` paths + human/gpt ``conversations`` with ``<image>`` tokens),
    then reuses ``LazySupervisedDataset._get_item`` for ``process_image`` / ``preprocess``.
    """

    def __init__(self, data_path: str, tokenizer, data_args):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = _load_list_data_dict(data_path)
        rank0_print(f"NAVIDA jsonl: loaded {len(self.list_data_dict)} samples from {data_path}")

    def _max_history_frames(self) -> int:
        return int(getattr(self.data_args, "navida_max_history_frames", 8))

    def _raw_to_synthetic(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        task = raw.get("task type", raw.get("task_type", "vln"))
        user_t, asst_t = _get_user_assistant(raw)
        answer = asst_t.get("value", "")
        paths = _coerce_image_list(user_t)
        user_value = user_t.get("value", "")
        k = self._max_history_frames()

        system_prefix = ""
        if raw.get("system"):
            system_prefix = str(raw["system"]).strip() + "\n\n"

        if task == "vln":
            if len(paths) > 1:
                history_paths = uniform_sample_with_ends(paths[:-1], k)
                current_path = paths[-1]
                ordered_paths = history_paths + [current_path]
                parts = user_value.split("current observation", 1)
                tail = parts[1] if len(parts) > 1 else ""
                question_body = (
                    "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations\n"
                    + (DEFAULT_IMAGE_TOKEN + "\n") * len(history_paths)
                    + "and an image of the current observation\n"
                    + DEFAULT_IMAGE_TOKEN
                    + "\n"
                    + tail
                )
            else:
                p0 = paths[0]
                ordered_paths = [p0, p0]
                parts = user_value.split("current observation", 1)
                tail = parts[1] if len(parts) > 1 else ""
                question_body = (
                    "Imagine you are a robot programmed for navigation tasks. You have been given a video of historical observations\n"
                    + DEFAULT_IMAGE_TOKEN
                    + "\n"
                    + "and an image of the current observation\n"
                    + DEFAULT_IMAGE_TOKEN
                    + "\n"
                    + tail
                )
        elif task == "trajectory summarization":
            sampled = uniform_sample_with_ends(paths, k)
            ordered_paths = sampled
            parts = user_value.split("images sequences", 1)
            tail = parts[1] if len(parts) > 1 else ""
            question_body = (
                "Assume you are a robot designed for navigation. You are provided with captured images sequences"
                + "\n"
                + (DEFAULT_IMAGE_TOKEN + "\n") * len(ordered_paths)
                + tail
            )
        elif task == "idm":
            if len(paths) < 2:
                raise ValueError("NAVIDA idm task requires at least two images.")
            ordered_paths = [paths[0], paths[1]]
            parts = user_value.split("goal view. ", 1)
            tail = parts[1] if len(parts) > 1 else ""
            question_body = (
                "Imagine you are a robot programmed for navigation tasks. You have been given an image of current view\n"
                + DEFAULT_IMAGE_TOKEN
                + "\n"
                + "and an image of the goal view\n"
                + DEFAULT_IMAGE_TOKEN
                + "\n"
                + tail
            )
        else:
            raise NotImplementedError(f"Unsupported NAVIDA task type: {task!r}")

        human_value = system_prefix + question_body
        conversations = [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": answer},
        ]
        return {"conversations": conversations, "image": ordered_paths}

    def _get_item(self, i) -> Dict[str, Any]:
        from llava.train.train import LazySupervisedDataset

        raw = self.list_data_dict[i]
        synthetic = self._raw_to_synthetic(raw)
        helper = LazySupervisedDataset.__new__(LazySupervisedDataset)
        helper.tokenizer = self.tokenizer
        helper.data_args = self.data_args
        helper.list_data_dict = self.list_data_dict
        self.list_data_dict[i] = synthetic
        try:
            return LazySupervisedDataset._get_item(helper, i)
        finally:
            self.list_data_dict[i] = raw

    def __getitem__(self, i) -> Dict[str, Any]:
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch NAVIDA sample {i}. Exception:", e)
                time.sleep(1)

        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                return self._get_item(next_index)
            except Exception as e:
                print(f"[Try other #{attempt_idx}] Failed NAVIDA sample {next_index}. Exception:", e)

        return self._get_item(i)

    def __len__(self):
        return len(self.list_data_dict)

    @staticmethod
    def _raw_has_images(sample: Dict[str, Any]) -> bool:
        for turn in sample.get("conversations", []):
            if isinstance(turn, dict) and turn.get("image") is not None:
                return True
        return False

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            conv = sample.get("conversations", [])
            img_tokens = 128 if self._raw_has_images(sample) else 0
            length_list.append(
                sum(len(t.get("value", "").split()) for t in conv if isinstance(t, dict))
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            conv = sample.get("conversations", [])
            cur_len = sum(
                len(t.get("value", "").split()) for t in conv if isinstance(t, dict)
            )
            has_img = self._raw_has_images(sample)
            length_list.append(cur_len if has_img else -cur_len)
        return length_list
