from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class CompletionOnlyCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            ids = item["input_ids"]
            labs = item["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            labels.append(labs + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_training_args_kwargs(training_args_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs for the installed Transformers version.

    Transformers renamed evaluation_strategy to eval_strategy in newer releases.
    This helper keeps the script usable across a wider set of versions.
    """
    params = set(inspect.signature(training_args_cls.__init__).parameters.keys())
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in params:
            out[key] = value
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        out["evaluation_strategy"] = kwargs["eval_strategy"]
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        out["eval_strategy"] = kwargs["evaluation_strategy"]
    return out
