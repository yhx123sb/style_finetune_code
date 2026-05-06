#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from style_finetune.training_utils import CompletionOnlyCollator, make_training_args_kwargs  # noqa: E402


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gpu_summary() -> str:
    if not torch.cuda.is_available():
        return "CUDA not available"
    lines = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB")
    return " | ".join(lines)


def dtype_from_config(value: str) -> torch.dtype:
    value = str(value).lower()
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quant_config(cfg: Dict[str, Any]) -> BitsAndBytesConfig | None:
    load_4bit = bool(cfg.get("load_in_4bit", True))
    load_8bit = bool(cfg.get("load_in_8bit", False))
    if not load_4bit and not load_8bit:
        return None
    compute_dtype = dtype_from_config(cfg.get("bnb_4bit_compute_dtype", "auto"))
    return BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        load_in_8bit=load_8bit,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def normalize_target_modules(value: Any) -> Any:
    if isinstance(value, str):
        if "," in value:
            return [x.strip() for x in value.split(",") if x.strip()]
        return value
    return value


def tokenize_example(example: Dict[str, Any], tokenizer: Any, max_seq_length: int) -> Dict[str, List[int]]:
    messages = example["messages"]
    if len(messages) < 2:
        raise ValueError("Each example must contain at least user and assistant messages")

    prompt_messages = messages[:-1]
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]

    # Keep the end of the sequence because the target assistant reply is at the end.
    cut_left = max(0, len(full_ids) - max_seq_length)
    input_ids = full_ids[cut_left:]
    prompt_len_after_cut = max(0, len(prompt_ids) - cut_left)

    labels = input_ids.copy()
    for i in range(min(prompt_len_after_cut, len(labels))):
        labels[i] = -100

    # If truncation removed the whole assistant answer, keep a single ignored label to avoid crashes.
    if all(x == -100 for x in labels):
        labels[-1] = input_ids[-1]

    return {"input_ids": input_ids, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA adapter for Chinese chat style imitation.")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--train-file", required=True, help="SFT train.jsonl created by 01_build_sft_dataset.py")
    parser.add_argument("--val-file", default="", help="Optional val.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_cfg = cfg.get("training", {})
    lora_cfg = cfg.get("lora", {})
    model_cfg = cfg.get("model", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    print(f"Environment: {gpu_summary()}")
    model_name = model_cfg.get("model_name_or_path") or cfg.get("model_name_or_path")
    if not model_name:
        raise ValueError("Config must set model.model_name_or_path")

    max_seq_length = int(train_cfg.get("max_seq_length", 1024))
    torch_dtype = dtype_from_config(model_cfg.get("torch_dtype", "auto"))
    quant_config = build_quant_config(model_cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_fast=bool(model_cfg.get("use_fast_tokenizer", True)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map: Any = None
    if quant_config is not None:
        if "LOCAL_RANK" in os.environ:
            device_map = {"": local_rank}
        else:
            device_map = {"": 0} if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map=device_map,
    )
    model.config.use_cache = False

    if quant_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        )

    target_modules = normalize_target_modules(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_files: Dict[str, str] = {"train": args.train_file}
    has_val = bool(args.val_file and Path(args.val_file).exists())
    if has_val:
        data_files["validation"] = args.val_file
    raw = load_dataset("json", data_files=data_files)

    def map_fn(ex: Dict[str, Any]) -> Dict[str, List[int]]:
        return tokenize_example(ex, tokenizer, max_seq_length)

    tokenized = raw.map(
        map_fn,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing and masking prompt tokens",
    )

    eval_steps = int(train_cfg.get("eval_steps", 50))
    save_steps = int(train_cfg.get("save_steps", 50))
    logging_steps = int(train_cfg.get("logging_steps", 10))
    bf16 = bool(train_cfg.get("bf16", "auto") == True) or (str(train_cfg.get("bf16", "auto")).lower() == "auto" and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    fp16 = bool(train_cfg.get("fp16", "auto") == True) or (str(train_cfg.get("fp16", "auto")).lower() == "auto" and torch.cuda.is_available() and not bf16)

    base_args = dict(
        output_dir=args.output_dir,
        overwrite_output_dir=bool(train_cfg.get("overwrite_output_dir", True)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        eval_steps=eval_steps if has_val else None,
        eval_strategy="steps" if has_val else "no",
        save_strategy="steps",
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        optim=train_cfg.get("optim", "paged_adamw_8bit" if quant_config is not None else "adamw_torch"),
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
    )
    base_args = {k: v for k, v in base_args.items() if v is not None}
    training_args = TrainingArguments(**make_training_args_kwargs(TrainingArguments, base_args))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation") if has_val else None,
        data_collator=CompletionOnlyCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "train_file": args.train_file, "val_file": args.val_file}, f, ensure_ascii=False, indent=2)
    print(f"Saved LoRA adapter to: {args.output_dir}")


if __name__ == "__main__":
    main()
