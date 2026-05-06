#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model for inference/export.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--safe-serialization", action="store_true")
    args = parser.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.lora_dir, trust_remote_code=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, args.lora_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
