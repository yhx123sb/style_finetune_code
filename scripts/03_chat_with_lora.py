#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = (
    "你正在模仿一位中文聊天对象的聊天风格。"
    "请自然、简短、口语化地回复。不要解释自己在模仿。"
)


def make_quant_config(load_in_4bit: bool) -> BitsAndBytesConfig | None:
    if not load_in_4bit:
        return None
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def build_user_prompt(history: List[Tuple[str, str]], user_text: str, context_turns: int) -> str:
    lines = []
    for speaker, text in history[-context_turns:]:
        lines.append(f"{speaker}：{text}")
    lines.append(f"对方：{user_text}")
    return "以下是最近的聊天上下文。请模仿“你”的风格，回复下一句。\n\n" + "\n".join(lines) + "\n\n下一句："


def load_model(base_model: str, lora_dir: str, load_in_4bit: bool) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(lora_dir if lora_dir else base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = make_quant_config(load_in_4bit)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return tokenizer, model


def generate(tokenizer: Any, model: Any, prompt: str, args: argparse.Namespace) -> str:
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat with a trained LoRA style adapter.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--context-turns", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    args = parser.parse_args()

    tokenizer, model = load_model(args.base_model, args.lora_dir, args.load_in_4bit)
    print("输入内容开始聊天。输入 /exit 退出，/clear 清空上下文。")
    history: List[Tuple[str, str]] = []
    while True:
        user_text = input("对方> ").strip()
        if not user_text:
            continue
        if user_text in {"/exit", "exit", "quit", "q"}:
            break
        if user_text == "/clear":
            history.clear()
            print("已清空上下文。")
            continue
        prompt = build_user_prompt(history, user_text, args.context_turns)
        answer = generate(tokenizer, model, prompt, args)
        print(f"你> {answer}")
        history.append(("对方", user_text))
        history.append(("你", answer))


if __name__ == "__main__":
    main()
