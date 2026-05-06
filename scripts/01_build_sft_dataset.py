#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from style_finetune.common import extract_message_rows, is_good_text, write_jsonl  # noqa: E402

DEFAULT_SYSTEM_PROMPT = (
    "你正在模仿一位中文聊天对象的聊天风格。"
    "请根据最近的聊天上下文，只回复下一句。"
    "回复要自然、简短、口语化；不要解释自己在模仿；不要输出引号。"
)


def make_context(rows: List[Dict[str, Any]], target_name: str) -> str:
    lines = []
    for r in rows:
        speaker = "你" if r["name"] == target_name else "对方"
        lines.append(f"{speaker}：{r['text']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build completion-only SFT data for chat style finetuning.")
    parser.add_argument("--input", required=True, help="Path to exported chat JSONL")
    parser.add_argument("--target-name", required=True, help="Sender name/remark/nickname to imitate")
    parser.add_argument("--output-dir", required=True, help="Directory for train.jsonl, val.jsonl and stats.json")
    parser.add_argument("--context-turns", type=int, default=8, help="Number of previous usable messages used as context")
    parser.add_argument("--min-response-chars", type=int, default=2)
    parser.add_argument("--max-response-chars", type=int, default=260)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means no cap")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no-anonymize", action="store_true", help="Do not mask phone/email/account-like text")
    args = parser.parse_args()

    rows = extract_message_rows(args.input, anonymize=not args.no_anonymize, min_chars=1)
    examples: List[Dict[str, Any]] = []

    for i, row in enumerate(rows):
        if row["name"] != args.target_name:
            continue
        answer = row["text"].strip()
        if not is_good_text(answer, min_chars=args.min_response_chars):
            continue
        if len(answer) > args.max_response_chars:
            continue
        context_rows = rows[max(0, i - args.context_turns): i]
        if not context_rows:
            continue
        context = make_context(context_rows, args.target_name)
        user_prompt = (
            "以下是最近的聊天上下文。请模仿“你”的风格，回复下一句。\n\n"
            f"{context}\n\n下一句："
        )
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": answer},
                ],
                "meta": {
                    "target_name": args.target_name,
                    "source_time": row["time"],
                    "source_id": row.get("id"),
                    "context_turns": len(context_rows),
                },
            }
        )

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_size = max(1, int(len(examples) * args.val_ratio)) if len(examples) >= 10 else 0
    val = examples[:val_size]
    train = examples[val_size:]

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)

    stats = {
        "input": str(args.input),
        "target_name": args.target_name,
        "usable_messages": len(rows),
        "examples_total": len(examples),
        "train_examples": len(train),
        "val_examples": len(val),
        "context_turns": args.context_turns,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "system_prompt": args.system_prompt,
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if len(train) < 50:
        print("WARNING: fewer than 50 training examples. Style may be weak or unstable.")


if __name__ == "__main__":
    main()
