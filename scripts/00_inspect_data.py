#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from style_finetune.common import extract_message_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exported chat JSONL and list possible style targets.")
    parser.add_argument("--input", required=True, help="Path to exported chunk_0001.jsonl")
    parser.add_argument("--no-anonymize", action="store_true", help="Do not mask phone/email/account-like text during inspection")
    parser.add_argument("--top", type=int, default=20, help="Show top N senders")
    args = parser.parse_args()

    rows = extract_message_rows(args.input, anonymize=not args.no_anonymize, min_chars=1)
    by_name = collections.defaultdict(list)
    for r in rows:
        by_name[r["name"]].append(r["text"])

    print(f"usable_text_messages: {len(rows)}")
    print(f"senders: {len(by_name)}")
    print("\nTop senders:")
    stats = []
    for name, texts in by_name.items():
        lens = [len(t) for t in texts]
        stats.append((len(texts), sum(lens), statistics.mean(lens), name))
    for count, total_chars, avg_chars, name in sorted(stats, reverse=True)[: args.top]:
        print(f"- {name}: {count} messages | {total_chars} chars | avg {avg_chars:.1f}")

    print("\nSample cleaned messages:")
    for r in rows[:5]:
        print(json.dumps({"time": r["time"], "name": r["name"], "text": r["text"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
