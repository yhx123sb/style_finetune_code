#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview SFT examples generated for style finetuning.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for idx, row in enumerate(rows[: args.n], 1):
        print("=" * 80)
        print(f"Example {idx}")
        for msg in row["messages"]:
            print(f"\n[{msg['role']}]\n{msg['content']}")


if __name__ == "__main__":
    main()
