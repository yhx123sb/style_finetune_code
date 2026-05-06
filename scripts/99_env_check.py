#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform


def version(pkg: str) -> str:
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return f"not installed ({exc.__class__.__name__})"


def main() -> None:
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    for pkg in ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes"]:
        print(f"{pkg}: {version(pkg)}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"CUDA device count: {n}")
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / 1024**3
                bf16 = torch.cuda.is_bf16_supported()
                print(f"GPU {i}: {props.name} | {total_gb:.1f} GB | capability {props.major}.{props.minor} | bf16={bf16}")
    except Exception as exc:  # noqa: BLE001
        print(f"Torch/CUDA check failed: {exc}")


if __name__ == "__main__":
    main()
