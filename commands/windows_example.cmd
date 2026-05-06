python scripts/00_inspect_data.py --input data/chunk_0001.jsonl
python scripts/01_build_sft_dataset.py --input data/chunk_0001.jsonl --target-name "樊真菡" --output-dir data/sft_fanzhenhan --context-turns 8
python scripts/05_preview_dataset.py --file data/sft_fanzhenhan/train.jsonl --n 3
python scripts/02_train_lora.py --config configs/qwen2_5_1_5b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-lora
python scripts/03_chat_with_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct --lora-dir outputs/fanzhenhan-lora --load-in-4bit
