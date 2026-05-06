# 中文聊天风格微调

这个项目用于把聊天导出的 `chunk_0001.jsonl` 转成监督微调数据，并用本地 NVIDIA GPU 训练 LoRA 或 QLoRA 风格适配器。

适用目标：让模型学习某个聊天对象的回复风格、语气、长度、常用表达。

不适用目标：从零训练一个通用大语言模型；聊天记录太少时也不适合训练知识型模型。

重要说明：本 README 里的所有命令都写成单行，适合直接复制到 Windows PowerShell、CMD、VS Code 终端、Linux shell 或 macOS 终端。没有使用 Linux/macOS 的反斜杠续行写法。

隐私提醒：请只训练你有权使用的数据。默认脚本会把手机号、邮箱、疑似账号数字等替换成占位符；如果你明确不想匿名化，可以使用 `--no-anonymize`。

## 目录结构

```text
style_finetune_gpu_code/
  configs/
    qwen_lora.yaml
    qwen2_5_1_5b_qlora.yaml
    qwen2_5_3b_qlora.yaml
    qwen2_5_7b_qlora.yaml
  data/
  outputs/
  scripts/
    99_env_check.py
    00_inspect_data.py
    01_build_sft_dataset.py
    02_train_lora.py
    03_chat_with_lora.py
    04_merge_lora.py
    05_preview_dataset.py
  src/style_finetune/
```

## 1. 安装环境

建议 Python 3.10 或 3.11。

先进入项目目录：

```powershell
cd style_finetune_gpu_code
```

安装依赖：

```powershell
pip install -r requirements.txt
```

检查 GPU：

```powershell
python scripts/99_env_check.py
```

看到 `CUDA available: True` 就可以继续。

## 2. 放入聊天数据

把你的原始文件放到：

```text
data/chunk_0001.jsonl
```

也可以不放进 `data/`，后续命令把 `--input` 改成你的实际路径。

## 3. 查看可训练对象

```powershell
python scripts/00_inspect_data.py --input data/chunk_0001.jsonl
```

它会列出每个发送者的可用文本数量。选择你要模仿的发送者名字

## 4. 生成 SFT 数据

例如模仿 ：

```powershell
python scripts/01_build_sft_dataset.py --input data/chunk_0001.jsonl --target-name "" --output-dir data/sft_fanzhenhan --context-turns 8
```

预览训练样本：

```powershell
python scripts/05_preview_dataset.py --file data/sft_fanzhenhan/train.jsonl --n 3
```

生成的每条样本格式是：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"最近聊天上下文..."},{"role":"assistant","content":"目标人物真实回复"}]}
```

训练脚本会只对最后的 `assistant` 回复计算 loss，前面的 system 和 user 上下文会被 mask 掉。

## 5. 按显存选择配置

| GPU 显存 | 推荐配置 | 命令中的 `--config` |
|---:|---|---|
| 8GB 到 12GB | Qwen2.5-1.5B-Instruct + QLoRA | `configs/qwen2_5_1_5b_qlora.yaml` |
| 16GB 到 24GB | Qwen2.5-3B-Instruct + QLoRA | `configs/qwen2_5_3b_qlora.yaml` |
| 24GB 以上 | 可尝试 Qwen2.5-7B-Instruct + QLoRA | `configs/qwen2_5_7b_qlora.yaml` |

数据量只有几百到一两千条时，1.5B 或 3B 往往比 7B 更稳。7B 可能更容易背聊天记录。

## 6. 开始训练

1.5B 默认配置：

```powershell
python scripts/02_train_lora.py --config configs/qwen2_5_1_5b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-lora
```

3B 配置：

```powershell
python scripts/02_train_lora.py --config configs/qwen2_5_3b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-3b-lora
```

7B 配置：

```powershell
python scripts/02_train_lora.py --config configs/qwen2_5_7b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-7b-lora
```

断点续训示例：

```powershell
python scripts/02_train_lora.py --config configs/qwen2_5_1_5b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-lora --resume-from-checkpoint outputs/fanzhenhan-lora/checkpoint-50
```

## 7. 训练后聊天测试

如果训练的是 1.5B：

```powershell
python scripts/03_chat_with_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct --lora-dir outputs/fanzhenhan-lora --load-in-4bit
```

如果训练的是 3B：

```powershell
python scripts/03_chat_with_lora.py --base-model Qwen/Qwen2.5-3B-Instruct --lora-dir outputs/fanzhenhan-3b-lora --load-in-4bit
```

如果训练的是 7B：

```powershell
python scripts/03_chat_with_lora.py --base-model Qwen/Qwen2.5-7B-Instruct --lora-dir outputs/fanzhenhan-7b-lora --load-in-4bit
```

交互命令：

```text
/clear  清空上下文
/exit   退出
```

## 8. 可选：合并 LoRA

合并后得到一个独立模型目录，方便部署推理。合并通常需要更多显存；如果显存不够，不合并也可以直接用 LoRA 目录推理。

```powershell
python scripts/04_merge_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct --lora-dir outputs/fanzhenhan-lora --output-dir outputs/fanzhenhan-merged --safe-serialization
```

## 9. 多卡训练

先配置 accelerate：

```powershell
accelerate config
```

然后启动训练：

```powershell
accelerate launch scripts/02_train_lora.py --config configs/qwen2_5_3b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-3b-lora
```

多卡时仍建议先单卡跑通数据转换和小规模训练。

## 10. 常见问题

### CUDA out of memory

优先改配置文件：

```yaml
training:
  max_seq_length: 512
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
lora:
  r: 8
  lora_alpha: 16
```

或者换 1.5B 模型。

### 训练后回复很像背原文

数据量小的时候常见。把配置改保守：

```yaml
training:
  num_train_epochs: 1
  learning_rate: 0.00005
lora:
  r: 8
  lora_alpha: 16
```

### 回复风格不明显

可以适当增加：

```yaml
training:
  num_train_epochs: 3
  learning_rate: 0.00015
lora:
  r: 16
```

也可以在生成数据时把 `--context-turns` 提高到 10 或 12。

## 11. 推荐起步命令

下面四条命令都可以直接复制到 PowerShell，每条都是单行。

```powershell
python scripts/00_inspect_data.py --input data/chunk_0001.jsonl
```

```powershell
python scripts/01_build_sft_dataset.py --input data/chunk_0001.jsonl --target-name "樊真菡" --output-dir data/sft_fanzhenhan --context-turns 8
```

```powershell
python scripts/02_train_lora.py --config configs/qwen2_5_1_5b_qlora.yaml --train-file data/sft_fanzhenhan/train.jsonl --val-file data/sft_fanzhenhan/val.jsonl --output-dir outputs/fanzhenhan-lora
```

```powershell
python scripts/03_chat_with_lora.py --base-model Qwen/Qwen2.5-1.5B-Instruct --lora-dir outputs/fanzhenhan-lora --load-in-4bit
```
