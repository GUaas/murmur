# Murmur

Murmur 是一个面向中文的紧凑型 Decoder-only Transformer 工程。本仓库提供从语料清洗、分词器构建、预训练、断点续训、全参数 SFT、评估到推理的完整模块化代码，以及 Murmur 203M 的可移植配置。

> English summary: Murmur is a compact Chinese decoder-only Transformer project. This repository contains the complete modular training, evaluation, and inference stack. Model weights are distributed separately through GitHub Releases.

## 开源范围

本仓库包含：

- 完整训练与推理源码：`muddywater/`
- 数据清洗、缓存、训练、评估和发布脚本：`scripts/`
- 文本简化数据管线与长文本推理代码
- 单元测试：`tests/`
- Murmur 203M 预训练与文本简化配置：`configs/`
- 32K SentencePiece tokenizer：`tokenizer/sp_unigram_32k.model`
- 模型结构、校验值和使用说明：`MODEL_CARD.md`、`models/README.md`

本仓库不包含训练数据、原始蒸馏数据、训练日志、内部评估结果、缓存、密钥或优化器状态。大模型权重不进入 Git 历史，而通过 GitHub Releases 发布。

## Murmur 203M 结构

| 项目 | 配置 |
| --- | --- |
| 参数量 | 203,037,056 |
| 层数 | 20 |
| 隐藏维度 | 896 |
| Attention heads / KV heads | 14 / 2 |
| 上下文长度 | 基础预训练 2,048 tokens；文本简化 896 tokens |
| 词表 | 32,000 |
| 归一化 / MLP / 位置编码 | RMSNorm / SwiGLU / RoPE |
| 其他 | tied embeddings、QK normalization、无 bias |

## 安装

需要 Python 3.10+ 和 PyTorch 2.2+。CUDA 训练环境建议使用 BF16。

```bash
git clone https://github.com/GUaas/murmur.git
cd murmur
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## 从头预训练

输入语料使用 UTF-8 JSONL，每行至少包含一个 `text` 字段：

```json
{"text":"要用于预训练的一段中文文本。"}
```

先构建分片 token cache：

```bash
python scripts/prepare_pretrain_cache.py \
  --input "data/pretrain/raw/*.jsonl" \
  --tokenizer tokenizer/sp_unigram_32k.model \
  --output-dir data/pretrain/cache \
  --jsonl-text-key text \
  --max-tokens-per-shard 100000000
```

检查并按硬件、数据量调整 `configs/pretrain_203m_example.yaml`，再启动训练：

```bash
bash run_pretrain.sh
```

该配置是可运行的 203M 起点模板，不冒充原始预训练运行的完整超参数复刻。训练代码会记录配置、运行环境、诊断、指标和断点信息。

## 文本简化 SFT

1. 从 Releases 下载基础预训练权重到 `model/murmur_203m_base_weights_only.pt`。
2. 准备成对数据，推荐字段为 `source` 和 `target`。
3. 将处理结果放到 `data/text_simplification/processed/`，或修改配置中的路径。
4. 启动全参数 SFT。

```bash
python scripts/prepare_text_simplification_data.py \
  --input /path/to/dataset.jsonl \
  --output-dir data/text_simplification/processed \
  --source-key source \
  --target-key target

bash run_text_simplification_sft.sh
```

训练与推理协议为：

```text
<|im_start|>{source}<|im_end|>{target}<eos>
```

损失仅覆盖目标文本和 EOS，源文本与结构标签不参与损失。

## 推理

从 Releases 下载文本简化权重到：

```text
model/murmur_203m_text_simplification_best_weights_only.pt
```

Linux：

```bash
./run_simplify.sh "由于近期连续出现强降雨天气，相关部门决定暂时关闭部分山区景区。"
```

Windows PowerShell：

```powershell
.\run_simplify.ps1 -Text "由于近期连续出现强降雨天气，相关部门决定暂时关闭部分山区景区。"
```

## 验证

```bash
pytest -q
python scripts/validate_text_simplification_setup.py \
  --config configs/sft_text_simplification_203m.yaml
```

## 模型与许可证

代码按 Apache License 2.0 发布。模型权重的许可证、训练数据来源、用途限制和 SHA-256 以对应 Release 与 `MODEL_CARD.md` 为准；数据集不会随仓库分发。

在生产环境使用前，请自行进行安全、偏差、事实性、隐私和提示注入测试。Murmur 不是医疗、法律、金融或其他高风险场景的替代决策系统。
