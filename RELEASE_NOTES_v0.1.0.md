# Murmur v0.1.0

首个公开版本，包含完整的模块化训练工程和两份 Murmur 203M weights-only 权重。

## Repository

- 语料清洗与 token cache 构建
- SentencePiece tokenizer 构建与校验
- Decoder-only Transformer 预训练、断点续训与诊断
- 全参数文本简化 SFT
- 评估、长文本推理与发布工具
- 86 项通过的自动化测试

训练数据、完整日志、未筛选的内部评估产物、缓存、密钥和优化器状态不随仓库发布。经过复核的训练指标、自动评估结果和实际推理样例见 [`EVALUATION.md`](https://github.com/GUaas/murmur/blob/main/EVALUATION.md)。

## Evaluation highlights

- Base checkpoint: step 68,000; best validation CE loss 2.911759; PPL 18.3891
- Text simplification: best validation CE loss 0.502785 at step 350
- 200-pair validation sample: SARI 0.716924; ROUGE-L 0.915904; chrF 0.790481
- Number preservation on 56 numbered samples: precision 0.991071; recall 1.000000
- Full methodology, caveats, loss history, and inference examples: [EVALUATION.md](https://github.com/GUaas/murmur/blob/main/EVALUATION.md)

## Model assets

### Base pretrained model

- File: `murmur_203m_base_weights_only.pt`
- Parameters: 203,037,056
- Context: 2,048 tokens
- Bytes: 812,223,200
- SHA-256: `5406a6ac67c47fe53eb0d65eff4f490aca200cdb89d172fdc8f56f24d2dd0297`

### Text simplification model

- File: `murmur_203m_text_simplification_best_weights_only.pt`
- Parameters: 203,037,056
- Context: 896 tokens
- Bytes: 812,221,131
- SHA-256: `da9690eb4a6806e8570f0df83da3c6149f1a8cd29b423d568be907e1ef913777`

## Download

```bash
gh release download v0.1.0 --repo GUaas/murmur --pattern "*.pt" --dir model
```

下载后请先核对 SHA-256，再加载 PyTorch 权重。
