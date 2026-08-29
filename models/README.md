# Model downloads

Model files are intentionally excluded from Git history because each checkpoint is hundreds of megabytes. Download verified weights from the repository's GitHub Releases page.

Expected layout:

```text
model/
├── murmur_203m_base_weights_only.pt
└── murmur_203m_text_simplification_best_weights_only.pt
```

With GitHub CLI:

```bash
gh release download v0.1.0 --repo GUaas/murmur --pattern "*.pt" --dir model
```

Always compare SHA-256 values with the Release notes and `MODEL_CARD.md` before loading a checkpoint. PyTorch checkpoints are executable serialization formats; only load files from a trusted release.
