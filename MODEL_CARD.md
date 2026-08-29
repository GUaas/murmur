# Murmur 203M Model Card

## Models

### Murmur 203M Base

- Architecture: decoder-only Transformer
- Parameters: 203,037,056
- Context: 2,048 tokens
- Vocabulary: 32,000 SentencePiece tokens
- Layers / hidden size: 20 / 896
- Attention heads / KV heads: 14 / 2
- Normalization / MLP / positions: RMSNorm / SwiGLU / RoPE
- Intended use: Chinese language-model research, continued pretraining, and task-specific fine-tuning
- Checkpoint filename: `murmur_203m_base_weights_only.pt`
- Checkpoint size: 812,223,200 bytes
- SHA-256: `5406a6ac67c47fe53eb0d65eff4f490aca200cdb89d172fdc8f56f24d2dd0297`
- Recorded training step: 68,000
- Recorded best validation loss: 2.9117594164

The base checkpoint is a weights-only PyTorch state dictionary. Load it with the model definition in `muddywater/model.py` and the matching YAML configuration.

### Murmur 203M Text Simplification

- Base: Murmur 203M Base
- Context: 896 tokens
- Training method: full-parameter supervised fine-tuning
- Input protocol: `<|im_start|>{source}<|im_end|>`
- Target protocol: `{target}<eos>`
- Supervision: target and EOS tokens only
- Best validation loss recorded by the provided run: 0.5027853699
- Training steps: 537
- Supervised training tokens: 6,580,812
- Checkpoint filename: `murmur_203m_text_simplification_best_weights_only.pt`
- Checkpoint size: 812,221,131 bytes
- SHA-256: `da9690eb4a6806e8570f0df83da3c6149f1a8cd29b423d568be907e1ef913777`

The SFT corpus itself is not distributed. The repository owner states that the released checkpoint was not trained on GPT-5.6 Sol distilled data. Optional distillation utilities present in the training code do not describe the provenance of the released checkpoint.

## Tokenizer

- File: `tokenizer/sp_unigram_32k.model`
- Type: SentencePiece unigram
- Vocabulary size: 32,000
- Byte fallback: enabled
- SHA-256: `bc784f8816ec143bc53d1744d3e607872fb27a4ba4d689792915add6a78003d4`
- Reserved atomic tokens: `<|im_start|>`, `<|im_end|>`

## Limitations

- The models are compact research models and may generate incorrect, incomplete, repetitive, biased, or unsafe text.
- The 896-token context window is substantially shorter than modern long-context systems.
- Text simplification may remove important qualifications, numbers, entities, or causal relationships.
- The long-text pipeline processes chunks and may introduce cross-chunk inconsistencies.
- No claim is made that the models are suitable for medical, legal, financial, safety-critical, or autonomous decision-making use.

## Evaluation guidance

Evaluate on data that matches the intended domain. At minimum, measure semantic preservation, compression, hallucination, named-entity retention, number retention, special-token leakage, empty-output rate, and latency. Human review is required for high-impact content.

## License

Repository code is Apache-2.0. Each model checkpoint's release notes define its license after upstream data and provider rights have been verified; the repository license does not automatically grant rights to separately distributed model files.
