# Contributing

Use Python 3.10 or newer, install the development dependencies, and run the test suite before opening a pull request:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

Keep model, data, training, evaluation, and release logic in separate modules. Do not commit training corpora, checkpoints, tokens, API keys, local paths, generated reports, or other private artifacts.

Changes to model architecture, tokenizer behavior, checkpoint loading, data splitting, label masking, or resumption semantics must include focused tests and migration notes.
