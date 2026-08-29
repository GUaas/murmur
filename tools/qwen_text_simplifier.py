"""Qwen entry point for the strict one-record text-simplification pipeline."""

from __future__ import annotations

import sys

from kimi_text_simplifier import main


if __name__ == "__main__":
    if "--provider" not in sys.argv:
        sys.argv[1:1] = ["--provider", "qwen"]
    raise SystemExit(main())
