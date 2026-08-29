from __future__ import annotations

import argparse
import json
from pathlib import Path

from muddywater.text_simplification.sol_distillation import load_and_validate_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one GPT-5.6-sol JSONL shard.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, issues = load_and_validate_shard(args.input, args.output)
    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    report = {
        "valid_rows": len(rows),
        "errors": [issue.to_dict() for issue in errors],
        "warnings": [issue.to_dict() for issue in warnings],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
