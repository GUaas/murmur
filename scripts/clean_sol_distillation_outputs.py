from __future__ import annotations

import argparse
import json
from pathlib import Path

from muddywater.text_simplification.sol_distillation import (
    read_jsonl,
    revert_surface_only_changes,
    validate_output_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Revert punctuation/spacing-only distillation changes.")
    parser.add_argument("--work-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    work_dir = parse_args().work_dir.resolve()
    manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    reverted: list[str] = []
    for shard in manifest["shards"]:
        input_rows = read_jsonl(work_dir / shard["input"])
        output_path = work_dir / shard["output"]
        output_rows = read_jsonl(output_path)
        cleaned, shard_reverted = revert_surface_only_changes(input_rows, output_rows)
        if shard_reverted:
            write_jsonl(output_path, cleaned)
            reverted.extend(shard_reverted)
        _, issues = validate_output_rows(input_rows, cleaned)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise ValueError(f"{output_path}: {len(errors)} validation errors after cleanup")
    print(json.dumps({"reverted": len(reverted), "keys": reverted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
