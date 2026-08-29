from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare identity-pair shards for GPT-5.6-sol.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=400)
    return parser.parse_args()


def load_identity_rows(input_dir: Path) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        path = input_dir / f"{split}.jsonl"
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                source = str(payload.get("source", ""))
                target = str(payload.get("target", ""))
                if source == target:
                    identities.append(
                        {
                            "key": f"{split}:{line_number}",
                            "split": split,
                            "line_number": line_number,
                            "source": source,
                        }
                    )
    return identities


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.shard_size < 1:
        raise ValueError("shard-size must be positive")
    input_dir = args.input_dir.resolve()
    work_dir = args.work_dir.resolve()
    input_shards = work_dir / "input_shards"
    output_shards = work_dir / "output_shards"
    input_shards.mkdir(parents=True, exist_ok=True)
    output_shards.mkdir(parents=True, exist_ok=True)
    identities = load_identity_rows(input_dir)
    shard_count = math.ceil(len(identities) / args.shard_size)
    manifest: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        start = shard_index * args.shard_size
        rows = identities[start : start + args.shard_size]
        filename = f"input_{shard_index:03d}.jsonl"
        write_jsonl(input_shards / filename, rows)
        manifest.append(
            {
                "index": shard_index,
                "input": f"input_shards/{filename}",
                "output": f"output_shards/output_{shard_index:03d}.jsonl",
                "rows": len(rows),
                "first_key": rows[0]["key"],
                "last_key": rows[-1]["key"],
            }
        )
    (work_dir / "manifest.json").write_text(
        json.dumps(
            {
                "teacher_model": "gpt-5.6-sol",
                "identity_rows": len(identities),
                "shard_size": args.shard_size,
                "shards": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"identity_rows": len(identities), "shards": shard_count, "work_dir": str(work_dir)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
