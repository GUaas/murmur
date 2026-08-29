from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))

from muddywater.source_target import SourceTargetTemplate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_dataset(
    input_path: Path,
    output_path: Path,
    *,
    input_source_key: str,
    input_target_key: str,
    output_source_key: str = "source",
    output_target_key: str = "target",
    target_separator: str = "<|im_start|>",
    overwrite: bool = False,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template = SourceTargetTemplate(
        source_key=output_source_key,
        target_key=output_target_key,
        target_separator=target_separator,
    )
    rows = 0
    source_chars = 0
    target_chars = 0
    max_source_chars = 0
    max_target_chars = 0
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as output_handle:
            temp_path = Path(output_handle.name)
            with input_path.open("r", encoding="utf-8-sig") as input_handle:
                for line_no, raw_line in enumerate(input_handle, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSONL at {input_path}:{line_no}: {exc}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"Expected a JSON object at {input_path}:{line_no}")
                    if input_source_key not in record or input_target_key not in record:
                        raise ValueError(
                            f"Missing {input_source_key!r} or {input_target_key!r} "
                            f"at {input_path}:{line_no}"
                        )
                    try:
                        source, target = template.normalize_pair(
                            record[input_source_key], record[input_target_key]
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid source/target at {input_path}:{line_no}: {exc}"
                        ) from exc

                    output_record = {
                        output_source_key: source,
                        output_target_key: target,
                    }
                    output_handle.write(
                        json.dumps(output_record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
                    rows += 1
                    source_chars += len(source)
                    target_chars += len(target)
                    max_source_chars = max(max_source_chars, len(source))
                    max_target_chars = max(max_target_chars, len(target))

        assert temp_path is not None
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return {
        "status": "converted",
        "input": str(input_path),
        "output": str(output_path),
        "rows": rows,
        "source_key": output_source_key,
        "target_key": output_target_key,
        "target_separator": target_separator,
        "source_chars": source_chars,
        "target_chars": target_chars,
        "max_source_chars": max_source_chars,
        "max_target_chars": max_target_chars,
        "input_sha256": sha256_file(input_path),
        "output_sha256": sha256_file(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired JSONL into compact source/target JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-source-key", default="原文")
    parser.add_argument("--input-target-key", default="简化")
    parser.add_argument("--output-source-key", default="source")
    parser.add_argument("--output-target-key", default="target")
    parser.add_argument("--target-separator", default="<|im_start|>")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        args.input,
        args.output,
        input_source_key=args.input_source_key,
        input_target_key=args.input_target_key,
        output_source_key=args.output_source_key,
        output_target_key=args.output_target_key,
        target_separator=args.target_separator,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
