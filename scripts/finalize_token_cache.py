from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.utils import atomic_write_text, file_sha256
from scripts.verify_pretrain_bundle import validate_cache_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate every token-cache shard and atomically promote staging."
    )
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def shard_paths(root: Path, entries: list[dict]) -> list[Path]:
    paths: list[Path] = []
    for entry in entries:
        name = str(entry.get("file", "")).strip()
        if not name:
            raise ValueError("Cache manifest contains a shard without a file name")
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Cache shard is missing: {path}")
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    staging_dir = Path(args.staging_dir).resolve()
    final_dir = Path(args.final_dir).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    if not staging_dir.is_dir():
        raise FileNotFoundError(f"Staging cache does not exist: {staging_dir}")
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite final cache: {final_dir}")

    manifest = read_json(staging_dir / "manifest.json")
    stats = read_json(staging_dir / "prepare_stats.json")
    tokenizer_sha256 = file_sha256(tokenizer_path)
    if str(manifest.get("tokenizer_sha256", "")).lower() != tokenizer_sha256.lower():
        raise ValueError("Cache manifest tokenizer SHA-256 does not match tokenizer")
    if int(manifest.get("vocab_size", -1)) != 32_000:
        raise ValueError("Cache manifest vocab_size is not 32000")
    if str(manifest.get("dtype")) != "uint16":
        raise ValueError("Cache dtype is not uint16")
    if int(stats.get("unk_tokens", -1)) != 0:
        raise ValueError("Cache preparation reported unknown tokens")

    started = time.time()
    train = validate_cache_split(
        token_paths=shard_paths(staging_dir, list(manifest.get("train_shards", []))),
        expected_documents=int(manifest["train_documents"]),
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=32_000,
        add_bos=True,
        max_seq_len=int(args.max_seq_len),
        document_attention=False,
        ignore_cross_document_targets=False,
    )
    print(
        f"validated_train_shards={train['shard_count']} "
        f"tokens={train['tokens']} bytes={train['bytes']}",
        flush=True,
    )
    validation = validate_cache_split(
        token_paths=shard_paths(staging_dir, list(manifest.get("val_shards", []))),
        expected_documents=int(manifest["val_documents"]),
        tokenizer_sha256=tokenizer_sha256,
        vocab_size=32_000,
        add_bos=True,
        max_seq_len=int(args.max_seq_len),
        document_attention=False,
        ignore_cross_document_targets=False,
    )
    print(
        f"validated_val_shards={validation['shard_count']} "
        f"tokens={validation['tokens']} bytes={validation['bytes']}",
        flush=True,
    )

    total_documents = int(train["documents"]) + int(validation["documents"])
    total_tokens = int(train["tokens"]) + int(validation["tokens"])
    if total_documents != int(stats["documents"]):
        raise ValueError(
            f"Total document mismatch: validated={total_documents}, stats={stats['documents']}"
        )
    if total_tokens != int(stats["train_tokens"]) + int(stats["val_tokens"]):
        raise ValueError("Validated token totals do not match prepare_stats.json")

    report = {
        "schema_version": 1,
        "status": "completed",
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": 32_000,
        "max_seq_len": int(args.max_seq_len),
        "documents": total_documents,
        "tokens": total_tokens,
        "unk_tokens": 0,
        "train": train,
        "validation": validation,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = staging_dir / "validation_report.json"
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2),
        overwrite=False,
    )
    staging_dir.rename(final_dir)
    print(
        json.dumps(
            {
                "status": "completed",
                "final_dir": str(final_dir),
                "validation_report": str(final_dir / report_path.name),
                "documents": total_documents,
                "tokens": total_tokens,
                "train_shards": train["shard_count"],
                "val_shards": validation["shard_count"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
