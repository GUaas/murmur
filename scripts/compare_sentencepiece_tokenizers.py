from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from muddywater.dataset import discover_data_files
from muddywater.tokenizer import BPETokenizer


@dataclass
class TokenizerMetrics:
    tokenizer: BPETokenizer
    tokens: int = 0
    unknown_tokens: int = 0
    byte_tokens: int = 0
    exact_round_trips: int = 0

    def add(self, text: str) -> None:
        ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        self.tokens += len(ids)
        self.unknown_tokens += sum(
            1 for token_id in ids if int(token_id) == self.tokenizer.unk_id
        )
        self.byte_tokens += sum(
            1
            for token_id in ids
            if bool(self.tokenizer.processor.is_byte(int(token_id)))
        )
        if self.tokenizer.decode(ids, skip_special_tokens=False) == text:
            self.exact_round_trips += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two SentencePiece tokenizers on the same held-out corpus."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_texts(paths: Iterable[str | Path], text_key: str) -> Iterable[str]:
    for path in discover_data_files(paths):
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                text = payload.get(text_key)
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"Missing text at {path}:{line_number}")
                yield text.strip()


def tokenizer_identity(path: Path, tokenizer: BPETokenizer) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "vocab_size": tokenizer.vocab_size,
        "special_ids": {
            "pad": tokenizer.pad_id,
            "unk": tokenizer.unk_id,
            "bos": tokenizer.bos_id,
            "eos": tokenizer.eos_id,
        },
    }


def metric_payload(
    identity: dict[str, object],
    metrics: TokenizerMetrics,
    documents: int,
    chars: int,
    utf8_bytes: int,
) -> dict[str, object]:
    return {
        **identity,
        "documents": documents,
        "chars": chars,
        "utf8_bytes": utf8_bytes,
        "tokens": metrics.tokens,
        "chars_per_token": round(chars / metrics.tokens, 6),
        "utf8_bytes_per_token": round(utf8_bytes / metrics.tokens, 6),
        "unknown_tokens": metrics.unknown_tokens,
        "unknown_token_rate": round(metrics.unknown_tokens / metrics.tokens, 10),
        "byte_tokens": metrics.byte_tokens,
        "byte_token_rate": round(metrics.byte_tokens / metrics.tokens, 10),
        "exact_round_trips": metrics.exact_round_trips,
        "exact_round_trip_rate": round(metrics.exact_round_trips / documents, 8),
    }


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite comparison report: {output_path}")

    baseline = BPETokenizer.load(baseline_path)
    candidate = BPETokenizer.load(candidate_path)
    if not isinstance(baseline, BPETokenizer) or not isinstance(candidate, BPETokenizer):
        raise TypeError("Both comparison inputs must be SentencePiece tokenizers")
    baseline_metrics = TokenizerMetrics(baseline)
    candidate_metrics = TokenizerMetrics(candidate)

    documents = 0
    chars = 0
    utf8_bytes = 0
    for text in iter_texts(args.input, text_key=args.text_key):
        documents += 1
        chars += len(text)
        utf8_bytes += len(text.encode("utf-8"))
        baseline_metrics.add(text)
        candidate_metrics.add(text)
    if documents == 0:
        raise ValueError("No held-out texts were found")

    baseline_payload = metric_payload(
        tokenizer_identity(baseline_path, baseline),
        baseline_metrics,
        documents,
        chars,
        utf8_bytes,
    )
    candidate_payload = metric_payload(
        tokenizer_identity(candidate_path, candidate),
        candidate_metrics,
        documents,
        chars,
        utf8_bytes,
    )
    token_reduction = 1.0 - candidate_metrics.tokens / baseline_metrics.tokens
    chars_per_token_gain = (
        candidate_payload["chars_per_token"] / baseline_payload["chars_per_token"] - 1.0
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "held_out_inputs": [str(path) for path in args.input],
        "baseline": baseline_payload,
        "candidate": candidate_payload,
        "comparison": {
            "candidate_token_reduction": round(token_reduction, 8),
            "candidate_token_reduction_percent": round(token_reduction * 100, 4),
            "candidate_chars_per_token_gain": round(chars_per_token_gain, 8),
            "candidate_chars_per_token_gain_percent": round(
                chars_per_token_gain * 100, 4
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
