from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .corpus_text import (
    RunningStats,
    cjk_ratio,
    clean_text,
    dedupe_digest,
    quality_reject_reason,
    split_long_text,
)
from .utils import atomic_write_text


@dataclass(frozen=True)
class CorpusCleaningPolicy:
    text_column: str = "text"
    source_column: str = "source"
    score_column: str = "score"
    min_score: float = 4.0
    score_scale_max: float = 5.0
    min_chars: int = 200
    max_chars: int = 16_384
    min_cjk_ratio: float = 0.25
    normalize_unicode: bool = False
    dedupe_mode: str = "normalized"
    max_repeated_line_ratio: float = 0.35
    max_url_ratio: float = 0.20
    max_html_ratio: float = 0.10
    max_punctuation_ratio: float = 0.45
    max_digit_ratio: float = 0.35
    batch_size: int = 2_048
    max_records_per_shard: int = 50_000
    keep_provenance: bool = True

    def validate(self) -> None:
        if self.score_scale_max <= 0:
            raise ValueError("score_scale_max must be positive")
        if not 0 <= self.min_score <= self.score_scale_max:
            raise ValueError(
                "min_score must be between 0 and score_scale_max; "
                f"got min_score={self.min_score}, "
                f"score_scale_max={self.score_scale_max}"
            )
        if self.min_chars <= 0:
            raise ValueError("min_chars must be positive")
        if self.max_chars < self.min_chars:
            raise ValueError("max_chars must be greater than or equal to min_chars")
        if not 0 <= self.min_cjk_ratio <= 1:
            raise ValueError("min_cjk_ratio must be between 0 and 1")
        if self.dedupe_mode not in {"normalized", "exact", "none"}:
            raise ValueError("dedupe_mode must be normalized, exact, or none")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_records_per_shard <= 0:
            raise ValueError("max_records_per_shard must be positive")


@dataclass(frozen=True)
class ParquetTextRecord:
    text: str
    source: str
    score: float
    source_file: str
    source_row: int


class JsonlShardWriter:
    """Write deterministic UTF-8 JSONL shards while hashing bytes in one pass."""

    def __init__(self, output_dir: Path, max_records_per_shard: int) -> None:
        self.output_dir = output_dir
        self.max_records_per_shard = int(max_records_per_shard)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.shards: list[dict[str, Any]] = []
        self.total_records = 0
        self._index = -1
        self._records_in_shard = 0
        self._bytes_in_shard = 0
        self._digest: hashlib._Hash | None = None
        self._handle = None
        self._path: Path | None = None

    def _open_next(self) -> None:
        self._index += 1
        self._path = self.output_dir / f"clean_{self._index:05d}.jsonl"
        self._handle = self._path.open("wb")
        self._digest = hashlib.sha256()
        self._records_in_shard = 0
        self._bytes_in_shard = 0

    def _close_current(self) -> None:
        if self._handle is None or self._path is None or self._digest is None:
            return
        self._handle.flush()
        self._handle.close()
        self.shards.append(
            {
                "file": self._path.name,
                "records": self._records_in_shard,
                "size_bytes": self._bytes_in_shard,
                "sha256": self._digest.hexdigest(),
            }
        )
        self._handle = None
        self._path = None
        self._digest = None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            self._open_next()
        elif self._records_in_shard >= self.max_records_per_shard:
            self._close_current()
            self._open_next()
        assert self._handle is not None and self._digest is not None
        encoded = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self._handle.write(encoded)
        self._digest.update(encoded)
        self._records_in_shard += 1
        self._bytes_in_shard += len(encoded)
        self.total_records += 1

    def close(self) -> None:
        self._close_current()


def load_completed_parquet_paths(
    manifest_path: str | Path,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], list[Path]]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if require_complete and manifest.get("status") != "completed":
        raise ValueError(
            f"Download manifest is not complete: status={manifest.get('status')!r}, path={path}"
        )
    completed = {
        str(item["repo_path"]): Path(str(item["local_path"])).resolve()
        for item in manifest.get("completed_files", [])
    }
    selected = [str(item) for item in manifest.get("selected_repo_paths", [])]
    missing = [
        repo_path
        for repo_path in selected
        if repo_path not in completed or not completed[repo_path].is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} selected Parquet files are missing; first={missing[:3]}"
        )
    return manifest, [completed[repo_path] for repo_path in selected]


def load_jsonl_dedupe_seed(
    input_dir: str | Path,
    *,
    dedupe_mode: str,
    expected_records: int | None = None,
    expected_shard_names: Iterable[str] | None = None,
    progress_interval: int = 500_000,
) -> tuple[set[bytes], dict[str, Any]]:
    """Build an exact dedupe seed from an existing cleaned JSONL corpus."""

    if dedupe_mode == "none":
        raise ValueError("Incremental corpus cleaning requires dedupe_mode != 'none'")
    if dedupe_mode not in {"normalized", "exact"}:
        raise ValueError("dedupe_mode must be normalized or exact")

    directory = Path(input_dir).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Cleaned seed directory does not exist: {directory}")

    if expected_shard_names is None:
        shard_paths = sorted(directory.glob("*.jsonl"))
    else:
        names = [str(name) for name in expected_shard_names]
        invalid_names = [name for name in names if Path(name).name != name]
        if invalid_names:
            raise ValueError(
                f"Expected shard names must be file names; first={invalid_names[:3]}"
            )
        shard_paths = [directory / name for name in names]

    if not shard_paths:
        raise FileNotFoundError(f"No cleaned JSONL shards found in {directory}")
    missing = [path.name for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} cleaned seed shards are missing; first={missing[:3]}"
        )

    digests: set[bytes] = set()
    records = 0
    duplicate_records = 0
    started = time.monotonic()
    for shard_path in shard_paths:
        with shard_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in cleaned seed: file={shard_path}, "
                        f"line={line_number}"
                    ) from exc
                text = payload.get("text") if isinstance(payload, dict) else None
                if not isinstance(text, str) or not text:
                    raise ValueError(
                        f"Missing non-empty text in cleaned seed: file={shard_path}, "
                        f"line={line_number}"
                    )
                digest = dedupe_digest(text, dedupe_mode)
                if digest is None:  # pragma: no cover - guarded above
                    raise AssertionError("Dedupe seed unexpectedly produced no digest")
                records += 1
                if digest in digests:
                    duplicate_records += 1
                else:
                    digests.add(digest)
                if progress_interval > 0 and records % progress_interval == 0:
                    elapsed = max(1e-6, time.monotonic() - started)
                    print(
                        f"seeded={records} unique_digests={len(digests)} "
                        f"records_per_s={records / elapsed:.1f}",
                        flush=True,
                    )

    if expected_records is not None and records != int(expected_records):
        raise ValueError(
            f"Cleaned seed record count mismatch: actual={records}, "
            f"expected={int(expected_records)}"
        )

    stats = {
        "input_dir": str(directory),
        "shards": len(shard_paths),
        "records": records,
        "unique_digests": len(digests),
        "duplicate_records": duplicate_records,
        "dedupe_mode": dedupe_mode,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return digests, stats


def iter_parquet_records(
    paths: Iterable[Path],
    policy: CorpusCleaningPolicy,
) -> Iterator[ParquetTextRecord]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "pyarrow is required for Parquet cleaning. Install requirements-data.txt."
        ) from exc

    columns = [policy.text_column, policy.source_column, policy.score_column]
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing_columns = [
            column for column in columns if column not in parquet.schema_arrow.names
        ]
        if missing_columns:
            raise KeyError(f"Missing columns {missing_columns} in {path}")
        row_offset = 0
        for batch in parquet.iter_batches(
            batch_size=policy.batch_size,
            columns=columns,
        ):
            values = batch.to_pydict()
            for index, raw_text in enumerate(values[policy.text_column]):
                raw_source = values[policy.source_column][index]
                raw_score = values[policy.score_column][index]
                if isinstance(raw_text, str):
                    try:
                        score = float(raw_score)
                    except (TypeError, ValueError):
                        score = float("-inf")
                    yield ParquetTextRecord(
                        text=raw_text,
                        source=str(raw_source or "unknown"),
                        score=score,
                        source_file=path.name,
                        source_row=row_offset + index,
                    )
            row_offset += batch.num_rows


def _quality_reason(text: str, policy: CorpusCleaningPolicy) -> str | None:
    if len(text) < policy.min_chars:
        return "too_short"
    if len(text) > policy.max_chars:
        return "too_long_after_split"
    if cjk_ratio(text) < policy.min_cjk_ratio:
        return "low_cjk_ratio"
    return quality_reject_reason(
        text,
        max_repeated_line_ratio=policy.max_repeated_line_ratio,
        max_url_ratio=policy.max_url_ratio,
        max_html_ratio=policy.max_html_ratio,
        max_punctuation_ratio=policy.max_punctuation_ratio,
        max_digit_ratio=policy.max_digit_ratio,
    )


def _output_record(
    *,
    text: str,
    record: ParquetTextRecord,
    chunk_index: int,
    policy: CorpusCleaningPolicy,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "text": text,
        "source": record.source,
        "score": record.score,
    }
    if policy.keep_provenance:
        output.update(
            {
                "source_file": record.source_file,
                "source_row": record.source_row,
                "chunk_index": chunk_index,
            }
        )
    return output


def clean_parquet_corpus(
    *,
    input_paths: list[Path],
    output_dir: str | Path,
    report_path: str | Path,
    source_manifest: dict[str, Any],
    policy: CorpusCleaningPolicy,
    progress_interval: int = 50_000,
    seed_digests: set[bytes] | None = None,
    report_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conservatively clean score_4_5 Parquet into auditable text JSONL shards."""

    policy.validate()
    final_dir = Path(output_dir).resolve()
    staging_dir = final_dir.with_name(f"{final_dir.name}.staging")
    report = Path(report_path).resolve()
    if final_dir.exists():
        raise FileExistsError(f"Cleaned output already exists: {final_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"Staging output already exists, likely from an interrupted run: {staging_dir}"
        )

    writer = JsonlShardWriter(
        output_dir=staging_dir,
        max_records_per_shard=policy.max_records_per_shard,
    )
    raw_lengths = RunningStats()
    cleaned_lengths = RunningStats()
    kept_lengths = RunningStats()
    drop_reasons: Counter[str] = Counter()
    raw_by_source: Counter[str] = Counter()
    kept_by_source: Counter[str] = Counter()
    seed = seed_digests if seed_digests is not None else set()
    seen: set[bytes] = set()
    raw_documents = 0
    candidate_chunks = 0
    chunked_documents = 0
    started = time.monotonic()

    try:
        for record in iter_parquet_records(input_paths, policy):
            raw_documents += 1
            raw_by_source[record.source] += 1
            raw_lengths.add(len(record.text))
            if record.score < policy.min_score:
                drop_reasons["score_below_minimum"] += 1
                continue

            text = clean_text(record.text, normalize_unicode=policy.normalize_unicode)
            cleaned_lengths.add(len(text))
            if not text:
                drop_reasons["empty_text_after_clean"] += 1
                continue
            if "\ufffd" in text:
                drop_reasons["replacement_character"] += 1
                continue

            chunks = split_long_text(
                text,
                max_len=policy.max_chars,
                min_len=policy.min_chars,
            )
            if len(chunks) > 1:
                chunked_documents += 1
            if not chunks:
                drop_reasons["no_valid_chunk"] += 1
                continue

            for chunk_index, chunk in enumerate(chunks):
                candidate_chunks += 1
                reason = _quality_reason(chunk, policy)
                if reason:
                    drop_reasons[reason] += 1
                    continue
                digest = dedupe_digest(chunk, policy.dedupe_mode)
                if digest is not None:
                    if digest in seed:
                        drop_reasons[
                            f"duplicate_against_seed_{policy.dedupe_mode}"
                        ] += 1
                        continue
                    if digest in seen:
                        drop_reasons[f"duplicate_{policy.dedupe_mode}"] += 1
                        continue
                    seen.add(digest)
                writer.write(
                    _output_record(
                        text=chunk,
                        record=record,
                        chunk_index=chunk_index,
                        policy=policy,
                    )
                )
                kept_lengths.add(len(chunk))
                kept_by_source[record.source] += 1

            if progress_interval > 0 and raw_documents % progress_interval == 0:
                elapsed = max(1e-6, time.monotonic() - started)
                print(
                    f"processed={raw_documents} kept={writer.total_records} "
                    f"dropped_candidates={sum(drop_reasons.values())} "
                    f"docs_per_s={raw_documents / elapsed:.1f}",
                    flush=True,
                )
    finally:
        writer.close()

    if writer.total_records == 0:
        raise RuntimeError("No records survived corpus cleaning.")
    staging_dir.replace(final_dir)

    result = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "completed",
        "source_dataset": source_manifest.get("dataset"),
        "source_selection": source_manifest.get("selection"),
        "input_files": len(input_paths),
        "output_dir": str(final_dir),
        "policy": asdict(policy),
        "dedupe_seed_unique_digests": len(seed),
        "new_unique_digests": len(seen),
        "raw_documents": raw_documents,
        "candidate_chunks": candidate_chunks,
        "chunked_documents": chunked_documents,
        "kept_records": writer.total_records,
        "drop_reasons": dict(drop_reasons),
        "raw_by_source": dict(raw_by_source),
        "kept_by_source": dict(kept_by_source),
        "raw_length": raw_lengths.as_dict(),
        "cleaned_length": cleaned_lengths.as_dict(),
        "kept_length": kept_lengths.as_dict(),
        "output_shards": writer.shards,
        "output_bytes": sum(int(shard["size_bytes"]) for shard in writer.shards),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if report_metadata:
        result["metadata"] = dict(report_metadata)
    atomic_write_text(report, json.dumps(result, ensure_ascii=False, indent=2))
    return result
