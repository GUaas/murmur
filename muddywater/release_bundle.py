from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .utils import atomic_write_text, file_sha256


RELEASE_MANIFEST_NAME = "release_manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"
RELEASE_SCHEMA_VERSION = 1
MAX_GITHUB_ASSET_BYTES = 2 * 1024**3
INSTALL_GROUPS = frozenset({"tokenizer", "token_cache"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CACHE_ARTIFACT_PATTERNS = (
    "train*.bin",
    "val*.bin",
    "*.bin.meta.json",
    "*.bin.doc_starts.npy",
    "manifest.json",
    "prepare_stats.json",
)


@dataclass(frozen=True)
class BundleSource:
    source_path: Path
    asset_name: str
    install_group: str
    install_name: str
    purpose: str


def _safe_basename(value: Any, field: str) -> str:
    name = str(value or "")
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"{field} must be a non-empty basename, got {name!r}")
    if "/" in name or "\\" in name or "\0" in name:
        raise ValueError(f"{field} must not contain path separators")
    return name


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {description} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _purpose_for_cache_file(name: str) -> str:
    if name == "manifest.json":
        return "token_cache_manifest"
    if name == "prepare_stats.json":
        return "token_cache_prepare_stats"
    if name.endswith(".bin.meta.json"):
        return "token_cache_shard_metadata"
    if name.endswith(".bin.doc_starts.npy"):
        return "token_cache_document_starts"
    if name.startswith("train") and name.endswith(".bin"):
        return "train_token_ids"
    if name.startswith("val") and name.endswith(".bin"):
        return "validation_token_ids"
    raise ValueError(f"Unsupported token-cache artifact: {name}")


def collect_cache_sources(cache_dir: str | Path) -> tuple[list[BundleSource], dict[str, Any]]:
    """Return the complete cache artifact set described by its cache manifest.

    Recognized cache files that are not referenced by the manifest are rejected.
    This catches interrupted rebuilds and old shards before release publication.
    """

    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Token-cache directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    manifest = _read_json_object(manifest_path, "token-cache manifest")
    expected_names = {"manifest.json", "prepare_stats.json"}

    saw_shard = False
    for split_key in ("train_shards", "val_shards"):
        shards = manifest.get(split_key)
        if not isinstance(shards, list):
            raise ValueError(f"Cache manifest field {split_key!r} must be a list")
        for index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                raise ValueError(f"{split_key}[{index}] must be an object")
            bin_name = _safe_basename(shard.get("file"), f"{split_key}[{index}].file")
            starts_name = _safe_basename(
                shard.get("doc_starts_file"),
                f"{split_key}[{index}].doc_starts_file",
            )
            expected_prefix = "train" if split_key == "train_shards" else "val"
            if not bin_name.startswith(expected_prefix) or not bin_name.endswith(".bin"):
                raise ValueError(
                    f"{split_key}[{index}].file has an unexpected split/name: {bin_name}"
                )
            if starts_name != f"{bin_name}.doc_starts.npy":
                raise ValueError(
                    f"{split_key}[{index}] document-start file does not match {bin_name}"
                )
            expected_names.update({bin_name, f"{bin_name}.meta.json", starts_name})
            saw_shard = True
    if not saw_shard:
        raise ValueError("Cache manifest contains no train or validation shards")

    recognized = {
        path.name
        for pattern in _CACHE_ARTIFACT_PATTERNS
        for path in root.glob(pattern)
        if path.is_file()
    }
    missing = sorted(name for name in expected_names if not (root / name).is_file())
    unexpected = sorted(recognized - expected_names)
    if missing:
        raise FileNotFoundError(f"Cache manifest references missing artifacts: {missing}")
    if unexpected:
        raise ValueError(
            "Cache directory contains stale/unreferenced cache artifacts: "
            f"{unexpected}"
        )

    sources = [
        BundleSource(
            source_path=root / name,
            asset_name=name,
            install_group="token_cache",
            install_name=name,
            purpose=_purpose_for_cache_file(name),
        )
        for name in sorted(expected_names)
    ]
    return sources, manifest


def _sentencepiece_piece_type(processor: Any, piece_id: int) -> str:
    predicates = (
        ("unknown", "is_unknown"),
        ("control", "is_control"),
        ("unused", "is_unused"),
        ("byte", "is_byte"),
    )
    for label, attribute in predicates:
        predicate = getattr(processor, attribute, None)
        if predicate is not None and bool(predicate(piece_id)):
            return label
    return "normal"


def export_sentencepiece_vocab_tsv(
    model_path: str | Path,
    output_path: str | Path,
    *,
    processor_factory: Callable[[], Any] | None = None,
) -> Path:
    """Export an auditable ``id/piece/score/type`` vocabulary table."""

    if processor_factory is None:
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - project dependency
            raise RuntimeError("sentencepiece is required to export vocabulary TSV") from exc
        processor_factory = spm.SentencePieceProcessor

    processor = processor_factory()
    loaded = processor.load(str(Path(model_path)))
    if loaded is False:
        raise ValueError(f"Could not load SentencePiece model: {model_path}")
    size = int(processor.get_piece_size())
    lines = ["id\tpiece\tscore\ttype"]
    for piece_id in range(size):
        piece = str(processor.id_to_piece(piece_id))
        # SentencePiece pieces should be single-line, but escaping keeps the TSV
        # structurally valid even for a custom/user-defined symbol.
        piece = piece.replace("\\", "\\\\").replace("\t", "\\t").replace("\r", "\\r").replace("\n", "\\n")
        score = float(processor.get_score(piece_id))
        piece_type = _sentencepiece_piece_type(processor, piece_id)
        lines.append(f"{piece_id}\t{piece}\t{score:.9g}\t{piece_type}")
    return atomic_write_text(output_path, "\n".join(lines) + "\n", overwrite=False)


def _assert_asset_size(path: Path) -> int:
    size = path.stat().st_size
    if size >= MAX_GITHUB_ASSET_BYTES:
        raise ValueError(
            f"GitHub Release asset must be smaller than 2 GiB: {path} ({size} bytes). "
            "Rebuild the token cache with smaller --max-tokens-per-shard."
        )
    return size


def _prepare_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Bundle output is not a directory: {path}")
        contents = list(path.iterdir())
        if contents:
            raise FileExistsError(
                f"Bundle output must be new or empty; found existing content in {path}"
            )
    else:
        path.mkdir(parents=True, exist_ok=False)


def _materialize(source: Path, destination: Path, mode: str) -> None:
    if mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError as exc:
            raise OSError(
                f"Could not hard-link {source} into {destination.parent}. "
                "Place --output-dir on the same filesystem or use --materialize copy."
            ) from exc
    elif mode == "copy":
        shutil.copyfile(source, destination)
    else:
        raise ValueError("materialize mode must be one of: hardlink, copy")


def _validate_unique_sources(sources: Iterable[BundleSource]) -> list[BundleSource]:
    result = list(sources)
    asset_names: set[str] = set()
    install_destinations: set[tuple[str, str]] = set()
    for source in result:
        _safe_basename(source.asset_name, "asset_name")
        _safe_basename(source.install_name, "install_name")
        if source.install_group not in INSTALL_GROUPS:
            raise ValueError(f"Unsupported install group: {source.install_group}")
        if source.asset_name in {RELEASE_MANIFEST_NAME, CHECKSUMS_NAME}:
            raise ValueError(f"Asset name is reserved: {source.asset_name}")
        if source.asset_name in asset_names:
            raise ValueError(f"Duplicate release asset name: {source.asset_name}")
        destination = (source.install_group, source.install_name)
        if destination in install_destinations:
            raise ValueError(f"Duplicate install destination: {destination}")
        asset_names.add(source.asset_name)
        install_destinations.add(destination)
    return result


def build_release_bundle(
    tokenizer_model: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    materialize: str = "hardlink",
    export_vocab_tsv: bool = True,
    include_native_vocab: bool = True,
    processor_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Create a flat, GitHub-Release-ready directory and integrity metadata."""

    tokenizer_path = Path(tokenizer_model)
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"SentencePiece model does not exist: {tokenizer_path}")
    if tokenizer_path.suffix != ".model":
        raise ValueError("Release tokenizer must be a SentencePiece .model file")
    _assert_asset_size(tokenizer_path)

    cache_sources, cache_manifest = collect_cache_sources(cache_dir)
    tokenizer_digest = file_sha256(tokenizer_path)
    manifest_digest = cache_manifest.get("tokenizer_sha256")
    if not isinstance(manifest_digest, str) or manifest_digest != tokenizer_digest:
        raise ValueError(
            "Token-cache manifest tokenizer_sha256 does not match the supplied tokenizer model"
        )

    sources = [
        BundleSource(
            source_path=tokenizer_path,
            asset_name=tokenizer_path.name,
            install_group="tokenizer",
            install_name=tokenizer_path.name,
            purpose="sentencepiece_model",
        ),
        *cache_sources,
    ]
    native_vocab = tokenizer_path.with_suffix(".vocab")
    if include_native_vocab and native_vocab.is_file():
        sources.append(
            BundleSource(
                source_path=native_vocab,
                asset_name=native_vocab.name,
                install_group="tokenizer",
                install_name=native_vocab.name,
                purpose="sentencepiece_native_vocab",
            )
        )
    diagnostics = tokenizer_path.with_suffix(
        tokenizer_path.suffix + ".diagnostics.json"
    )
    if diagnostics.is_file():
        sources.append(
            BundleSource(
                source_path=diagnostics,
                asset_name=diagnostics.name,
                install_group="tokenizer",
                install_name=diagnostics.name,
                purpose="sentencepiece_diagnostics",
            )
        )
    sources = _validate_unique_sources(sources)
    for source in sources:
        _assert_asset_size(source.source_path)

    output = Path(output_dir)
    _prepare_empty_directory(output)
    try:
        for source in sources:
            _materialize(source.source_path, output / source.asset_name, materialize)

        if export_vocab_tsv:
            tsv_name = tokenizer_path.with_suffix(".vocab.tsv").name
            if any(source.asset_name == tsv_name for source in sources):
                raise ValueError(f"Vocabulary TSV collides with another asset: {tsv_name}")
            export_sentencepiece_vocab_tsv(
                tokenizer_path,
                output / tsv_name,
                processor_factory=processor_factory,
            )
            sources.append(
                BundleSource(
                    source_path=output / tsv_name,
                    asset_name=tsv_name,
                    install_group="tokenizer",
                    install_name=tsv_name,
                    purpose="sentencepiece_vocab_tsv",
                )
            )

        records: list[dict[str, Any]] = []
        source_by_name = {source.asset_name: source for source in sources}
        for asset_name in sorted(source_by_name):
            source = source_by_name[asset_name]
            asset_path = output / asset_name
            records.append(
                {
                    "name": asset_name,
                    "install_group": source.install_group,
                    "install_name": source.install_name,
                    "purpose": source.purpose,
                    "size_bytes": _assert_asset_size(asset_path),
                    "sha256": file_sha256(asset_path),
                }
            )

        manifest: dict[str, Any] = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "bundle_type": "murmur_pretrain_assets",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "asset_limit_bytes_exclusive": MAX_GITHUB_ASSET_BYTES,
            "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
            "tokenizer_model": tokenizer_path.name,
            "assets": records,
        }
        atomic_write_text(
            output / RELEASE_MANIFEST_NAME,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            overwrite=False,
        )
        checksum_text = "".join(
            f"{record['sha256']}  {record['name']}\n" for record in records
        )
        atomic_write_text(output / CHECKSUMS_NAME, checksum_text, overwrite=False)
        return manifest
    except Exception:
        # A failed build must not leave a directory that looks publishable.
        shutil.rmtree(output, ignore_errors=True)
        raise


def parse_sha256sums(path: str | Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_no, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line:
            continue
        if len(raw_line) < 67 or raw_line[64:66] != "  ":
            raise ValueError(f"Invalid SHA256SUMS line {line_no}")
        digest = raw_line[:64].lower()
        name = _safe_basename(raw_line[66:], f"SHA256SUMS line {line_no} filename")
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"Invalid SHA-256 at SHA256SUMS line {line_no}")
        if name in checksums:
            raise ValueError(f"Duplicate SHA256SUMS filename: {name}")
        checksums[name] = digest
    return checksums


def validate_downloaded_bundle(
    directory: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate release metadata, exact file set, sizes and every asset digest."""

    root = Path(directory)
    manifest_path = root / RELEASE_MANIFEST_NAME
    checksums_path = root / CHECKSUMS_NAME
    if expected_manifest_sha256 is not None:
        expected = expected_manifest_sha256.lower()
        if not _SHA256_RE.fullmatch(expected):
            raise ValueError("expected_manifest_sha256 must be 64 lowercase/uppercase hex characters")
        actual = file_sha256(manifest_path)
        if actual != expected:
            raise ValueError(
                f"Release manifest SHA-256 mismatch: expected {expected}, got {actual}"
            )

    manifest = _read_json_object(manifest_path, "release manifest")
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported release manifest schema: {manifest.get('schema_version')!r}")
    if manifest.get("bundle_type") != "murmur_pretrain_assets":
        raise ValueError("Release manifest has the wrong bundle_type")
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("Release manifest assets must be a non-empty list")

    expected_names: set[str] = set()
    install_destinations: set[tuple[str, str]] = set()
    model_count = 0
    cache_manifest_count = 0
    for index, asset in enumerate(raw_assets):
        if not isinstance(asset, dict):
            raise ValueError(f"Release asset {index} must be an object")
        name = _safe_basename(asset.get("name"), f"assets[{index}].name")
        install_name = _safe_basename(
            asset.get("install_name"), f"assets[{index}].install_name"
        )
        group = asset.get("install_group")
        if group not in INSTALL_GROUPS:
            raise ValueError(f"assets[{index}] has invalid install_group: {group!r}")
        destination = (str(group), install_name)
        if name in expected_names or destination in install_destinations:
            raise ValueError(f"Duplicate release asset or install destination: {name}")
        expected_names.add(name)
        install_destinations.add(destination)
        size = asset.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"assets[{index}].size_bytes must be a non-negative integer")
        if size >= MAX_GITHUB_ASSET_BYTES:
            raise ValueError(f"assets[{index}] violates the 2 GiB GitHub asset limit")
        digest = str(asset.get("sha256", "")).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"assets[{index}].sha256 is invalid")
        purpose = asset.get("purpose")
        model_count += int(purpose == "sentencepiece_model")
        cache_manifest_count += int(purpose == "token_cache_manifest")
    if model_count != 1 or cache_manifest_count != 1:
        raise ValueError(
            "Release must contain exactly one SentencePiece model and one token-cache manifest"
        )
    declared_total = manifest.get("total_size_bytes")
    computed_total = sum(int(asset["size_bytes"]) for asset in raw_assets)
    if declared_total != computed_total:
        raise ValueError(
            "Release manifest total_size_bytes does not equal the sum of its assets"
        )

    checksums = parse_sha256sums(checksums_path)
    if set(checksums) != expected_names:
        raise ValueError("SHA256SUMS file set does not exactly match release_manifest.json")

    actual_entries = list(root.iterdir())
    non_files = sorted(entry.name for entry in actual_entries if not entry.is_file() or entry.is_symlink())
    if non_files:
        raise ValueError(f"Downloaded release contains non-regular files: {non_files}")
    actual_names = {entry.name for entry in actual_entries}
    allowed_names = expected_names | {RELEASE_MANIFEST_NAME, CHECKSUMS_NAME}
    if actual_names != allowed_names:
        raise ValueError(
            "Downloaded release file set differs from its manifest: "
            f"missing={sorted(allowed_names - actual_names)}, "
            f"unexpected={sorted(actual_names - allowed_names)}"
        )

    for asset in raw_assets:
        name = str(asset["name"])
        path = root / name
        actual_size = path.stat().st_size
        if actual_size != int(asset["size_bytes"]):
            raise ValueError(
                f"Release asset size mismatch for {name}: "
                f"expected {asset['size_bytes']}, got {actual_size}"
            )
        actual_digest = file_sha256(path)
        manifest_digest = str(asset["sha256"]).lower()
        if checksums[name] != manifest_digest or actual_digest != manifest_digest:
            raise ValueError(f"Release asset SHA-256 mismatch for {name}")
    return manifest


def validate_install_layout(
    manifest: dict[str, Any],
    tokenizer_dir: str | Path,
    cache_dir: str | Path,
) -> None:
    """Validate cache completeness and tokenizer/cache pairing before publish."""

    assets = manifest.get("assets", [])
    tokenizer_assets = [
        asset for asset in assets if asset.get("install_group") == "tokenizer"
    ]
    cache_assets = [
        asset for asset in assets if asset.get("install_group") == "token_cache"
    ]
    model_assets = [
        asset for asset in tokenizer_assets if asset.get("purpose") == "sentencepiece_model"
    ]
    if len(model_assets) != 1:
        raise ValueError("Install layout must contain exactly one SentencePiece model")

    cache_sources, cache_manifest = collect_cache_sources(cache_dir)
    expected_cache_names = {source.install_name for source in cache_sources}
    declared_cache_names = {str(asset.get("install_name")) for asset in cache_assets}
    if expected_cache_names != declared_cache_names:
        raise ValueError(
            "Installed cache artifacts do not exactly match the cache manifest and release layout"
        )

    model_name = _safe_basename(model_assets[0].get("install_name"), "model install_name")
    model_digest = file_sha256(Path(tokenizer_dir) / model_name)
    if cache_manifest.get("tokenizer_sha256") != model_digest:
        raise ValueError(
            "Installed SentencePiece model does not match token-cache tokenizer_sha256"
        )


def download_github_release(
    repo: str,
    tag: str,
    destination: str | Path,
    *,
    gh_executable: str = "gh",
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    if not repo.strip() or not tag.strip():
        raise ValueError("Both GitHub repository and release tag are required")
    runner(
        [
            gh_executable,
            "release",
            "download",
            tag,
            "--repo",
            repo,
            "--dir",
            str(Path(destination)),
        ],
        check=True,
    )


def _ensure_install_targets_absent(targets: Sequence[Path]) -> None:
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise FileExistsError(
            "Refusing to mix release assets with existing tokenizer/cache paths: "
            f"{existing}. Move or remove them explicitly before installation."
        )


def install_github_release_bundle(
    repo: str,
    tag: str,
    data_root: str | Path = "data/full",
    *,
    tokenizer_dir_name: str = "tokenizer",
    cache_dir_name: str = "token_cache_sp_unigram_24k_2048",
    expected_manifest_sha256: str | None = None,
    gh_executable: str = "gh",
    runner: Callable[..., Any] = subprocess.run,
) -> tuple[Path, Path]:
    """Download, verify and atomically publish both training asset directories.

    Each directory is published with one rename. Both destinations are checked
    before download, and the first rename is rolled back if the second fails.
    """

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    tokenizer_target = root / _safe_basename(tokenizer_dir_name, "tokenizer_dir_name")
    cache_target = root / _safe_basename(cache_dir_name, "cache_dir_name")
    _ensure_install_targets_absent((tokenizer_target, cache_target))

    staging = Path(tempfile.mkdtemp(prefix=".pretrain-release-", dir=root))
    download_dir = staging / "download"
    incoming_tokenizer = staging / "incoming-tokenizer"
    incoming_cache = staging / "incoming-cache"
    download_dir.mkdir()
    incoming_tokenizer.mkdir()
    incoming_cache.mkdir()
    tokenizer_published = False
    try:
        download_github_release(
            repo,
            tag,
            download_dir,
            gh_executable=gh_executable,
            runner=runner,
        )
        manifest = validate_downloaded_bundle(
            download_dir,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        _ensure_install_targets_absent((tokenizer_target, cache_target))
        group_dirs = {
            "tokenizer": incoming_tokenizer,
            "token_cache": incoming_cache,
        }
        for asset in manifest["assets"]:
            source = download_dir / str(asset["name"])
            destination = group_dirs[str(asset["install_group"])] / str(asset["install_name"])
            source.replace(destination)

        validate_install_layout(manifest, incoming_tokenizer, incoming_cache)

        # Preserve the validated release provenance with the cache. These files
        # are copied only after every listed asset has passed its size/hash check.
        shutil.copyfile(download_dir / RELEASE_MANIFEST_NAME, incoming_cache / RELEASE_MANIFEST_NAME)
        shutil.copyfile(download_dir / CHECKSUMS_NAME, incoming_cache / CHECKSUMS_NAME)

        incoming_tokenizer.replace(tokenizer_target)
        tokenizer_published = True
        try:
            incoming_cache.replace(cache_target)
        except Exception:
            if tokenizer_target.exists() and not incoming_tokenizer.exists():
                tokenizer_target.replace(incoming_tokenizer)
                tokenizer_published = False
            raise
        return tokenizer_target, cache_target
    finally:
        # If an exceptional platform failure prevented rollback, leave the
        # published tokenizer visible instead of deleting user-visible data.
        if not tokenizer_published or cache_target.exists():
            shutil.rmtree(staging, ignore_errors=True)
