from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .utils import atomic_write_text


@dataclass(frozen=True)
class DownloadedFile:
    repo_path: str
    local_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DownloadRetryPolicy:
    """Bound Hugging Face metadata requests and retry transient failures."""

    max_attempts: int = 5
    etag_timeout_seconds: float = 30.0
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 30.0

    def validate(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.etag_timeout_seconds <= 0:
            raise ValueError("etag_timeout_seconds must be positive")
        if self.initial_backoff_seconds < 0:
            raise ValueError("initial_backoff_seconds cannot be negative")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds cannot be smaller than initial_backoff_seconds"
            )


def resolve_dataset_revision(repo_id: str, revision: str | None = None) -> str:
    """Resolve a mutable dataset reference to an immutable commit SHA."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise RuntimeError(
            "huggingface_hub is required for dataset downloads. "
            "Install requirements-data.txt."
        ) from exc
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a commit SHA for {repo_id!r}.")
    return str(info.sha)


def list_parquet_files(
    repo_id: str,
    subset: str,
    revision: str,
) -> list[str]:
    """List sorted Parquet files under one dataset subset directory."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise RuntimeError(
            "huggingface_hub is required for dataset downloads. "
            "Install requirements-data.txt."
        ) from exc
    normalized_subset = subset.strip("/")
    prefix = f"{normalized_subset}/"
    info = HfApi().dataset_info(repo_id=repo_id, revision=revision, files_metadata=False)
    paths = sorted(
        sibling.rfilename
        for sibling in info.siblings or []
        if sibling.rfilename.startswith(prefix) and sibling.rfilename.endswith(".parquet")
    )
    if not paths:
        raise RuntimeError(
            f"No Parquet files found for dataset={repo_id!r}, subset={subset!r}, "
            f"revision={revision!r}."
        )
    return paths


def select_evenly_spaced(paths: Iterable[str], count: int) -> list[str]:
    """Select one centered item from each equal-width region of a sorted list."""

    ordered = sorted(dict.fromkeys(str(path) for path in paths))
    requested = int(count)
    if requested <= 0:
        raise ValueError("count must be positive")
    if requested > len(ordered):
        raise ValueError(f"Requested {requested} files, but only {len(ordered)} are available.")
    if requested == len(ordered):
        return ordered

    selected_indices = [
        min(len(ordered) - 1, int((index + 0.5) * len(ordered) / requested))
        for index in range(requested)
    ]
    if len(set(selected_indices)) != requested:
        raise AssertionError("Even selection unexpectedly produced duplicate indices.")
    return [ordered[index] for index in selected_indices]


def select_evenly_spaced_extension(
    paths: Iterable[str],
    existing_paths: Iterable[str],
    target_total_count: int,
) -> list[str]:
    """Select a balanced, non-overlapping supplement for an existing selection."""

    ordered = sorted(dict.fromkeys(str(path) for path in paths))
    available = set(ordered)
    existing = sorted(dict.fromkeys(str(path) for path in existing_paths))
    unknown = [path for path in existing if path not in available]
    if unknown:
        raise ValueError(
            f"{len(unknown)} existing paths are absent from the upstream file list; "
            f"first={unknown[:3]}"
        )

    target = int(target_total_count)
    if target < len(existing):
        raise ValueError(
            f"target_total_count={target} is smaller than the existing selection "
            f"of {len(existing)} files."
        )
    if target > len(ordered):
        raise ValueError(
            f"target_total_count={target} exceeds the {len(ordered)} available files."
        )

    supplement_count = target - len(existing)
    if supplement_count == 0:
        return []
    existing_set = set(existing)
    remaining = [path for path in ordered if path not in existing_set]
    return select_evenly_spaced(remaining, supplement_count)


def file_sha256(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_download_manifest(
    *,
    repo_id: str,
    revision: str,
    subset: str,
    all_paths: list[str],
    selected_paths: list[str],
    upstream_subset_tokens: int | None,
    selection_strategy: str = "centered_even_spacing_over_sorted_subset_files",
    selection_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_fraction = len(selected_paths) / len(all_paths)
    estimated_tokens = (
        round(int(upstream_subset_tokens) * selected_fraction)
        if upstream_subset_tokens is not None
        else None
    )
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "planned",
        "dataset": {
            "repo_id": repo_id,
            "revision": revision,
            "subset": subset,
        },
        "selection": {
            "strategy": selection_strategy,
            "available_files": len(all_paths),
            "selected_files": len(selected_paths),
            "selected_fraction": round(selected_fraction, 10),
            "upstream_subset_tokens": upstream_subset_tokens,
            "estimated_selected_tokens_before_local_cleaning": estimated_tokens,
        },
        "selected_repo_paths": selected_paths,
        "completed_files": [],
        "completed_count": 0,
        "completed_bytes": 0,
        "failures": [],
    }
    if selection_metadata:
        manifest["selection"].update(selection_metadata)
    return manifest


def _validate_existing_manifest(
    manifest: dict[str, Any],
    *,
    repo_id: str,
    revision: str,
    subset: str,
    selected_paths: list[str],
) -> None:
    dataset = manifest.get("dataset", {})
    expected = {
        "repo_id": repo_id,
        "revision": revision,
        "subset": subset,
    }
    if dataset != expected:
        raise ValueError(
            "Existing download manifest targets a different dataset revision or subset: "
            f"existing={dataset}, requested={expected}"
        )
    if manifest.get("selected_repo_paths") != selected_paths:
        raise ValueError(
            "Existing download manifest has a different selected file set. "
            "Use another target directory for a new selection."
        )


def _download_one(
    *,
    repo_id: str,
    revision: str,
    repo_path: str,
    raw_dir: Path,
    retry_policy: DownloadRetryPolicy,
) -> DownloadedFile:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional data environment
        raise RuntimeError(
            "huggingface_hub is required for dataset downloads. "
            "Install requirements-data.txt."
        ) from exc
    retry_policy.validate()
    local_path: Path | None = None
    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                    revision=revision,
                    local_dir=raw_dir,
                    etag_timeout=retry_policy.etag_timeout_seconds,
                )
            ).resolve()
            break
        except Exception:
            if attempt >= retry_policy.max_attempts:
                raise
            delay = min(
                retry_policy.initial_backoff_seconds * (2 ** (attempt - 1)),
                retry_policy.max_backoff_seconds,
            )
            print(
                f"retrying={repo_path} attempt={attempt + 1}/"
                f"{retry_policy.max_attempts} backoff_s={delay:.1f}",
                flush=True,
            )
            time.sleep(delay)
    if local_path is None:  # pragma: no cover - defensive invariant
        raise RuntimeError(f"Download did not return a local path: {repo_path}")
    return DownloadedFile(
        repo_path=repo_path,
        local_path=str(local_path),
        size_bytes=local_path.stat().st_size,
        sha256=file_sha256(local_path),
    )


def _completed_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["repo_path"]): dict(item)
        for item in manifest.get("completed_files", [])
        if isinstance(item, dict) and item.get("repo_path")
    }


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic_write_text(path, json.dumps(manifest, ensure_ascii=False, indent=2))


def download_dataset_selection(
    *,
    repo_id: str,
    revision: str,
    subset: str,
    selected_paths: list[str],
    all_paths: list[str],
    target_root: str | Path,
    workers: int = 6,
    checkpoint_interval: int = 10,
    upstream_subset_tokens: int | None = None,
    manifest_name: str = "download_manifest.json",
    raw_dir_name: str = "raw",
    selection_strategy: str = "centered_even_spacing_over_sorted_subset_files",
    selection_metadata: dict[str, Any] | None = None,
    retry_policy: DownloadRetryPolicy | None = None,
) -> dict[str, Any]:
    """Download and hash a fixed dataset selection with a resumable manifest."""

    root = Path(target_root).resolve()
    if Path(manifest_name).name != manifest_name:
        raise ValueError("manifest_name must be a file name, not a path")
    if Path(raw_dir_name).name != raw_dir_name:
        raise ValueError("raw_dir_name must be a directory name, not a path")
    raw_dir = root / raw_dir_name
    manifest_path = root / manifest_name
    root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_existing_manifest(
            manifest,
            repo_id=repo_id,
            revision=revision,
            subset=subset,
            selected_paths=selected_paths,
        )
    else:
        manifest = build_download_manifest(
            repo_id=repo_id,
            revision=revision,
            subset=subset,
            all_paths=all_paths,
            selected_paths=selected_paths,
            upstream_subset_tokens=upstream_subset_tokens,
            selection_strategy=selection_strategy,
            selection_metadata=selection_metadata,
        )
        _write_manifest(manifest_path, manifest)

    completed = _completed_index(manifest)
    pending_paths = []
    for repo_path in selected_paths:
        record = completed.get(repo_path)
        if record is None:
            pending_paths.append(repo_path)
            continue
        local_path = Path(str(record.get("local_path", "")))
        if not local_path.is_file() or local_path.stat().st_size != int(record["size_bytes"]):
            completed.pop(repo_path, None)
            pending_paths.append(repo_path)

    manifest["status"] = "downloading" if pending_paths else "completed"
    manifest["failures"] = []
    manifest["completed_files"] = [completed[path] for path in selected_paths if path in completed]
    manifest["completed_count"] = len(completed)
    manifest["completed_bytes"] = sum(int(item["size_bytes"]) for item in completed.values())
    _write_manifest(manifest_path, manifest)
    if not pending_paths:
        return manifest

    lock = threading.Lock()
    completed_since_checkpoint = 0
    failures: list[dict[str, str]] = []
    started = time.monotonic()
    effective_retry_policy = retry_policy or DownloadRetryPolicy()
    effective_retry_policy.validate()

    def submit(
        executor: ThreadPoolExecutor,
        repo_path: str,
    ) -> Future[DownloadedFile]:
        return executor.submit(
            _download_one,
            repo_id=repo_id,
            revision=revision,
            repo_path=repo_path,
            raw_dir=raw_dir,
            retry_policy=effective_retry_policy,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {submit(executor, path): path for path in pending_paths}
        for future in as_completed(futures):
            repo_path = futures[future]
            try:
                downloaded = future.result()
            except Exception as exc:  # pragma: no cover - network failures are environment-specific
                failures.append(
                    {
                        "repo_path": repo_path,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(
                    f"download_failed path={repo_path} error={type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            with lock:
                completed[repo_path] = asdict(downloaded)
                completed_since_checkpoint += 1
                done = len(completed)
                elapsed = max(1e-6, time.monotonic() - started)
                session_bytes = sum(
                    int(completed[path]["size_bytes"])
                    for path in pending_paths
                    if path in completed
                )
                print(
                    f"downloaded={done}/{len(selected_paths)} "
                    f"session_GiB={session_bytes / 2**30:.3f} "
                    f"MiB_per_s={session_bytes / 2**20 / elapsed:.2f} "
                    f"path={repo_path}",
                    flush=True,
                )
                if completed_since_checkpoint >= max(1, int(checkpoint_interval)):
                    manifest["completed_files"] = [
                        completed[path] for path in selected_paths if path in completed
                    ]
                    manifest["completed_count"] = len(completed)
                    manifest["completed_bytes"] = sum(
                        int(item["size_bytes"]) for item in completed.values()
                    )
                    _write_manifest(manifest_path, manifest)
                    completed_since_checkpoint = 0

    manifest["completed_files"] = [
        completed[path] for path in selected_paths if path in completed
    ]
    manifest["completed_count"] = len(completed)
    manifest["completed_bytes"] = sum(int(item["size_bytes"]) for item in completed.values())
    manifest["failures"] = failures
    manifest["status"] = (
        "completed"
        if len(completed) == len(selected_paths) and not failures
        else "incomplete"
    )
    _write_manifest(manifest_path, manifest)
    if manifest["status"] != "completed":
        raise RuntimeError(
            f"Dataset download incomplete: completed={len(completed)}/"
            f"{len(selected_paths)}, failures={len(failures)}. Re-run to resume."
        )
    return manifest


def combine_completed_download_manifests(
    manifest_paths: Iterable[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Combine completed, non-overlapping selections from one pinned dataset."""

    paths = [Path(path).resolve() for path in manifest_paths]
    if not paths:
        raise ValueError("At least one manifest is required")

    manifests = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    dataset = manifests[0].get("dataset", {})
    selected: dict[str, dict[str, Any]] = {}
    components: list[dict[str, Any]] = []
    available_files = 0
    upstream_subset_tokens: int | None = None
    estimated_tokens = 0
    has_token_estimate = True

    for path, manifest in zip(paths, manifests):
        if manifest.get("status") != "completed":
            raise ValueError(
                f"Cannot combine incomplete manifest: path={path}, "
                f"status={manifest.get('status')!r}"
            )
        if manifest.get("dataset", {}) != dataset:
            raise ValueError(
                f"Manifest targets a different dataset: path={path}, "
                f"dataset={manifest.get('dataset', {})}, expected={dataset}"
            )

        records = {
            str(item["repo_path"]): dict(item)
            for item in manifest.get("completed_files", [])
            if isinstance(item, dict) and item.get("repo_path")
        }
        repo_paths = [str(item) for item in manifest.get("selected_repo_paths", [])]
        missing_records = [repo_path for repo_path in repo_paths if repo_path not in records]
        if missing_records:
            raise ValueError(
                f"Manifest has {len(missing_records)} selected paths without completed "
                f"records: path={path}, first={missing_records[:3]}"
            )
        overlaps = [repo_path for repo_path in repo_paths if repo_path in selected]
        if overlaps:
            raise ValueError(
                f"Download manifests overlap on {len(overlaps)} files; "
                f"first={overlaps[:3]}"
            )
        selected.update({repo_path: records[repo_path] for repo_path in repo_paths})

        selection = manifest.get("selection", {})
        available_files = max(available_files, int(selection.get("available_files", 0)))
        component_upstream_tokens = selection.get("upstream_subset_tokens")
        if component_upstream_tokens is not None:
            component_upstream_tokens = int(component_upstream_tokens)
            if (
                upstream_subset_tokens is not None
                and upstream_subset_tokens != component_upstream_tokens
            ):
                raise ValueError("Component manifests use inconsistent upstream token totals")
            upstream_subset_tokens = component_upstream_tokens
        component_estimate = selection.get(
            "estimated_selected_tokens_before_local_cleaning"
        )
        if component_estimate is None:
            has_token_estimate = False
        else:
            estimated_tokens += int(component_estimate)
        components.append(
            {
                "manifest": path.name,
                "selected_files": len(repo_paths),
                "completed_bytes": int(manifest.get("completed_bytes", 0)),
            }
        )

    ordered_paths = sorted(selected)
    combined = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "completed",
        "dataset": dataset,
        "selection": {
            "strategy": "union_of_non_overlapping_pinned_manifests",
            "available_files": available_files,
            "selected_files": len(ordered_paths),
            "selected_fraction": (
                round(len(ordered_paths) / available_files, 10)
                if available_files
                else None
            ),
            "upstream_subset_tokens": upstream_subset_tokens,
            "estimated_selected_tokens_before_local_cleaning": (
                estimated_tokens if has_token_estimate else None
            ),
            "components": components,
        },
        "selected_repo_paths": ordered_paths,
        "completed_files": [selected[repo_path] for repo_path in ordered_paths],
        "completed_count": len(ordered_paths),
        "completed_bytes": sum(
            int(selected[repo_path]["size_bytes"]) for repo_path in ordered_paths
        ),
        "failures": [],
    }
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, json.dumps(combined, ensure_ascii=False, indent=2))
    return combined
