from __future__ import annotations

import logging
import math
import os
import random
import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch


TRAINING_ARTIFACT_PATTERNS = (
    "train.log",
    "train.rank*.log",
    "config*.yaml",
    "training_summary*.json",
    "run_manifest*.json",
    "diagnostics*.json",
    "*.pt",
)

RESUME_METADATA_ARTIFACTS = (
    "run_manifest.json",
    "diagnostics.json",
)


@dataclass(frozen=True)
class TrainingOutputPreparation:
    """Result of preparing one output directory for a fresh or resumed run."""

    output_dir: Path
    mode: Literal["fresh", "resume"]
    resume_from: Path | None = None
    archived_artifacts: tuple[Path, ...] = ()


def find_training_artifacts(output_dir: str | Path) -> tuple[Path, ...]:
    """Return known training artifacts already present in ``output_dir``."""

    directory = Path(output_dir)
    if not directory.exists():
        return ()
    artifacts: set[Path] = set()
    for pattern in TRAINING_ARTIFACT_PATTERNS:
        artifacts.update(path for path in directory.glob(pattern) if path.is_file())
    return tuple(sorted(artifacts, key=lambda path: path.name))


def archive_artifact(path: str | Path, label: str = "before-resume") -> Path | None:
    """Move an artifact to a unique sibling path and return the archive path."""

    source = Path(path)
    if not source.exists():
        return None
    safe_label = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in label)
    suffix = "".join(source.suffixes)
    stem = source.name[: -len(suffix)] if suffix else source.name
    candidate = source.with_name(f"{stem}.{safe_label}{suffix}")
    index = 1
    while candidate.exists():
        candidate = source.with_name(f"{stem}.{safe_label}.{index}{suffix}")
        index += 1
    source.replace(candidate)
    return candidate


def prepare_training_output_dir(
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
) -> TrainingOutputPreparation:
    """Safely prepare an output directory before any run artifact is written.

    Fresh runs reject directories containing known training artifacts. Resumed
    runs require an explicit checkpoint but do not mutate existing metadata;
    call :func:`archive_resume_metadata` only after checkpoint validation.
    """

    directory = Path(output_dir)
    resume_value = str(resume_from).strip() if resume_from is not None else ""
    if not resume_value:
        existing = find_training_artifacts(directory)
        if existing:
            names = ", ".join(path.name for path in existing[:8])
            if len(existing) > 8:
                names += f", ... (+{len(existing) - 8} more)"
            raise FileExistsError(
                f"Refusing to start a fresh training run in {directory}: "
                f"existing training artifacts found ({names}). Choose a new "
                "training.output_dir or set training.resume_from explicitly."
            )
        directory.mkdir(parents=True, exist_ok=True)
        return TrainingOutputPreparation(output_dir=directory, mode="fresh")

    checkpoint = Path(resume_value).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
    directory.mkdir(parents=True, exist_ok=True)
    return TrainingOutputPreparation(
        output_dir=directory,
        mode="resume",
        resume_from=checkpoint,
    )


def archive_resume_metadata(
    preparation: TrainingOutputPreparation,
) -> TrainingOutputPreparation:
    """Archive lightweight canonical metadata after a resume is validated."""

    if preparation.mode != "resume":
        return preparation
    archived: list[Path] = []
    for name in RESUME_METADATA_ARTIFACTS:
        archive_path = archive_artifact(preparation.output_dir / name)
        if archive_path is not None:
            archived.append(archive_path)
    return TrainingOutputPreparation(
        output_dir=preparation.output_dir,
        mode="resume",
        resume_from=preparation.resume_from,
        archived_artifacts=tuple(archived),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = "auto") -> torch.device:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def ensure_dir(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def atomic_write_text(
    path: str | Path,
    content: str,
    *,
    encoding: str = "utf-8",
    overwrite: bool = True,
) -> Path:
    """Atomically publish a text artifact without exposing partial contents."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and output_path.exists():
        raise FileExistsError(output_path)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and output_path.exists():
            raise FileExistsError(output_path)
        temp_path.replace(output_path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return output_path


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logging(
    log_file: str | Path | None = None,
    *,
    file_mode: Literal["a", "w", "x"] = "a",
    enable_stream: bool = True,
) -> logging.Logger:
    logger = logging.getLogger("muddywater")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if enable_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode=file_mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def perplexity_from_loss(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    if loss >= 100:
        return float("inf")
    return math.exp(loss)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def project_root_from_file(file_path: str | Path, levels: int = 1) -> Path:
    path = Path(file_path).resolve()
    for _ in range(levels):
        path = path.parent
    return path


def enable_torch_backends() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def now_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def get_num_workers(default: int = 0) -> int:
    if os.name == "nt":
        return 0
    return default
