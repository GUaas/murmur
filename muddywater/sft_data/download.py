from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .catalog import DUCONV_URL, HF_DATASETS, ULTRADATA, HfDataset


def _download_hf_dataset(spec: HfDataset, raw_dir: Path) -> list[str]:
    local_dir = raw_dir / spec.key
    local_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for filename in spec.files:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=spec.revision,
            local_dir=local_dir,
        )
        paths.append(str(Path(path).resolve()))
    for metadata_name in ("README.md", "LICENSE", "LICENSE.md"):
        try:
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=metadata_name,
                repo_type="dataset",
                revision=spec.revision,
                local_dir=local_dir,
            )
        except Exception:
            pass
    return paths


def _download_url(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Murmur-SFT-Builder/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    os.replace(temporary, destination)
    return destination


def download_sources(root: Path, include_ultradata: bool = True) -> dict[str, Any]:
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"downloaded": {}, "skipped": {}}
    for spec in HF_DATASETS:
        report["downloaded"][spec.key] = _download_hf_dataset(spec, raw_dir)

    report["downloaded"]["duconv"] = [
        str(_download_url(DUCONV_URL, raw_dir / "duconv" / "DuConv.zip").resolve())
    ]

    if include_ultradata:
        try:
            report["downloaded"][ULTRADATA.key] = _download_hf_dataset(ULTRADATA, raw_dir)
        except Exception as exc:
            report["skipped"][ULTRADATA.key] = f"{type(exc).__name__}: {exc}"

    report_path = root / "reports" / "download_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
