from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentBoundarySettings:
    policy: str
    document_attention: bool
    ignore_cross_document_targets: bool
    single_document_windows: bool = False

    @property
    def strategy(self) -> str:
        """Human-facing policy name for diagnostics and logs."""
        return {
            "none": "none",
            "label_only": "label_only_fast",
            "strict": "strict_dense",
            "strict_varlen": "strict_varlen",
            "single_doc_windows": "single_doc_windows",
        }[self.policy]


def _normalize_legacy_document_attention(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {
        "1",
        "true",
        "yes",
        "on",
        "strict",
        "strict_dense",
        "strict_varlen",
        "varlen",
        "mask",
    }:
        return True
    if normalized in {
        "",
        "0",
        "false",
        "no",
        "off",
        "none",
        "label_only",
        "label-only",
        "label_only_fast",
        "single_doc_windows",
    }:
        return False
    raise ValueError(
        "data.document_attention must be a boolean, strict/mask, strict_dense, "
        "label_only_fast, single_doc_windows, or none"
    )


def _normalize_boundary_policy(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "": "label_only",
        "target_only": "label_only",
        "targets_only": "label_only",
        "label": "label_only",
        "labels": "label_only",
        "label_only_fast": "label_only",
        "fast": "label_only",
        "causal": "strict",
        "mask": "strict",
        "masked": "strict",
        "document_attention": "strict",
        "attention": "strict",
        "strict_dense": "strict",
        "strict_varlen": "strict_varlen",
        "varlen": "strict_varlen",
        "flash_varlen": "strict_varlen",
        "cu_seqlens": "strict_varlen",
        "packed_varlen": "strict_varlen",
        "dense": "strict",
        "single_doc": "single_doc_windows",
        "single_document": "single_doc_windows",
        "single_document_windows": "single_doc_windows",
        "single-doc-windows": "single_doc_windows",
        "off": "none",
        "false": "none",
        "0": "none",
    }
    policy = aliases.get(normalized, normalized)
    if policy not in {"none", "label_only", "strict", "strict_varlen", "single_doc_windows"}:
        raise ValueError(
            "data.document_boundary_policy must be one of: none, label_only_fast, "
            "strict_dense, strict_varlen, single_doc_windows"
        )
    return policy


def resolve_document_boundary_settings(data_config: dict[str, Any]) -> DocumentBoundarySettings:
    """Resolve packed-document boundary behavior from modern and legacy config keys."""

    if "document_boundary_policy" in data_config and data_config.get("document_boundary_policy") is not None:
        policy = _normalize_boundary_policy(data_config.get("document_boundary_policy"))
        if "document_attention" in data_config and data_config.get("document_attention") is not None:
            legacy_document_attention = _normalize_legacy_document_attention(
                data_config.get("document_attention")
            )
            policy_document_attention = policy in {"strict", "strict_varlen"}
            if legacy_document_attention != policy_document_attention:
                raise ValueError(
                    "data.document_boundary_policy and data.document_attention disagree. "
                    "Use document_boundary_policy: strict_dense or strict_varlen to enable "
                    "document attention, or remove the legacy document_attention key."
                )
        document_attention = policy in {"strict", "strict_varlen"}
        default_ignore = policy in {"label_only", "strict", "strict_varlen", "single_doc_windows"}
        ignore_cross_document_targets = bool(
            data_config.get("ignore_cross_document_targets", default_ignore)
        )
        return DocumentBoundarySettings(
            policy=policy,
            document_attention=document_attention,
            ignore_cross_document_targets=ignore_cross_document_targets,
            single_document_windows=policy == "single_doc_windows",
        )

    document_attention = _normalize_legacy_document_attention(
        data_config.get("document_attention", True)
    )
    ignore_cross_document_targets = bool(data_config.get("ignore_cross_document_targets", True))
    if document_attention:
        policy = "strict"
    elif ignore_cross_document_targets:
        policy = "label_only"
    else:
        policy = "none"
    return DocumentBoundarySettings(
        policy=policy,
        document_attention=document_attention,
        ignore_cross_document_targets=ignore_cross_document_targets,
        single_document_windows=False,
    )


def resolve_document_attention(data_config: dict[str, Any]) -> bool:
    return resolve_document_boundary_settings(data_config).document_attention
