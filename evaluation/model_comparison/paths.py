from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationPaths:
    """All filesystem locations used by the comparison suite."""

    workspace: Path
    v1_dir: Path
    v2_dir: Path
    v2_source: Path
    v2_config: Path
    validation_file: Path
    output_dir: Path

    @classmethod
    def discover(cls, output_dir: str | Path = "model_comparison_results") -> "EvaluationPaths":
        workspace = Path(__file__).resolve().parents[1]
        v1_dir = workspace / "simpl_best_model" / "simpl_best_model"
        v2_dir = workspace / "murmur-203m-text-simplification-project-best-only-20260814"
        paths = cls(
            workspace=workspace,
            v1_dir=v1_dir,
            v2_dir=v2_dir,
            v2_source=v2_dir / "source",
            v2_config=v2_dir / "configs" / "inference_text_simplification_portable.yaml",
            validation_file=(
                v2_dir
                / "data"
                / "text_simplification_pass_filtered"
                / "processed"
                / "validation.jsonl"
            ),
            output_dir=(workspace / output_dir).resolve()
            if not Path(output_dir).is_absolute()
            else Path(output_dir).resolve(),
        )
        paths.validate()
        return paths

    def validate(self) -> None:
        required = (
            self.v1_dir / "model.safetensors",
            self.v1_dir / "config.json",
            self.v1_dir / "standalone_inference.py",
            self.v2_config,
            self.validation_file,
            self.v2_dir
            / "outputs"
            / "sft_203m_text_simplification_pass_filtered"
            / "murmur_203m_text_simplification_best_weights_only.pt",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing evaluation inputs:\n" + "\n".join(missing))

    @property
    def raw_dir(self) -> Path:
        return self.output_dir / "raw"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"
