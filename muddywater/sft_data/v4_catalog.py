from __future__ import annotations

from dataclasses import dataclass


INFINITY_REPO_ID = "BAAI/Infinity-Instruct"
INFINITY_REVISION = "bddc39a8feadbd679c30623197f4e736b7e75b48"


@dataclass(frozen=True, slots=True)
class InfinityShard:
    key: str
    subset: str
    filename: str
    reward_thresholds: tuple[tuple[str, float], ...]
    quotas: tuple[tuple[str, int], ...]
    allowed_sources: tuple[str, ...] = ()

    def reward_threshold(self, language: str) -> float:
        return dict(self.reward_thresholds)[language]

    def quota(self, language: str) -> int:
        return dict(self.quotas)[language]

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(language for language, _ in self.quotas)


INFINITY_7M_CORE = InfinityShard(
    key="infinity_7m_core",
    subset="7M_core",
    filename="train-00000-of-00015.parquet",
    reward_thresholds=(("zh", 7.0), ("en", 18.0)),
    quotas=(("zh", 2_500), ("en", 7_500)),
    allowed_sources=(
        "Subjective",
        "OpenHermes-2.5",
        "code_bagel",
        "CodeFeedback",
        "MathInstruct",
        "code_exercises",
        "Evol-Instruct-Code-80K",
        "CodeExercise-Python-27k",
        "self-oss-instruct-sc2-exec-filter-50k",
        "python-code-dataset-500k",
        "Glaive-code-assistant-v3",
    ),
)

INFINITY_GEN = InfinityShard(
    key="infinity_gen",
    subset="Gen",
    filename="train-00000-of-00015.parquet",
    reward_thresholds=(("zh", 10.0), ("en", 24.0)),
    quotas=(("zh", 7_000), ("en", 10_000)),
    allowed_sources=("Subjective",),
)

INFINITY_SHARDS = (INFINITY_7M_CORE, INFINITY_GEN)

