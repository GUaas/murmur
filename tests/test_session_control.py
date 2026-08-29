from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from muddywater.config_validation import validate_pretrain_config
from muddywater.session_control import (
    SESSION_DEADLINE_ENV,
    TORCHINDUCTOR_AUTOGRAD_CACHE_ENV,
    TORCHINDUCTOR_CACHE_DIR_ENV,
    TORCHINDUCTOR_FX_GRAPH_CACHE_ENV,
    TrainingSessionBudget,
    build_session_deadline,
    configure_persistent_compile_cache,
    session_deadline_from_environment,
)
from muddywater.training_report import build_training_summary


def minimal_strict_config() -> dict:
    return {
        "strict_config": True,
        "seed": 1,
        "device": "cpu",
        "tokenizer": {
            "path": "tokenizer.model",
            "vocab_size": 128,
            "sample_size": 1,
            "diagnostic_size": 1,
            "num_threads": 1,
        },
        "model": {},
        "data": {
            "train_paths": ["train.jsonl"],
            "jsonl_format": "text",
            "min_chars": 1,
            "validation_split": 0.0,
            "token_cache_dir": "cache",
            "train_token_files": ["train.bin"],
            "stride": 8,
        },
        "training": {
            "output_dir": "outputs/test",
            "batch_size": 1,
            "grad_accum_steps": 1,
            "max_epochs": 1,
            "max_steps": 1,
            "learning_rate": 1e-3,
            "num_workers": 0,
            "eval_num_workers": 0,
            "log_interval": 1,
            "eval_interval": 1,
            "save_interval": 1,
            "session_max_seconds": 10,
        },
    }


class TrainingSessionBudgetTests(unittest.TestCase):
    def test_persistent_compile_cache_defaults_under_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            environment = configure_persistent_compile_cache(
                project_root=temp_dir,
                environ={"KEEP_ME": "yes"},
            )
            expected_cache_dir = Path(temp_dir).resolve() / ".torch_compile_cache"
            self.assertEqual(environment["KEEP_ME"], "yes")
            self.assertEqual(
                environment[TORCHINDUCTOR_CACHE_DIR_ENV],
                str(expected_cache_dir),
            )
            self.assertEqual(environment[TORCHINDUCTOR_FX_GRAPH_CACHE_ENV], "1")
            self.assertEqual(environment[TORCHINDUCTOR_AUTOGRAD_CACHE_ENV], "1")
            self.assertTrue(expected_cache_dir.is_dir())

    def test_persistent_compile_cache_preserves_explicit_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_cache_dir = Path(temp_dir) / "custom-cache"
            environment = configure_persistent_compile_cache(
                project_root=Path(temp_dir) / "project",
                environ={
                    TORCHINDUCTOR_CACHE_DIR_ENV: str(custom_cache_dir),
                    TORCHINDUCTOR_FX_GRAPH_CACHE_ENV: "0",
                    TORCHINDUCTOR_AUTOGRAD_CACHE_ENV: "0",
                },
            )
            self.assertEqual(
                environment[TORCHINDUCTOR_CACHE_DIR_ENV],
                str(custom_cache_dir),
            )
            self.assertEqual(environment[TORCHINDUCTOR_FX_GRAPH_CACHE_ENV], "0")
            self.assertEqual(environment[TORCHINDUCTOR_AUTOGRAD_CACHE_ENV], "0")
            self.assertTrue(custom_cache_dir.is_dir())

    def test_budget_pauses_at_limit(self) -> None:
        budget = TrainingSessionBudget(started_at=100.0, max_seconds=10.0)
        self.assertFalse(budget.should_pause(109.999))
        self.assertTrue(budget.should_pause(110.0))

    def test_budget_can_be_disabled(self) -> None:
        budget = TrainingSessionBudget.from_training_config({}, started_at=100.0)
        self.assertFalse(budget.enabled)
        self.assertFalse(budget.should_pause(1_000.0))

    def test_launcher_deadline_includes_pretrainer_initialization(self) -> None:
        budget = TrainingSessionBudget.from_training_config(
            {"session_max_seconds": 10.0},
            started_at=105.0,
            launcher_deadline_at=110.0,
        )
        self.assertEqual(budget.deadline_at, 110.0)
        self.assertEqual(budget.remaining_seconds(108.0), 2.0)
        self.assertTrue(budget.should_pause(110.0))

    def test_trainer_deadline_cannot_extend_launcher_deadline(self) -> None:
        budget = TrainingSessionBudget.from_training_config(
            {"session_max_seconds": 10.0},
            started_at=100.0,
            launcher_deadline_at=200.0,
        )
        self.assertEqual(budget.deadline_at, 110.0)

    def test_session_deadline_environment_round_trip(self) -> None:
        deadline = build_session_deadline(started_at=100.0, max_seconds=10.0)
        parsed = session_deadline_from_environment(
            {SESSION_DEADLINE_ENV: str(deadline)}
        )
        self.assertEqual(parsed, 110.0)

    def test_invalid_session_deadline_environment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, SESSION_DEADLINE_ENV):
            session_deadline_from_environment({SESSION_DEADLINE_ENV: "nan"})

    def test_strict_config_accepts_positive_session_limit(self) -> None:
        validate_pretrain_config(minimal_strict_config())

    def test_strict_config_rejects_non_positive_session_limit(self) -> None:
        config = minimal_strict_config()
        config["training"]["session_max_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "session_max_seconds"):
            validate_pretrain_config(config)

    def test_training_summary_accepts_paused_status(self) -> None:
        model = torch.nn.Linear(2, 2)
        summary = build_training_summary(
            model=model,
            config={"model": {}},
            elapsed_seconds=1.0,
            global_step=1,
            seen_train_tokens=8,
            best_val_loss=float("inf"),
            final_metrics=None,
            device=torch.device("cpu"),
            precision_name="fp32",
            world_size=1,
            total_steps=10,
            warmup_steps=0,
            tokens_per_step_estimate=8,
            status="paused",
            termination_reason="session_time_limit_reached",
            target_reached=False,
        )
        self.assertEqual(summary["status"], "paused")


if __name__ == "__main__":
    unittest.main()
