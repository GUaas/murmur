from __future__ import annotations

import math
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoint import load_checkpoint, restore_checkpoint, save_checkpoint
from .config import save_config
from .distributed import (
    DistributedContext,
    all_gather_int,
    all_gather_objects,
    all_reduce_sum,
    barrier,
    unwrap_model,
)
from .evaluation import evaluate_language_model
from .optim import (
    CosineLRScheduler,
    build_grad_scaler,
    cosine_weight_decay_at_step,
    muon_momentum_at_step,
    resolve_precision,
    resolve_warmup_steps,
    resolve_warmup_tokens,
    validate_warmup_config,
)
from .resume_validation import validate_resume_model_config, validate_resume_training_state
from .run_manifest import build_training_identity
from .sampling import set_loader_start_batch
from .session_control import TrainingSessionBudget, session_deadline_from_environment
from .training_guard import LossDescentGuard, LossGuardConfig, LossGuardDecision
from .training_report import build_training_summary, write_training_summary
from .training_state import ScheduleState, capture_rng_state, restore_rng_state
from .utils import (
    TrainingOutputPreparation,
    archive_artifact,
    count_parameters,
    format_seconds,
    perplexity_from_loss,
    prepare_training_output_dir,
    setup_logging,
)


@dataclass(frozen=True)
class TrainStepMetrics:
    total_loss: float
    ce_loss: float
    z_loss: float
    lr: float
    grad_norm: float
    tokens: int
    processed_tokens: int
    step_applied: bool
    skip_reason: str | None = None


@dataclass(frozen=True)
class OptimizerStepResult:
    grad_norm: float
    applied: bool
    skip_reason: str | None = None


class TrainingRunFailedError(RuntimeError):
    """Base error for an intentional, evidence-backed training abort."""

    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


class TrainingTargetNotReachedError(TrainingRunFailedError):
    """Raised when an explicit step/token target was not actually reached."""


class LossDescentGuardError(TrainingRunFailedError):
    """Raised after the configured loss guard exhausts its patience."""


def _grad_scaler_step_applied(scaler: Any, scale_before: float) -> bool:
    """Return whether ``GradScaler.step`` executed the optimizer update.

    Both ``torch.amp.GradScaler`` and the legacy ``torch.cuda.amp.GradScaler``
    reduce their public loss scale during ``update()`` when non-finite gradients
    caused ``step()`` to skip the optimizer. Optimizer return values cannot be
    used here because most PyTorch optimizers return ``None`` on a real step.
    """

    return float(scaler.get_scale()) >= float(scale_before)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        config: dict[str, Any],
        device: torch.device,
        pad_token_id: int,
        distributed: DistributedContext | None = None,
        tokenizer: Any | None = None,
        output_preparation: TrainingOutputPreparation | None = None,
        resume_checkpoint: dict[str, Any] | None = None,
    ) -> None:
        self.distributed = distributed or DistributedContext(device=device)
        self.is_main_process = self.distributed.is_main_process
        self.config = config
        self.device = device
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.training_config = config.get("training", {})
        requested_output_dir = Path(self.training_config.get("output_dir", "outputs/run"))
        resume_from = self.training_config.get("resume_from")
        if output_preparation is None:
            output_preparation = prepare_training_output_dir(
                requested_output_dir,
                resume_from=resume_from,
            )
        elif output_preparation.output_dir.resolve() != requested_output_dir.resolve():
            raise ValueError(
                "Prepared output directory does not match training.output_dir: "
                f"{output_preparation.output_dir} != {requested_output_dir}"
            )
        has_resume = bool(str(resume_from).strip()) if resume_from is not None else False
        expected_mode = "resume" if has_resume else "fresh"
        if output_preparation.mode != expected_mode:
            raise ValueError(
                f"Output directory was prepared for {output_preparation.mode}, "
                f"but trainer configuration requires {expected_mode}."
            )
        self.output_preparation = output_preparation
        self.run_mode = output_preparation.mode
        self.output_dir = output_preparation.output_dir
        if resume_checkpoint is not None and self.run_mode != "resume":
            raise ValueError("A preloaded resume checkpoint was provided for a fresh run.")
        self._resume_checkpoint = resume_checkpoint

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.ignore_index = int(self.training_config.get("ignore_index", -100))
        self.grad_accum_steps = int(self.training_config.get("grad_accum_steps", 1))

        compile_unavailable = False
        if bool(self.training_config.get("compile", False)):
            if hasattr(torch, "compile"):
                self.model = torch.compile(self.model)
            else:
                compile_unavailable = True

        if self.distributed.enabled:
            device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
            self.model = DistributedDataParallel(self.model, device_ids=device_ids)

        betas = tuple(self.training_config.get("betas", [0.9, 0.95]))
        self.base_learning_rate = float(self.training_config.get("learning_rate", 3e-4))
        optimizer_name = str(self.training_config.get("optimizer", "adamw")).lower()
        optimizer_model = unwrap_model(self.model)
        self.optimizer = optimizer_model.configure_optimizers(
            weight_decay=float(self.training_config.get("weight_decay", 0.1)),
            learning_rate=self.base_learning_rate,
            betas=(float(betas[0]), float(betas[1])),
            device_type=device.type,
            optimizer_type=optimizer_name,
            muon_momentum=float(self.training_config.get("muon_momentum", 0.95)),
            muon_ns_steps=int(self.training_config.get("muon_ns_steps", 5)),
            muon_update_scale=float(self.training_config.get("muon_update_scale", 1.0)),
            muon_nesterov=bool(self.training_config.get("muon_nesterov", True)),
            muon_orthogonalization=str(
                self.training_config.get("muon_orthogonalization", "polar_express")
            ),
            muon_row_equilibration=bool(
                self.training_config.get("muon_row_equilibration", True)
            ),
            muon_renormalize=bool(self.training_config.get("muon_renormalize", True)),
            embedding_learning_rate=self._optional_float("embedding_lr"),
            value_embedding_learning_rate=self._optional_float("value_embedding_lr"),
            lm_head_learning_rate=self._optional_float("lm_head_lr"),
            matrix_learning_rate=self._optional_float("matrix_lr"),
            scalar_learning_rate=self._optional_float("scalar_lr"),
        )
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group.get("lr", self.base_learning_rate))
            group.setdefault("initial_weight_decay", group.get("weight_decay", 0.0))
        self.muon_momentum_schedule = bool(
            self.training_config.get(
                "muon_momentum_schedule",
                optimizer_name in {"muon", "muon_adamw", "hybrid_muon"},
            )
        )
        self.muon_weight_decay_schedule = bool(
            self.training_config.get(
                "muon_weight_decay_schedule",
                optimizer_name in {"muon", "muon_adamw", "hybrid_muon"},
            )
        )

        self.precision = resolve_precision(self.training_config, device.type)
        self.use_amp = self.precision.amp_enabled
        self.scaler = build_grad_scaler(self.precision)
        self.global_step = 0
        self.start_epoch = 0
        self.start_batch_in_epoch = 0
        self.best_val_loss = float("inf")
        # ``seen_train_tokens`` is retained as a compatibility alias for
        # supervised (non-ignore) labels on successfully applied updates.
        self.seen_train_tokens = 0
        self.processed_train_tokens = 0
        self.attempted_optimizer_steps = 0
        self.overflow_skip_count = 0
        self.nonfinite_grad_skip_count = 0
        self.eval_call_count = 0
        self.loss_guard = LossDescentGuard(
            LossGuardConfig.from_value(self.training_config.get("loss_guard"))
        )
        self.run_identity = build_training_identity(config, train_loader.dataset)

        self.steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        self.tokens_per_step_estimate = self._estimate_tokens_per_step()
        max_steps = self._resolve_max_steps()
        self.max_train_tokens_limit = self._resolve_max_train_tokens()
        estimated_steps_from_tokens = self._estimate_steps_from_tokens(self.max_train_tokens_limit)
        configured_max_epochs = int(self.training_config.get("max_epochs", 1))
        self.max_steps_limit = max_steps
        schedule_steps = max_steps if max_steps is not None else estimated_steps_from_tokens
        if schedule_steps is not None:
            self.max_epochs = max(configured_max_epochs, math.ceil(schedule_steps / self.steps_per_epoch))
        else:
            self.max_epochs = configured_max_epochs
        self.total_steps = schedule_steps if schedule_steps is not None else self.steps_per_epoch * self.max_epochs
        self._validate_warmup_config()
        self.warmup_steps = self._resolve_warmup_steps()
        self.warmup_tokens = resolve_warmup_tokens(
            self.training_config,
            total_tokens=self.max_train_tokens_limit,
            tokens_per_step=self.tokens_per_step_estimate,
        )
        self.schedule_total_steps = self.total_steps
        self.schedule_total_tokens = self.max_train_tokens_limit
        self.lr_scheduler = CosineLRScheduler(
            learning_rate=float(self.training_config.get("learning_rate", 3e-4)),
            min_lr=self._minimum_lr(),
            total_steps=self.schedule_total_steps,
            warmup_steps=self.warmup_steps,
            decay=bool(self.training_config.get("lr_decay", True)),
        )
        self.z_loss_weight = float(self.training_config.get("z_loss_weight", 0.0))
        self.session_start_global_step = 0
        self.session_start_seen_train_tokens = 0
        self.session_start_processed_train_tokens = 0
        self._validate_and_cache_resume_checkpoint()
        self._setup_run_logger()
        if compile_unavailable and self.is_main_process:
            self.logger.warning("training.compile=true ignored because torch.compile is unavailable")

    def _setup_run_logger(self) -> None:
        log_file = self.output_dir / "train.log"
        if not self.is_main_process and bool(self.training_config.get("log_rank_files", False)):
            log_file = self.output_dir / f"train.rank{self.distributed.rank}.log"
        self.logger = setup_logging(
            log_file
            if self.is_main_process or self.training_config.get("log_rank_files", False)
            else None,
            file_mode="a" if self.run_mode == "resume" else "x",
            enable_stream=self.is_main_process,
        )
        if not self.is_main_process:
            return
        self.logger.info("--- training invocation ---")
        if self.run_mode == "resume":
            self.logger.info(
                "Run mode: RESUME | output_dir=%s | checkpoint=%s",
                self.output_dir,
                self.output_preparation.resume_from,
            )
        elif self.training_config.get("init_from"):
            self.logger.info(
                "Run mode: FRESH_INIT | output_dir=%s | weights=%s",
                self.output_dir,
                self.training_config.get("init_from"),
            )
        else:
            self.logger.info("Run mode: FRESH | output_dir=%s", self.output_dir)
        for archived_path in self.output_preparation.archived_artifacts:
            self.logger.info("Archived previous run metadata: %s", archived_path)

    def _validate_and_cache_resume_checkpoint(self) -> None:
        if self.run_mode != "resume":
            return
        checkpoint = self._resume_checkpoint
        if checkpoint is None:
            assert self.output_preparation.resume_from is not None
            checkpoint = load_checkpoint(
                self.output_preparation.resume_from,
                map_location="cpu",
            )
        validate_resume_model_config(checkpoint, unwrap_model(self.model))
        validate_resume_training_state(
            checkpoint,
            self.training_config,
            world_size=self.distributed.world_size,
            precision_name=self.precision.name,
            seed=int(self.config.get("seed", 42)),
            current_run_identity=self.run_identity,
        )
        self._restore_schedule_horizon(checkpoint)
        self._resume_checkpoint = checkpoint

    @property
    def supervised_train_tokens(self) -> int:
        return int(self.seen_train_tokens)

    def _schedule_axis(self, training_config: Mapping[str, Any] | None = None) -> str:
        config = self.training_config if training_config is None else training_config
        return "tokens" if config.get("max_train_tokens") is not None else "steps"

    def _schedule_state(self) -> ScheduleState:
        return ScheduleState(
            axis="tokens" if self.schedule_total_tokens is not None else "steps",
            total_steps=self.schedule_total_steps,
            warmup_steps=self.warmup_steps,
            total_tokens=self.schedule_total_tokens,
            warmup_tokens=self.warmup_tokens,
        )

    def _legacy_schedule_state(self, checkpoint: Mapping[str, Any]) -> ScheduleState:
        checkpoint_config = checkpoint.get("config")
        checkpoint_training = (
            checkpoint_config.get("training")
            if isinstance(checkpoint_config, Mapping)
            else None
        )
        if not isinstance(checkpoint_training, Mapping):
            raise ValueError(
                "Resume checkpoint has no schedule_state or saved training config; "
                "use init_from instead."
            )
        original_max_tokens = checkpoint_training.get("max_train_tokens")
        original_max_tokens = (
            None if original_max_tokens is None else max(1, int(original_max_tokens))
        )
        original_max_steps = checkpoint_training.get("max_steps")
        if original_max_steps is not None:
            original_total_steps = max(1, int(original_max_steps))
        elif original_max_tokens is not None:
            original_total_steps = self._estimate_steps_from_tokens(original_max_tokens) or 1
        else:
            original_total_steps = self.steps_per_epoch * int(
                checkpoint_training.get("max_epochs", 1)
            )
        original_warmup_steps = resolve_warmup_steps(
            dict(checkpoint_training),
            total_steps=original_total_steps,
            tokens_per_step=self.tokens_per_step_estimate,
        )
        original_warmup_tokens = resolve_warmup_tokens(
            dict(checkpoint_training),
            total_tokens=original_max_tokens,
            tokens_per_step=self.tokens_per_step_estimate,
        )
        return ScheduleState(
            axis="tokens" if original_max_tokens is not None else "steps",
            total_steps=original_total_steps,
            warmup_steps=original_warmup_steps,
            total_tokens=original_max_tokens,
            warmup_tokens=original_warmup_tokens,
        )

    def _restore_schedule_horizon(self, checkpoint: Mapping[str, Any]) -> None:
        extra = checkpoint.get("extra")
        raw_state = extra.get("schedule_state") if isinstance(extra, Mapping) else None
        state = (
            ScheduleState.from_mapping(raw_state)
            if isinstance(raw_state, Mapping)
            else self._legacy_schedule_state(checkpoint)
        )
        current_axis = self._schedule_axis()
        if state.axis != current_axis:
            raise ValueError(
                "Full resume cannot switch schedule progress axes: "
                f"checkpoint={state.axis}, current={current_axis}. Use init_from instead."
            )
        self.schedule_total_steps = int(state.total_steps)
        self.schedule_total_tokens = (
            None if state.total_tokens is None else int(state.total_tokens)
        )
        self.warmup_steps = int(state.warmup_steps)
        self.warmup_tokens = (
            None if state.warmup_tokens is None else int(state.warmup_tokens)
        )
        self.lr_scheduler = CosineLRScheduler(
            learning_rate=float(self.training_config.get("learning_rate", 3e-4)),
            min_lr=self._minimum_lr(),
            total_steps=self.schedule_total_steps,
            warmup_steps=self.warmup_steps,
            decay=bool(self.training_config.get("lr_decay", True)),
        )

    def _optional_float(self, key: str) -> float | None:
        value = self.training_config.get(key)
        if value in {None, ""}:
            return None
        return float(value)

    def _minimum_lr(self) -> float:
        value = self.training_config.get("min_lr")
        if value is None:
            value = self.base_learning_rate * 0.1
        return float(value)

    def _estimate_tokens_per_step(self) -> int:
        batch_size = getattr(self.train_loader, "batch_size", None)
        if batch_size is None:
            batch_size = int(self.training_config.get("batch_size", 1))
        model_config = getattr(unwrap_model(self.model), "config", None)
        seq_len = getattr(model_config, "max_seq_len", None)
        if seq_len is None:
            seq_len = getattr(getattr(self.train_loader, "dataset", None), "max_seq_len", 0)
        return (
            int(batch_size)
            * int(seq_len or 0)
            * self.grad_accum_steps
            * max(1, self.distributed.world_size)
        )

    def _resolve_max_steps(self) -> int | None:
        max_steps = self.training_config.get("max_steps")
        if max_steps is not None:
            return int(max_steps)
        return None

    def _resolve_max_train_tokens(self) -> int | None:
        max_train_tokens = self.training_config.get("max_train_tokens")
        if max_train_tokens is None:
            return None
        return max(1, int(max_train_tokens))

    def _estimate_steps_from_tokens(self, max_train_tokens: int | None) -> int | None:
        if max_train_tokens is None:
            return None
        if self.tokens_per_step_estimate <= 0:
            raise ValueError("max_train_tokens requires a positive tokens_per_step estimate")
        return max(1, math.ceil(int(max_train_tokens) / self.tokens_per_step_estimate))

    def _validate_warmup_config(self) -> None:
        validate_warmup_config(self.training_config)

    def _resolve_warmup_steps(self) -> int:
        return resolve_warmup_steps(
            self.training_config,
            total_steps=self.total_steps,
            tokens_per_step=self.tokens_per_step_estimate,
        )

    def _training_limits_reached(self) -> bool:
        if self.max_steps_limit is not None and self.global_step >= self.max_steps_limit:
            return True
        if (
            self.max_train_tokens_limit is not None
            and self.seen_train_tokens >= self.max_train_tokens_limit
        ):
            return True
        return False

    def _termination_state(self) -> tuple[str, bool]:
        step_reached = (
            self.max_steps_limit is not None
            and self.global_step >= self.max_steps_limit
        )
        tokens_reached = (
            self.max_train_tokens_limit is not None
            and self.seen_train_tokens >= self.max_train_tokens_limit
        )
        if step_reached and tokens_reached:
            return "max_steps_and_train_tokens_reached", True
        if step_reached:
            return "max_steps_reached", True
        if tokens_reached:
            return "max_train_tokens_reached", True
        if self.max_steps_limit is not None and self.max_train_tokens_limit is not None:
            return "max_epochs_exhausted_before_configured_targets", False
        if self.max_steps_limit is not None:
            return "max_epochs_exhausted_before_max_steps", False
        if self.max_train_tokens_limit is not None:
            return "max_epochs_exhausted_before_max_train_tokens", False
        return "max_epochs_reached", True

    def _target_reached_during_training(self) -> bool:
        if self.max_steps_limit is not None or self.max_train_tokens_limit is not None:
            return self._training_limits_reached()
        return False

    def _maybe_save_best(
        self,
        metrics: dict[str, float],
        epoch: int,
        batch_in_epoch: int,
        reason: str,
        save_best: bool = True,
    ) -> Path | None:
        loss = float(metrics.get("ce_loss", metrics.get("loss", float("nan"))))
        if not save_best or not math.isfinite(loss):
            return None
        if loss >= self.best_val_loss:
            return None
        self.best_val_loss = loss
        best_path = self._save("best.pt", epoch=epoch, batch_in_epoch=batch_in_epoch)
        if self.is_main_process:
            self.logger.info(
                "Saved best checkpoint: %s (%s step=%d val_loss=%.4f)",
                best_path,
                reason,
                self.global_step,
                loss,
            )
        return best_path

    def _run_final_eval(
        self,
        epoch: int,
        batch_in_epoch: int,
        eval_batches: int | None,
        save_best: bool,
    ) -> dict[str, float] | None:
        if self.val_loader is None or not bool(self.training_config.get("final_eval", True)):
            return None
        metrics = self.evaluate(max_batches=eval_batches)
        if self.is_main_process:
            self.logger.info(
                "final eval step=%d val_ce_loss=%.4f val_ppl=%.2f",
                self.global_step,
                metrics["ce_loss"],
                metrics["perplexity"],
            )
        self._maybe_save_best(
            metrics=metrics,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            reason="final_eval",
            save_best=save_best,
        )
        return metrics

    def _load_start_state(self) -> None:
        resume_from = self.training_config.get("resume_from")
        init_from = self.training_config.get("init_from")
        strict = bool(self.training_config.get("strict_load", True))

        if resume_from:
            checkpoint = self._resume_checkpoint
            if checkpoint is None:
                raise RuntimeError("Resume checkpoint was not validated during trainer setup.")
            restore_checkpoint(
                checkpoint,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                strict=strict,
            )
            self._resume_checkpoint = None
            self.global_step = int(checkpoint.get("step", 0))
            self.start_epoch = int(checkpoint.get("epoch", 0))
            self.start_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0) or 0)
            self.best_val_loss = float(checkpoint.get("best_val_loss") or float("inf"))
            extra = checkpoint.get("extra", {})
            if not isinstance(extra, dict):
                extra = {}
            restored_tokens = extra.get("seen_train_tokens", extra.get("global_valid_tokens"))
            restored_tokens = extra.get("supervised_train_tokens", restored_tokens)
            if restored_tokens is None and self.max_train_tokens_limit is not None:
                restored_tokens = self.global_step * self.tokens_per_step_estimate
            self.seen_train_tokens = int(restored_tokens or 0)
            self.processed_train_tokens = int(
                extra.get("processed_train_tokens", self.seen_train_tokens) or 0
            )
            self.attempted_optimizer_steps = int(
                extra.get("attempted_optimizer_steps", self.global_step) or 0
            )
            self.overflow_skip_count = int(extra.get("overflow_skip_count", 0) or 0)
            self.nonfinite_grad_skip_count = int(
                extra.get("nonfinite_grad_skip_count", 0) or 0
            )
            raw_loss_guard_state = extra.get("loss_guard_state")
            if isinstance(raw_loss_guard_state, Mapping):
                self.loss_guard.load_state_dict(raw_loss_guard_state)
            rng_states = extra.get("rng_states_by_rank")
            if isinstance(rng_states, list) and self.distributed.rank < len(rng_states):
                restore_rng_state(rng_states[self.distributed.rank])
            else:
                restore_rng_state(extra.get("rng_state"))
            self.logger.info("Resume model config validation passed.")
            self.logger.info(
                "Resumed training from %s at step %d epoch %d batch %d "
                "supervised_tokens=%d processed_tokens=%d",
                resume_from,
                self.global_step,
                self.start_epoch,
                self.start_batch_in_epoch,
                self.seen_train_tokens,
                self.processed_train_tokens,
            )
        elif init_from:
            load_checkpoint(
                init_from,
                model=self.model,
                map_location=self.device,
                strict=strict,
            )
            self.logger.info("Loaded model weights from %s", init_from)

    def _lr_at_step(self, step: int, next_valid_tokens: int = 0) -> float:
        if self.schedule_total_tokens is not None:
            progress_tokens = self.seen_train_tokens + max(0, int(next_valid_tokens))
            return self.lr_scheduler.lr_at_tokens(
                tokens_seen=progress_tokens,
                total_tokens=self.schedule_total_tokens,
                warmup_tokens=int(self.warmup_tokens or 0),
            )
        return self.lr_scheduler.lr_at(step)

    def _set_lr(self, lr: float) -> None:
        multiplier = float(lr) / self.base_learning_rate if self.base_learning_rate > 0 else 1.0
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = float(param_group.get("initial_lr", lr)) * multiplier

    def _set_optimizer_schedules(self) -> None:
        for param_group in self.optimizer.param_groups:
            if not param_group.get("use_muon", False):
                continue
            if self.muon_momentum_schedule:
                param_group["muon_momentum"] = muon_momentum_at_step(
                    step=self.global_step,
                    total_steps=self.schedule_total_steps,
                    warmup_steps=int(self.training_config.get("muon_momentum_warmup_steps", 400)),
                    start=float(self.training_config.get("muon_momentum_start", 0.85)),
                    peak=float(self.training_config.get("muon_momentum_peak", 0.97)),
                    final=float(self.training_config.get("muon_momentum_final", 0.90)),
                    warmdown_ratio=float(self.training_config.get("warmdown_ratio", 0.65)),
                )
            if self.muon_weight_decay_schedule:
                param_group["weight_decay"] = cosine_weight_decay_at_step(
                    base_weight_decay=float(param_group.get("initial_weight_decay", 0.0)),
                    step=self.global_step,
                    total_steps=self.schedule_total_steps,
                )

    def _save(self, name: str, epoch: int, batch_in_epoch: int = 0) -> Path:
        path = self.output_dir / name
        rng_states = all_gather_objects(capture_rng_state())
        if self.is_main_process:
            save_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=epoch,
                step=self.global_step,
                batch_in_epoch=batch_in_epoch,
                best_val_loss=None if math.isinf(self.best_val_loss) else self.best_val_loss,
                config=self.config,
                extra={
                    "rng_state": rng_states[0],
                    "rng_states_by_rank": rng_states,
                    "supervised_train_tokens": int(self.seen_train_tokens),
                    "seen_train_tokens": int(self.seen_train_tokens),
                    "processed_train_tokens": int(self.processed_train_tokens),
                    "attempted_optimizer_steps": int(self.attempted_optimizer_steps),
                    "overflow_skip_count": int(self.overflow_skip_count),
                    "nonfinite_grad_skip_count": int(self.nonfinite_grad_skip_count),
                    "loss_guard_state": self.loss_guard.state_dict(),
                    "schedule_state": self._schedule_state().to_dict(),
                    "run_identity": self.run_identity,
                    "resume_runtime": {
                        "world_size": int(self.distributed.world_size),
                        "precision": self.precision.name,
                    },
                },
            )
        barrier(self.distributed)
        return path

    def _prepare_canonical_artifact(self, path: Path) -> None:
        """Make one canonical metadata path safe for an exclusive write."""

        if not path.exists():
            return
        if self.run_mode != "resume":
            raise FileExistsError(
                f"Refusing to overwrite training artifact during a fresh run: {path}"
            )
        archived_path = archive_artifact(path)
        if archived_path is not None:
            self.logger.info("Archived previous run metadata: %s", archived_path)

    def _write_training_summary(
        self,
        elapsed_seconds: float,
        final_metrics: dict[str, Any] | None,
        *,
        status: str,
        termination_reason: str,
        target_reached: bool,
    ) -> Path | None:
        if not self.is_main_process or not bool(self.training_config.get("write_training_summary", True)):
            return None
        summary = build_training_summary(
            model=self.model,
            config=self.config,
            elapsed_seconds=elapsed_seconds,
            global_step=self.global_step,
            seen_train_tokens=self.seen_train_tokens,
            best_val_loss=self.best_val_loss,
            final_metrics=final_metrics,
            device=self.device,
            precision_name=self.precision.name,
            world_size=self.distributed.world_size,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            tokens_per_step_estimate=self.tokens_per_step_estimate,
            session_start_global_step=self.session_start_global_step,
            session_start_seen_train_tokens=self.session_start_seen_train_tokens,
            processed_train_tokens=self.processed_train_tokens,
            session_start_processed_train_tokens=(
                self.session_start_processed_train_tokens
            ),
            attempted_optimizer_steps=self.attempted_optimizer_steps,
            overflow_skip_count=self.overflow_skip_count,
            nonfinite_grad_skip_count=self.nonfinite_grad_skip_count,
            schedule_total_steps=self.schedule_total_steps,
            schedule_total_tokens=self.schedule_total_tokens,
            status=status,
            termination_reason=termination_reason,
            target_reached=target_reached,
            loss_guard=self.loss_guard.to_dict(),
        )
        summary_path = self.output_dir / "training_summary.json"
        self._prepare_canonical_artifact(summary_path)
        return write_training_summary(self.output_dir, summary, overwrite=False)

    def _build_training_result(
        self,
        *,
        status: str,
        termination_reason: str,
        target_reached: bool,
        final_path: Path | None = None,
        latest_path: Path | None = None,
        final_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": status,
            "termination_reason": termination_reason,
            "target_reached": bool(target_reached),
            "best_val_loss": self.best_val_loss,
            "global_step": self.global_step,
            "supervised_train_tokens": self.seen_train_tokens,
            "processed_train_tokens": self.processed_train_tokens,
            "seen_train_tokens": self.seen_train_tokens,
            "loss_guard": self.loss_guard.to_dict(),
        }
        if final_path is not None:
            result["final_checkpoint"] = str(final_path)
        if latest_path is not None:
            result["latest_checkpoint"] = str(latest_path)
        if final_metrics is not None:
            result["final_val_loss"] = final_metrics["loss"]
        return result

    def _abort_training(
        self,
        *,
        error_type: type[TrainingRunFailedError],
        message: str,
        termination_reason: str,
        target_reached: bool,
        elapsed_seconds: float,
        latest_path: Path,
        final_metrics: dict[str, Any] | None = None,
    ) -> None:
        result = self._build_training_result(
            status="failed",
            termination_reason=termination_reason,
            target_reached=target_reached,
            latest_path=latest_path,
            final_metrics=final_metrics,
        )
        summary_path = self._write_training_summary(
            elapsed_seconds=elapsed_seconds,
            final_metrics=final_metrics,
            status="failed",
            termination_reason=termination_reason,
            target_reached=target_reached,
        )
        if summary_path is not None:
            result["training_summary"] = str(summary_path)
        self.logger.error("Training aborted: %s", message)
        self.close()
        raise error_type(message, result)

    def _pause_training(
        self,
        *,
        elapsed_seconds: float,
        latest_path: Path,
    ) -> dict[str, Any]:
        termination_reason = "session_time_limit_reached"
        result = self._build_training_result(
            status="paused",
            termination_reason=termination_reason,
            target_reached=False,
            latest_path=latest_path,
        )
        summary_path = self._write_training_summary(
            elapsed_seconds=elapsed_seconds,
            final_metrics=None,
            status="paused",
            termination_reason=termination_reason,
            target_reached=False,
        )
        if summary_path is not None:
            result["training_summary"] = str(summary_path)
        self.logger.info(
            "Training session paused safely at step=%d supervised_train_tokens=%d; "
            "resume from %s",
            self.global_step,
            self.seen_train_tokens,
            latest_path,
        )
        self.close()
        return result

    def _log_loss_guard_decision(self, decision: LossGuardDecision) -> None:
        if not decision.message:
            return
        if decision.failed:
            self.logger.error(decision.message)
        elif decision.checked and not decision.passed:
            self.logger.warning(decision.message)
        else:
            self.logger.info(decision.message)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.precision.amp_dtype,
            enabled=self.use_amp,
        )

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def _set_sampler_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _eval_sampler_epoch(self) -> int:
        if bool(self.training_config.get("eval_resample_each_eval", False)):
            return self.global_step + self.eval_call_count
        return int(self.training_config.get("eval_sampler_epoch", 0) or 0)

    def _set_eval_sampler_epoch(self) -> None:
        if self.val_loader is None:
            return
        sampler = getattr(self.val_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._eval_sampler_epoch())
        self.eval_call_count += 1

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | None]:
        moved: dict[str, torch.Tensor | None] = {
            "input_ids": batch["input_ids"].to(self.device, non_blocking=True),
            "labels": batch["labels"].to(self.device, non_blocking=True),
        }
        attention_mask = batch.get("attention_mask")
        moved["attention_mask"] = (
            attention_mask.to(self.device, non_blocking=True) if attention_mask is not None else None
        )
        document_ids = batch.get("document_ids")
        document_mask_needed = batch.get("document_mask_needed")
        needs_document_mask = (
            bool(document_mask_needed.any().item())
            if isinstance(document_mask_needed, torch.Tensor)
            else document_ids is not None
        )
        moved["document_ids"] = (
            document_ids.to(self.device, non_blocking=True)
            if needs_document_mask and document_ids is not None
            else None
        )
        return moved

    def _valid_tokens(self, labels: torch.Tensor) -> int:
        return int((labels != self.ignore_index).sum().item())

    def _remaining_train_tokens(self) -> int | None:
        if self.max_train_tokens_limit is None:
            return None
        return max(0, int(self.max_train_tokens_limit) - int(self.seen_train_tokens))

    def _rank_token_quota(self, local_valid_tokens: int) -> int:
        remaining = self._remaining_train_tokens()
        if remaining is None:
            return int(local_valid_tokens)
        if remaining <= 0 or local_valid_tokens <= 0:
            return 0
        rank_counts = all_gather_int(local_valid_tokens, device=self.device)
        prior_ranks = sum(rank_counts[: self.distributed.rank])
        return max(0, min(int(local_valid_tokens), int(remaining) - int(prior_ranks)))

    def _mask_labels_to_token_quota(self, labels: torch.Tensor, quota: int) -> int:
        quota = max(0, int(quota))
        flat_labels = labels.reshape(-1)
        valid_positions = flat_labels != self.ignore_index
        valid_count = int(valid_positions.sum().item())
        if quota >= valid_count:
            return valid_count
        if quota <= 0:
            flat_labels[valid_positions] = self.ignore_index
            return 0
        valid_indices = valid_positions.nonzero(as_tuple=False).flatten()
        flat_labels[valid_indices[quota:]] = self.ignore_index
        return quota

    def _loss_sum_from_outputs(
        self,
        outputs: dict[str, torch.Tensor | list | None],
        valid_tokens: int,
    ) -> torch.Tensor:
        loss_sum = outputs.get("loss_sum")
        if isinstance(loss_sum, torch.Tensor):
            return loss_sum
        loss = outputs.get("loss")
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError("Model did not return a training loss.")
        return loss * max(valid_tokens, 1)

    def _named_loss_sum_from_outputs(
        self,
        outputs: dict[str, torch.Tensor | list | None],
        name: str,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        value = outputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
        return fallback.new_zeros(()) if fallback is not None else torch.tensor(0.0, device=self.device)

    def _validate_loss_token_count(
        self,
        outputs: dict[str, torch.Tensor | list | None],
        expected_tokens: int,
    ) -> None:
        reported = outputs.get("loss_token_count")
        if reported is None:
            return
        if isinstance(reported, torch.Tensor):
            if reported.numel() != 1:
                raise RuntimeError("Model loss_token_count must be a scalar")
            reported_count = int(reported.detach().item())
        else:
            reported_count = int(reported)
        if reported_count != int(expected_tokens):
            raise RuntimeError(
                "Loss normalization mismatch: model reported "
                f"{reported_count} supervised tokens, labels contain {expected_tokens}."
            )

    def _optimizer_step(self, lr: float, grad_clip: float) -> OptimizerStepResult:
        self._set_lr(lr)
        self._set_optimizer_schedules()
        self.attempted_optimizer_steps += 1
        scale_before = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        max_norm = grad_clip if grad_clip > 0 else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        grad_norm_value = float(
            grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        )
        if not math.isfinite(grad_norm_value) and not self.precision.scaler_enabled:
            self.optimizer.zero_grad(set_to_none=True)
            self.nonfinite_grad_skip_count += 1
            return OptimizerStepResult(
                grad_norm=grad_norm_value,
                applied=False,
                skip_reason="nonfinite_gradients",
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        step_applied = _grad_scaler_step_applied(self.scaler, scale_before)
        self.optimizer.zero_grad(set_to_none=True)
        if step_applied:
            self.global_step += 1
        else:
            self.overflow_skip_count += 1
        return OptimizerStepResult(
            grad_norm=grad_norm_value,
            applied=step_applied,
            skip_reason=None if step_applied else "amp_overflow",
        )

    def _train_accumulation_group(
        self,
        group: list[tuple[int, dict[str, torch.Tensor]]],
        grad_clip: float,
    ) -> TrainStepMetrics:
        prepared: list[tuple[int, dict[str, torch.Tensor | None], int]] = []
        initial_valid_tokens = 0
        local_processed_tokens = 0
        for batch_idx, batch in group:
            moved = self._move_batch(batch)
            input_ids = moved["input_ids"]
            labels = moved["labels"]
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            local_processed_tokens += int(input_ids.numel())
            valid_tokens = self._valid_tokens(labels)
            prepared.append((batch_idx, moved, valid_tokens))
            initial_valid_tokens += valid_tokens

        local_token_quota = self._rank_token_quota(initial_valid_tokens)
        capped_prepared: list[tuple[int, dict[str, torch.Tensor | None], int]] = []
        quota_left = local_token_quota
        total_valid_tokens = 0
        for batch_idx, moved, valid_tokens in prepared:
            labels = moved["labels"]
            assert isinstance(labels, torch.Tensor)
            if local_token_quota < initial_valid_tokens:
                valid_tokens = self._mask_labels_to_token_quota(labels, quota_left)
            quota_left -= valid_tokens
            total_valid_tokens += valid_tokens
            capped_prepared.append((batch_idx, moved, valid_tokens))
        prepared = capped_prepared

        valid_tokens_tensor = torch.tensor(
            float(total_valid_tokens),
            dtype=torch.float64,
            device=self.device,
        )
        global_valid_tokens = int(all_reduce_sum(valid_tokens_tensor).item())
        if global_valid_tokens <= 0:
            raise RuntimeError("Accumulation group has no valid training tokens.")
        processed_tokens_tensor = torch.tensor(
            float(local_processed_tokens),
            dtype=torch.float64,
            device=self.device,
        )
        global_processed_tokens = int(all_reduce_sum(processed_tokens_tensor).item())

        group_total_loss_sum = 0.0
        group_ce_loss_sum = 0.0
        group_z_loss_sum = 0.0
        for micro_idx, (_, moved, valid_tokens) in enumerate(prepared):
            input_ids = moved["input_ids"]
            labels = moved["labels"]
            attention_mask = moved["attention_mask"]
            document_ids = moved.get("document_ids")
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(labels, torch.Tensor)

            no_sync = getattr(self.model, "no_sync", None)
            sync_gradients = micro_idx == len(prepared) - 1
            sync_context = nullcontext() if sync_gradients or no_sync is None else no_sync()
            with sync_context:
                with self._autocast():
                    outputs = self.model(
                        input_ids,
                        labels=labels,
                        attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                        document_ids=document_ids if isinstance(document_ids, torch.Tensor) else None,
                        ignore_index=self.ignore_index,
                        z_loss_weight=self.z_loss_weight,
                        return_logits=False,
                    )
                    self._validate_loss_token_count(outputs, valid_tokens)
                    loss_sum = self._loss_sum_from_outputs(outputs, valid_tokens)
                    ce_loss_sum = self._named_loss_sum_from_outputs(
                        outputs,
                        name="ce_loss_sum",
                        fallback=loss_sum,
                    )
                    z_loss_sum = self._named_loss_sum_from_outputs(
                        outputs,
                        name="z_loss_sum",
                        fallback=loss_sum,
                    )
                    scaled_loss = (
                        loss_sum
                        * max(1, self.distributed.world_size)
                        / global_valid_tokens
                    )

                raw_loss = loss_sum.detach() / max(valid_tokens, 1)
                if not torch.isfinite(raw_loss):
                    raise FloatingPointError(f"Non-finite loss detected: {raw_loss.item()}")

                self.scaler.scale(scaled_loss).backward()
            group_total_loss_sum += float(loss_sum.detach().item())
            group_ce_loss_sum += float(ce_loss_sum.detach().item())
            group_z_loss_sum += float(z_loss_sum.detach().item())

        lr = self._lr_at_step(self.global_step, next_valid_tokens=global_valid_tokens)
        optimizer_step = self._optimizer_step(lr=lr, grad_clip=grad_clip)
        self.processed_train_tokens += global_processed_tokens
        if optimizer_step.applied:
            self.seen_train_tokens += global_valid_tokens
        group_sums = torch.tensor(
            [group_total_loss_sum, group_ce_loss_sum, group_z_loss_sum],
            dtype=torch.float64,
            device=self.device,
        )
        global_sums = all_reduce_sum(group_sums)
        return TrainStepMetrics(
            total_loss=float(global_sums[0].item()) / global_valid_tokens,
            ce_loss=float(global_sums[1].item()) / global_valid_tokens,
            z_loss=float(global_sums[2].item()) / global_valid_tokens,
            lr=lr,
            grad_norm=optimizer_step.grad_norm,
            tokens=global_valid_tokens,
            processed_tokens=global_processed_tokens,
            step_applied=optimizer_step.applied,
            skip_reason=optimizer_step.skip_reason,
        )

    @torch.no_grad()
    def evaluate(self, max_batches: int | None = None) -> dict[str, float]:
        if self.val_loader is None:
            return {"loss": float("nan"), "ce_loss": float("nan"), "perplexity": float("nan")}

        self._set_eval_sampler_epoch()
        return evaluate_language_model(
            model=self.model,
            data_loader=self.val_loader,
            device=self.device,
            ignore_index=self.ignore_index,
            max_batches=max_batches,
            autocast_factory=self._autocast,
            reduce_distributed=True,
            tokenizer=self.tokenizer,
        ).to_dict()

    def fit(self) -> dict[str, Any]:
        self._load_start_state()
        self.session_start_global_step = self.global_step
        self.session_start_seen_train_tokens = self.seen_train_tokens
        self.session_start_processed_train_tokens = self.processed_train_tokens
        fit_start_time = time.time()
        session_budget = TrainingSessionBudget.from_training_config(
            self.training_config,
            started_at=fit_start_time,
            launcher_deadline_at=session_deadline_from_environment(),
        )
        if self.is_main_process:
            config_path = self.output_dir / "config.yaml"
            self._prepare_canonical_artifact(config_path)
            if self.run_mode == "resume":
                self._prepare_canonical_artifact(self.output_dir / "training_summary.json")
            save_config(self.config, config_path, overwrite=False)
        barrier(self.distributed)

        if self.is_main_process:
            self.logger.info(
                "Device: %s | precision=%s | AMP: %s | world_size=%d",
                self.device,
                self.precision.name,
                self.use_amp,
                self.distributed.world_size,
            )
            self.logger.info(
                "Trainable parameters: %.2fM",
                count_parameters(unwrap_model(self.model)) / 1e6,
            )
            self.logger.info(
                "Training limit estimate: %d optimizer steps | persisted schedule horizon: %d",
                self.total_steps,
                self.schedule_total_steps,
            )
            self.logger.info("Estimated tokens per optimizer step: %d", self.tokens_per_step_estimate)
            self.logger.info("Effective warmup steps: %d", self.warmup_steps)
            if self.max_train_tokens_limit is not None:
                self.logger.info(
                    "Token-based limit: %d supervised train tokens | "
                    "schedule_horizon_tokens=%d | warmup_tokens=%d",
                    self.max_train_tokens_limit,
                    int(self.schedule_total_tokens or 0),
                    int(self.warmup_tokens or 0),
                )
            if session_budget.enabled:
                self.logger.info(
                    "Resumable session deadline: %.0f | remaining after initialization: %.0f seconds",
                    float(session_budget.deadline_at or 0.0),
                    float(session_budget.remaining_seconds(time.time()) or 0.0),
                )

        eval_batches = self.training_config.get("eval_batches")
        eval_batches = int(eval_batches) if eval_batches is not None else None
        save_best = bool(self.training_config.get("save_best", True))

        should_run_initial_eval = self.val_loader is not None and (
            bool(self.training_config.get("initial_eval", False))
            or self.loss_guard.needs_validation_baseline
        )
        if should_run_initial_eval:
            initial_metrics = self.evaluate(max_batches=eval_batches)
            self.logger.info(
                "initial eval step=%d val_ce_loss=%.4f val_ppl=%.2f",
                self.global_step,
                initial_metrics["ce_loss"],
                initial_metrics["perplexity"],
            )
            initial_decision = self.loss_guard.observe_validation(
                self.global_step,
                initial_metrics["ce_loss"],
            )
            self._log_loss_guard_decision(initial_decision)
        elif self.loss_guard.enabled and self.val_loader is None:
            self.logger.warning(
                "Loss guard has no validation loader; only the training-window check is active."
            )

        if self._training_limits_reached():
            self.logger.info(
                "Training already reached a configured limit at resume step=%d seen_tokens=%d; "
                "no optimizer step run.",
                self.global_step,
                self.seen_train_tokens,
            )
            latest_path = self._save(
                "latest.pt",
                epoch=self.start_epoch,
                batch_in_epoch=self.start_batch_in_epoch,
            )
            final_metrics = self._run_final_eval(
                epoch=self.start_epoch,
                batch_in_epoch=self.start_batch_in_epoch,
                eval_batches=eval_batches,
                save_best=save_best,
            )
            if final_metrics is not None:
                final_decision = self.loss_guard.observe_validation(
                    self.global_step,
                    final_metrics["ce_loss"],
                )
                self._log_loss_guard_decision(final_decision)
            final_path = self._save(
                "final.pt",
                epoch=self.start_epoch,
                batch_in_epoch=self.start_batch_in_epoch,
            )
            self.logger.info("Saved latest checkpoint: %s", latest_path)
            self.logger.info("Final checkpoint: %s", final_path)
            termination_reason, target_reached = self._termination_state()
            result = self._build_training_result(
                status="completed",
                termination_reason=termination_reason,
                target_reached=target_reached,
                final_path=final_path,
                latest_path=latest_path,
                final_metrics=final_metrics,
            )
            summary_path = self._write_training_summary(
                elapsed_seconds=time.time() - fit_start_time,
                final_metrics=final_metrics,
                status="completed",
                termination_reason=termination_reason,
                target_reached=target_reached,
            )
            if summary_path is not None:
                result["training_summary"] = str(summary_path)
            self.close()
            return result

        grad_accum_steps = self.grad_accum_steps
        max_epochs = self.max_epochs
        log_interval = int(self.training_config.get("log_interval", 20))
        eval_interval = int(self.training_config.get("eval_interval", 500))
        save_interval = int(self.training_config.get("save_interval", 1000))
        grad_clip = float(self.training_config.get("grad_clip", 1.0))

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        last_log_time = start_time
        recent_total_losses: list[float] = []
        recent_ce_losses: list[float] = []
        recent_z_losses: list[float] = []
        recent_grad_norms: list[float] = []

        stop_training = False
        pause_training = False
        final_epoch = self.start_epoch
        final_batch_in_epoch = self.start_batch_in_epoch
        latest_path: Path | None = None
        for epoch in range(self.start_epoch, max_epochs):
            self._set_sampler_epoch(epoch)
            final_epoch = epoch + 1
            skip_batches = self.start_batch_in_epoch if epoch == self.start_epoch else 0
            direct_resume = set_loader_start_batch(self.train_loader, skip_batches)
            if skip_batches:
                if direct_resume:
                    self.logger.info(
                        "Resuming directly after %d already-processed batches in epoch %d",
                        skip_batches,
                        epoch,
                    )
                else:
                    self.logger.info(
                        "Skipping %d already-processed batches in epoch %d",
                        skip_batches,
                        epoch,
                    )
            remaining_batches = len(self.train_loader)
            epoch_total_batches = (
                skip_batches + remaining_batches if direct_resume else remaining_batches
            )
            progress = tqdm(
                enumerate(self.train_loader, start=skip_batches if direct_resume else 0),
                total=epoch_total_batches,
                initial=skip_batches if direct_resume else 0,
                desc=f"epoch {epoch + 1}/{max_epochs}",
                leave=False,
                disable=not self.is_main_process,
            )
            accumulation_group: list[tuple[int, dict[str, torch.Tensor]]] = []
            for batch_idx, batch in progress:
                if not direct_resume and batch_idx < skip_batches:
                    continue
                final_batch_in_epoch = batch_idx + 1
                accumulation_group.append((batch_idx, batch))
                if (
                    len(accumulation_group) < grad_accum_steps
                    and batch_idx + 1 < epoch_total_batches
                ):
                    continue

                step_metrics = self._train_accumulation_group(
                    accumulation_group,
                    grad_clip=grad_clip,
                )
                last_batch_idx = accumulation_group[-1][0]
                accumulation_group = []
                if not step_metrics.step_applied:
                    self.logger.warning(
                        "Skipped optimizer step (%s); step=%d supervised_tokens=%d "
                        "processed_tokens=%d and optimizer schedules remain unchanged.",
                        step_metrics.skip_reason or "unknown",
                        self.global_step,
                        self.seen_train_tokens,
                        self.processed_train_tokens,
                    )
                    progress.set_postfix(
                        skipped=step_metrics.skip_reason or "unknown",
                        lr=f"{step_metrics.lr:.2e}",
                    )
                    continue

                recent_total_losses.append(step_metrics.total_loss)
                recent_ce_losses.append(step_metrics.ce_loss)
                recent_z_losses.append(step_metrics.z_loss)
                recent_grad_norms.append(step_metrics.grad_norm)

                train_guard_decision = self.loss_guard.observe_train(
                    self.global_step,
                    step_metrics.ce_loss,
                )
                self._log_loss_guard_decision(train_guard_decision)
                if train_guard_decision.failed:
                    progress.close()
                    latest_path = self._save(
                        "latest.pt",
                        epoch=epoch,
                        batch_in_epoch=last_batch_idx + 1,
                    )
                    self._abort_training(
                        error_type=LossDescentGuardError,
                        message=train_guard_decision.message
                        or "Training loss did not descend.",
                        termination_reason=(
                            self.loss_guard.failure_reason
                            or "training_loss_did_not_descend"
                        ),
                        target_reached=self._target_reached_during_training(),
                        elapsed_seconds=time.time() - fit_start_time,
                        latest_path=latest_path,
                    )

                window = min(len(recent_ce_losses), log_interval)
                mean_recent_ce_loss = sum(recent_ce_losses[-log_interval:]) / window
                mean_recent_total_loss = sum(recent_total_losses[-log_interval:]) / window
                mean_recent_z_loss = sum(recent_z_losses[-log_interval:]) / window
                mean_recent_grad_norm = sum(recent_grad_norms[-log_interval:]) / window
                progress.set_postfix(
                    ce_loss=f"{mean_recent_ce_loss:.4f}",
                    grad_norm=f"{mean_recent_grad_norm:.3f}",
                    lr=f"{step_metrics.lr:.2e}",
                )

                if self.global_step % log_interval == 0:
                    elapsed = time.time() - last_log_time
                    last_log_time = time.time()
                    self.logger.info(
                        "step=%d ce_loss=%.4f ce_ppl=%.2f total_loss=%.4f z_loss=%.6f "
                        "grad_norm=%.4f lr=%.3e supervised_tokens=%d "
                        "processed_tokens=%d cumulative_supervised_tokens=%d "
                        "cumulative_processed_tokens=%d interval=%s",
                        self.global_step,
                        mean_recent_ce_loss,
                        perplexity_from_loss(mean_recent_ce_loss),
                        mean_recent_total_loss,
                        mean_recent_z_loss,
                        mean_recent_grad_norm,
                        step_metrics.lr,
                        step_metrics.tokens,
                        step_metrics.processed_tokens,
                        self.seen_train_tokens,
                        self.processed_train_tokens,
                        format_seconds(elapsed),
                    )

                if self.val_loader is not None and eval_interval > 0 and self.global_step % eval_interval == 0:
                    metrics = self.evaluate(max_batches=eval_batches)
                    self.logger.info(
                        "eval step=%d val_ce_loss=%.4f val_ppl=%.2f",
                        self.global_step,
                        metrics["ce_loss"],
                        metrics["perplexity"],
                    )
                    val_guard_decision = self.loss_guard.observe_validation(
                        self.global_step,
                        metrics["ce_loss"],
                    )
                    self._log_loss_guard_decision(val_guard_decision)
                    if val_guard_decision.failed:
                        progress.close()
                        latest_path = self._save(
                            "latest.pt",
                            epoch=epoch,
                            batch_in_epoch=last_batch_idx + 1,
                        )
                        self._abort_training(
                            error_type=LossDescentGuardError,
                            message=val_guard_decision.message
                            or "Validation loss did not improve.",
                            termination_reason=(
                                self.loss_guard.failure_reason
                                or "validation_loss_did_not_improve"
                            ),
                            target_reached=self._target_reached_during_training(),
                            elapsed_seconds=time.time() - fit_start_time,
                            latest_path=latest_path,
                            final_metrics=metrics,
                        )
                    if save_best and metrics["loss"] < self.best_val_loss:
                        self._maybe_save_best(
                            metrics=metrics,
                            epoch=epoch,
                            batch_in_epoch=last_batch_idx + 1,
                            reason="interval_eval",
                            save_best=save_best,
                        )

                if save_interval > 0 and self.global_step % save_interval == 0:
                    latest_path = self._save("latest.pt", epoch=epoch, batch_in_epoch=last_batch_idx + 1)
                    self.logger.info("Saved latest checkpoint: %s", latest_path)

                if self._training_limits_reached():
                    stop_training = True
                    break
                local_pause = session_budget.should_pause(time.time())
                pause_training = any(
                    all_gather_int(int(local_pause), device=self.device)
                )
                if pause_training:
                    progress.close()
                    latest_path = self._save(
                        "latest.pt",
                        epoch=epoch,
                        batch_in_epoch=last_batch_idx + 1,
                    )
                    self.logger.info(
                        "Saved session-boundary checkpoint: %s",
                        latest_path,
                    )
                    break

            if stop_training:
                final_epoch = epoch
                latest_path = self._save(
                    "latest.pt",
                    epoch=epoch,
                    batch_in_epoch=final_batch_in_epoch,
                )
                self.logger.info("Saved latest checkpoint: %s", latest_path)
                break
            if pause_training:
                break
            self.start_batch_in_epoch = 0
            final_batch_in_epoch = 0
            latest_path = self._save("latest.pt", epoch=epoch + 1, batch_in_epoch=0)
            self.logger.info("Saved latest checkpoint: %s", latest_path)

        if pause_training:
            if latest_path is None:
                raise RuntimeError("Session pause requested without a saved checkpoint.")
            return self._pause_training(
                elapsed_seconds=time.time() - fit_start_time,
                latest_path=latest_path,
            )

        termination_reason, target_reached = self._termination_state()
        if not target_reached:
            latest_path = self._save(
                "latest.pt",
                epoch=final_epoch,
                batch_in_epoch=final_batch_in_epoch,
            )
            target_description = (
                f"max_steps={self.max_steps_limit}, "
                f"max_train_tokens={self.max_train_tokens_limit}"
            )
            self._abort_training(
                error_type=TrainingTargetNotReachedError,
                message=(
                    "Training exhausted its epoch budget before reaching the explicit "
                    f"target ({target_description}); global_step={self.global_step}, "
                    f"supervised_train_tokens={self.seen_train_tokens}, "
                    f"attempted_optimizer_steps={self.attempted_optimizer_steps}, "
                    f"overflow_skips={self.overflow_skip_count}, "
                    f"nonfinite_gradient_skips={self.nonfinite_grad_skip_count}."
                ),
                termination_reason=termination_reason,
                target_reached=False,
                elapsed_seconds=time.time() - fit_start_time,
                latest_path=latest_path,
            )

        final_metrics = self._run_final_eval(
            epoch=final_epoch,
            batch_in_epoch=final_batch_in_epoch,
            eval_batches=eval_batches,
            save_best=save_best,
        )
        if final_metrics is not None:
            final_guard_decision = self.loss_guard.observe_validation(
                self.global_step,
                final_metrics["ce_loss"],
            )
            self._log_loss_guard_decision(final_guard_decision)
            if final_guard_decision.failed:
                latest_path = self._save(
                    "latest.pt",
                    epoch=final_epoch,
                    batch_in_epoch=final_batch_in_epoch,
                )
                self._abort_training(
                    error_type=LossDescentGuardError,
                    message=final_guard_decision.message
                    or "Validation loss did not improve.",
                    termination_reason=(
                        self.loss_guard.failure_reason
                        or "validation_loss_did_not_improve"
                    ),
                    target_reached=target_reached,
                    elapsed_seconds=time.time() - fit_start_time,
                    latest_path=latest_path,
                    final_metrics=final_metrics,
                )

        final_path = self._save(
            "final.pt",
            epoch=final_epoch,
            batch_in_epoch=final_batch_in_epoch,
        )

        total_time = time.time() - start_time
        self.logger.info("Training finished in %s", format_seconds(total_time))
        self.logger.info("Final checkpoint: %s", final_path)
        result = self._build_training_result(
            status="completed",
            termination_reason=termination_reason,
            target_reached=target_reached,
            final_path=final_path,
            latest_path=latest_path,
            final_metrics=final_metrics,
        )
        summary_path = self._write_training_summary(
            elapsed_seconds=total_time,
            final_metrics=final_metrics,
            status="completed",
            termination_reason=termination_reason,
            target_reached=target_reached,
        )
        if summary_path is not None:
            result["training_summary"] = str(summary_path)
        self.close()
        return result
