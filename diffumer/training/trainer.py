"""Minimal training loop for DiffuMER masked diffusion."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from diffumer.model.diffumer_model import DiffuMERModel

from .losses import masked_diffusion_loss


TrainMode = Literal["full", "freeze_llada", "audio_adapter"]
SchedulerType = Literal["linear", "cosine", "constant"]
Precision = Literal["fp32", "fp16", "bf16"]
LossFunction = Callable[..., torch.Tensor]


@dataclass(slots=True)
class TrainerConfig:
    output_dir: str = "outputs/diffumer"
    epochs: int = 1
    max_steps: int | None = None
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    scheduler: SchedulerType = "linear"
    warmup_steps: int | None = None
    warmup_ratio: float = 0.0
    min_lr_ratio: float = 0.0
    precision: Precision = "fp32"
    train_mode: TrainMode = "audio_adapter"
    log_every_steps: int = 10
    save_every_steps: int | None = 500
    save_at_end: bool = True
    save_trainable_only: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must lie in [0, 1).")
        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative or None.")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must lie in [0, 1].")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive.")
        if self.save_every_steps is not None and self.save_every_steps <= 0:
            raise ValueError("save_every_steps must be positive or None.")
        if self.train_mode not in {"full", "freeze_llada", "audio_adapter"}:
            raise ValueError(f"Unsupported train_mode: {self.train_mode}")
        if self.scheduler not in {"linear", "cosine", "constant"}:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported precision: {self.precision}")


@dataclass(slots=True)
class TrainerState:
    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DiffuMERTrainer:
    """Connect a DataLoader/collator, DiffuMERModel, loss, and optimizer loop."""

    def __init__(
        self,
        model: DiffuMERModel,
        train_dataloader: DataLoader[Any],
        collator: Callable[..., Any],
        *,
        config: TrainerConfig | None = None,
        loss_fn: LossFunction = masked_diffusion_loss,
        optimizer: Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.collator = collator
        self.config = config or TrainerConfig()
        self.loss_fn = loss_fn
        self.state = TrainerState()
        seed_everything(self.config.seed)

        if len(train_dataloader) == 0:
            raise ValueError("train_dataloader must contain at least one batch.")

        self._configure_trainable_parameters(self.config.train_mode)
        self.trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not self.trainable_parameters:
            raise ValueError("No trainable model parameters remain.")

        self.optimizer = optimizer or AdamW(
            self.trainable_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
        self.total_steps = self._compute_total_steps()
        self.scheduler = scheduler or self._build_scheduler()

        device_type = self._primary_device_type()
        if self.config.precision == "fp16" and device_type != "cuda":
            raise ValueError("fp16 training requires a CUDA model device.")
        use_scaler = self.config.precision == "fp16" and device_type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def _configure_trainable_parameters(self, mode: TrainMode) -> None:
        if mode == "full":
            self.model.requires_grad_(True)
            self.model.set_llada_trainable(True)
        elif mode == "freeze_llada":
            self.model.requires_grad_(True)
            self.model.set_llada_trainable(False)
        else:
            self.model.requires_grad_(False)
            self.model.audio_adapter.requires_grad_(True)
            self.model.set_llada_trainable(False)

    def _compute_total_steps(self) -> int:
        steps_per_epoch = math.ceil(
            len(self.train_dataloader)
            / self.config.gradient_accumulation_steps
        )
        estimated_steps = steps_per_epoch * self.config.epochs
        if self.config.max_steps is not None:
            return min(estimated_steps, self.config.max_steps)
        return estimated_steps

    def _build_scheduler(self) -> LambdaLR:
        warmup_steps = self.config.warmup_steps
        if warmup_steps is None:
            warmup_steps = int(self.total_steps * self.config.warmup_ratio)
        warmup_steps = min(warmup_steps, self.total_steps)

        def lr_multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max(
                    self.config.min_lr_ratio,
                    float(step + 1) / float(warmup_steps),
                )
            if self.config.scheduler == "constant":
                return 1.0
            decay_steps = max(1, self.total_steps - warmup_steps)
            progress = min(
                1.0,
                max(0.0, (step - warmup_steps) / decay_steps),
            )
            if self.config.scheduler == "cosine":
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                factor = 1.0 - progress
            return self.config.min_lr_ratio + (
                1.0 - self.config.min_lr_ratio
            ) * factor

        return LambdaLR(self.optimizer, lr_lambda=lr_multiplier)

    def _primary_device_type(self) -> str:
        embedding_weight = self.model.llada_v.get_input_embeddings().weight
        return embedding_weight.device.type

    def _autocast_context(self) -> Any:
        if self.config.precision == "fp32":
            return nullcontext()
        dtype = (
            torch.float16
            if self.config.precision == "fp16"
            else torch.bfloat16
        )
        return torch.autocast(
            device_type=self._primary_device_type(),
            dtype=dtype,
        )

    def _forward_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        corruption_mask = batch.get("corruption_mask")
        mask_probabilities = batch.get("mask_probabilities")
        if corruption_mask is None or mask_probabilities is None:
            raise ValueError(
                "Training batches must contain corruption_mask and "
                "mask_probabilities. Use DiffuMERDataCollator(training=True)."
            )

        outputs = self.model(
            canvas=batch["canvas"],
            video_frames=batch.get("video_frames"),
            audio_features=batch.get("audio_features"),
            audio_attention_mask=batch.get("audio_attention_mask"),
            transcripts=batch.get("transcripts"),
        )
        return self.loss_fn(
            logits=outputs.logits,
            labels=outputs.labels,
            corruption_mask=corruption_mask,
            mask_probabilities=mask_probabilities,
        )

    def _next_position(
        self,
        epoch: int,
        batch_index: int,
    ) -> tuple[int, int]:
        next_batch = batch_index + 1
        if next_batch >= len(self.train_dataloader):
            return epoch + 1, 0
        return epoch, next_batch

    def _log(self, loss: float) -> None:
        payload = {
            "step": self.state.global_step,
            "epoch": self.state.epoch,
            "loss": loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    def train(
        self,
        *,
        resume_from_checkpoint: str | Path | None = None,
    ) -> TrainerState:
        if resume_from_checkpoint is not None:
            self.load_checkpoint(resume_from_checkpoint)

        self.optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        accumulated_microbatches = 0
        should_stop = self.state.global_step >= self.total_steps

        for epoch in range(self.state.epoch, self.config.epochs):
            if should_stop:
                break
            self.model.train()
            resume_batch = self.state.batch_in_epoch if epoch == self.state.epoch else 0

            for batch_index, batch in enumerate(self.train_dataloader):
                if batch_index < resume_batch:
                    continue

                accumulation = self.config.gradient_accumulation_steps
                window_start = (batch_index // accumulation) * accumulation
                window_end = min(
                    window_start + accumulation,
                    len(self.train_dataloader),
                )
                window_size = window_end - window_start

                with self._autocast_context():
                    loss = self._forward_loss(batch)
                    backward_loss = loss / window_size

                self.scaler.scale(backward_loss).backward()
                accumulated_loss += float(loss.detach().cpu())
                accumulated_microbatches += 1

                if batch_index + 1 != window_end:
                    continue

                self.scaler.unscale_(self.optimizer)
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        self.config.max_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                self.state.global_step += 1
                self.state.epoch, self.state.batch_in_epoch = self._next_position(
                    epoch,
                    batch_index,
                )

                if self.state.global_step % self.config.log_every_steps == 0:
                    self._log(accumulated_loss / accumulated_microbatches)
                    accumulated_loss = 0.0
                    accumulated_microbatches = 0

                if (
                    self.config.save_every_steps is not None
                    and self.state.global_step % self.config.save_every_steps == 0
                ):
                    self.save_checkpoint()

                if self.state.global_step >= self.total_steps:
                    should_stop = True
                    break

            if not should_stop:
                self.state.epoch = epoch + 1
                self.state.batch_in_epoch = 0

        if accumulated_microbatches:
            self._log(accumulated_loss / accumulated_microbatches)
        if self.config.save_at_end:
            self.save_checkpoint(name="checkpoint-final.pt")
        return self.state

    def _model_state_for_checkpoint(self) -> dict[str, torch.Tensor]:
        state_dict = self.model.state_dict()
        if not self.config.save_trainable_only:
            return state_dict
        trainable_names = {
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: value
            for name, value in state_dict.items()
            if name in trainable_names
        }

    def save_checkpoint(self, *, name: str | None = None) -> Path:
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = name or f"checkpoint-{self.state.global_step}.pt"
        checkpoint_path = output_dir / filename
        temporary_path = output_dir / f".{filename}.tmp"

        payload: dict[str, Any] = {
            "model": self._model_state_for_checkpoint(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "trainer_state": asdict(self.state),
            "trainer_config": asdict(self.config),
            "trainable_only": self.config.save_trainable_only,
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "checkpoint-final.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        incompatible = self.model.load_state_dict(payload["model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected model keys in checkpoint: "
                f"{incompatible.unexpected_keys[:10]}"
            )
        if not payload.get("trainable_only", True) and incompatible.missing_keys:
            raise RuntimeError(
                "Full checkpoint is missing model keys: "
                f"{incompatible.missing_keys[:10]}"
            )

        self.optimizer.load_state_dict(payload["optimizer"])
        for parameter, optimizer_state in self.optimizer.state.items():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(parameter.device)
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload.get("scaler", {}))
        self.state = TrainerState(**payload["trainer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
