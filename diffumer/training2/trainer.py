"""DiffuMER training with reconstructible checkpoints and verifier objectives."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Literal, Mapping

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from diffumer.model.audio_adapter import AudioAdapter
from diffumer.model.canvas import (
    CanvasBuilder,
    CanvasConfig,
    DIFFUMER_SPECIAL_TOKENS,
    MASK_TOKEN,
)
from diffumer.model.diffumer_model import DiffuMERModel

from diffumer.training2.losses import masked_diffusion_loss


TrainMode = Literal["full", "freeze_llada", "audio_adapter", "verifier"]
SchedulerType = Literal["linear", "cosine", "constant"]
Precision = Literal["fp32", "fp16", "bf16"]
LossFunction = Callable[..., torch.Tensor]
VerifierLossFunction = Callable[..., dict[str, torch.Tensor]]


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
    lambda_quality: float = 0.0
    lambda_grounding: float = 0.0
    lambda_adequacy: float = 0.0
    lambda_relation: float = 0.0
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
        if self.train_mode not in {"full", "freeze_llada", "audio_adapter", "verifier"}:
            raise ValueError(f"Unsupported train_mode: {self.train_mode}")
        weights = (
            self.lambda_quality, self.lambda_grounding,
            self.lambda_adequacy, self.lambda_relation,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("Verifier loss weights must be finite and non-negative.")
        if self.train_mode == "verifier" and not any(weights):
            raise ValueError("verifier mode requires at least one positive loss weight.")
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


def _plain_config(value: Any) -> Any:
    """Normalize construction data without pickling arbitrary config objects."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain_config(asdict(value))
    if hasattr(value, "to_dict"):
        return _plain_config(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_config(item) for item in value]
    if isinstance(value, (Path, torch.dtype)):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported construction config value: {type(value).__name__}")


def _audio_adapter_config(model: DiffuMERModel) -> dict[str, Any]:
    adapter = model.audio_adapter
    linears = [layer for layer in adapter.projector if isinstance(layer, nn.Linear)]
    if not linears:
        raise ValueError("Audio adapter projector contains no linear layers.")
    return {
        "hubert_hidden_size": adapter.hubert_hidden_size,
        "llada_hidden_size": adapter.llada_hidden_size,
        "num_audio_latents": adapter.num_audio_latents,
        "num_attention_heads": adapter.temporal_resampler.num_heads,
        "projector_hidden_size": (
            linears[0].out_features if len(linears) > 1 else adapter.llada_hidden_size
        ),
        "num_layers": len(linears),
        "dropout": adapter.temporal_resampler.dropout,
        "add_modality_embedding": adapter.audio_embedding is not None,
        "initializer_range": adapter.initializer_range,
    }


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
        model_config: Mapping[str, Any] | None = None,
        verifier_config: Mapping[str, Any] | None = None,
        verifier_loss_fn: VerifierLossFunction | None = None,
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.collator = collator
        self.config = config or TrainerConfig()
        self.loss_fn = loss_fn
        self.state = TrainerState()
        self.model_config = _plain_config(
            model_config or getattr(model, "model_construction_config", {})
        )
        self.verifier_config = _plain_config(verifier_config)
        self.verifier_is_trained = bool(getattr(model, "verifier_is_trained", False))
        self.verifier_loss_weights = {
            "quality": self.config.lambda_quality,
            "grounding": self.config.lambda_grounding,
            "adequacy": self.config.lambda_adequacy,
            "relation": self.config.lambda_relation,
        }
        self.verifier_loss_fn = verifier_loss_fn
        if any(self.verifier_loss_weights.values()):
            if self.model.verifier_heads is None:
                raise ValueError("Positive verifier loss weights require verifier_heads.")
            if self.verifier_loss_fn is None:
                from diffumer.training2 import losses as loss_module

                self.verifier_loss_fn = getattr(loss_module, "verifier_loss", None)
            if not callable(self.verifier_loss_fn):
                raise ValueError(
                    "Implement training2.losses.verifier_loss or pass verifier_loss_fn before "
                    "enabling verifier supervision."
                )
        self.last_loss_components: dict[str, float] = {}
        self._loss_component_totals: dict[str, float] = {}
        self._loss_component_batches = 0
        self._batch_has_verifier_supervision = False
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
        optimized_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimized_ids != {id(parameter) for parameter in self.trainable_parameters}:
            raise ValueError("Optimizer must cover exactly the selected trainable parameters.")
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
        elif mode == "audio_adapter":
            self.model.requires_grad_(False)
            self.model.audio_adapter.requires_grad_(True)
            self.model.set_llada_trainable(False)
        elif mode == "verifier":
            if self.model.verifier_heads is None:
                raise ValueError("verifier mode requires registered verifier_heads.")
            self.model.requires_grad_(False)
            self.model.verifier_heads.requires_grad_(True)
            self.model.set_llada_trainable(False)
        else:
            raise ValueError(f"Unsupported train_mode: {mode}")

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
        """Return a scalar; verifier annotations are loss inputs, never AVT inputs.

        verifier_loss_fn follows verifier_loss(predictions, targets, valid_masks)
        and returns quality/grounding/adequacy/relation scalar losses. Annotations
        use independent verification_targets and verification_supervision_mask
        mappings. Missing annotations must not be synthesized from gold text.
        """
        corruption_mask = batch.get("corruption_mask")
        mask_probabilities = batch.get("mask_probabilities")
        if corruption_mask is None or mask_probabilities is None:
            raise ValueError(
                "Training batches must contain corruption_mask and "
                "mask_probabilities. Use DiffuMERDataCollator(mode='train')."
            )

        model_inputs = {
            "video_frames": batch.get("video_frames"),
            "audio_features": batch.get("audio_features"),
            "audio_attention_mask": batch.get("audio_attention_mask"),
            "text_input_ids": batch.get("text_input_ids"),
            "text_attention_mask": batch.get("text_attention_mask"),
            "transcripts": batch.get("transcripts"),
        }
        enabled = any(self.verifier_loss_weights.values())
        targets = batch.get("verification_targets")
        valid_masks = batch.get("verification_supervision_mask")
        if enabled:
            if not isinstance(targets, Mapping) or not isinstance(valid_masks, Mapping):
                raise ValueError(
                    "Verifier losses require independent verification_targets and "
                    "verification_supervision_mask mappings."
                )
            conditions = self.model.encode_conditions(
                **model_inputs,
                batch_size=batch["canvas"].input_ids.shape[0],
            )
            # Both losses use this fresh canvas pass; AVT is encoded only once.
            outputs = self.model.verify_canvas(conditions, batch["canvas"])
        else:
            outputs = self.model(canvas=batch["canvas"], **model_inputs)
        diffusion_loss = self.loss_fn(
            logits=outputs.logits,
            labels=outputs.labels,
            corruption_mask=corruption_mask,
            mask_probabilities=mask_probabilities,
        )
        if not isinstance(diffusion_loss, torch.Tensor) or diffusion_loss.ndim != 0:
            raise ValueError("Diffusion loss must be a scalar tensor.")
        pieces = {
            name: diffusion_loss.new_zeros(())
            for name in self.verifier_loss_weights
        }
        self._batch_has_verifier_supervision = False
        total_loss = diffusion_loss
        if enabled:
            scores = outputs.verification_scores
            if not isinstance(scores, Mapping):
                raise TypeError("Verifier heads must return a named score mapping.")
            active_targets = {}
            active_masks = {}
            for name, weight in self.verifier_loss_weights.items():
                if weight == 0:
                    continue
                active_targets[name] = targets.get(name)
                active_masks[name] = valid_masks.get(name)
                target = active_targets[name]
                mask = active_masks[name]
                if target is not None:
                    if mask is None:
                        raise ValueError(f"{name} annotations require a valid supervision mask.")
                    target = torch.as_tensor(target)
                    mask = torch.as_tensor(mask, device=target.device, dtype=torch.bool)
                    if mask.shape != target.shape:
                        raise ValueError(f"{name} supervision mask must match target shape.")
                    self._batch_has_verifier_supervision |= bool(
                        (mask & target.ne(-100)).any()
                    )
            verifier_losses = self.verifier_loss_fn(
                predictions=scores,
                targets=active_targets,
                valid_masks=active_masks,
            )
            for name, weight in self.verifier_loss_weights.items():
                if weight == 0:
                    continue
                loss = verifier_losses.get(name)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                    raise ValueError(f"{name} verifier loss must be a scalar tensor.")
                pieces[name] = loss.to(diffusion_loss.device)
                total_loss = total_loss + weight * pieces[name]
            # Keep all-empty verifier batches backward-compatible even if the
            # supplied loss function returns detached zero tensors.
            score_anchor = next(
                (value for value in scores.values() if isinstance(value, torch.Tensor)),
                None,
            )
            if score_anchor is not None:
                total_loss = total_loss + score_anchor.reshape(-1)[:0].float().sum().to(
                    diffusion_loss.device
                )
        components = {"diffusion_loss": diffusion_loss}
        components.update({f"{name}_loss": value for name, value in pieces.items()})
        self.last_loss_components = {
            name: float(value.detach().cpu()) for name, value in components.items()
        }
        for name, value in self.last_loss_components.items():
            self._loss_component_totals[name] = self._loss_component_totals.get(name, 0.0) + value
        self._loss_component_batches += 1
        return total_loss

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
        if self._loss_component_batches:
            payload["loss_components"] = {
                name: value / self._loss_component_batches
                for name, value in self._loss_component_totals.items()
            }
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        self._loss_component_totals.clear()
        self._loss_component_batches = 0

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
        window_has_verifier_supervision = False
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
                window_has_verifier_supervision |= self._batch_has_verifier_supervision

                if batch_index + 1 != window_end:
                    continue

                if self.config.train_mode == "verifier" and not window_has_verifier_supervision:
                    # Avoid AdamW weight decay when no verifier annotation is valid.
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.epoch, self.state.batch_in_epoch = self._next_position(
                        epoch, batch_index
                    )
                    continue
                self.scaler.unscale_(self.optimizer)
                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        self.config.max_grad_norm,
                    )
                heads = self.model.verifier_heads
                verifier_has_gradients = heads is not None and any(
                    parameter.requires_grad and parameter.grad is not None
                    for parameter in heads.parameters()
                )
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                stepped = self.scaler.get_scale() >= old_scale
                if stepped and window_has_verifier_supervision and verifier_has_gradients:
                    self.verifier_is_trained = True
                    self.model.verifier_is_trained = True
                window_has_verifier_supervision = False
                self.optimizer.zero_grad(set_to_none=True)
                self.state.epoch, self.state.batch_in_epoch = self._next_position(
                    epoch,
                    batch_index,
                )
                if not stepped:
                    continue
                self.scheduler.step()
                self.state.global_step += 1

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
            name for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        # Keep auxiliary weights from earlier stages even when now frozen.
        return {
            name: value for name, value in state_dict.items()
            if name in trainable_names
            or name.startswith(("audio_adapter.", "verifier_heads."))
        }

    def _checkpoint_model_config(self) -> dict[str, Any]:
        llada_config = self.model.llada_v.config
        base_path = self.model_config.get("model_name_or_path") or getattr(
            llada_config, "_name_or_path", None
        )
        if not base_path:
            raise ValueError(
                "Pass model_config={'model_name_or_path': ...} to the trainer "
                "when the base model config does not contain _name_or_path."
            )
        heads = self.model.verifier_heads
        head_config = self.verifier_config
        if heads is not None and head_config is None:
            head_config = self.model_config.get("verifier")
            if head_config is None:
                head_config = getattr(heads, "config", None)
            if head_config is None:
                raise ValueError(
                    "Saving a verifier requires verifier_config or "
                    "verifier_heads.config for accurate reconstruction."
                )
        head_state = heads.state_dict() if heads is not None else {}
        loading = dict(self.model_config.get("loader_kwargs", {}))
        if "torch_dtype" not in loading:
            dtype = self.model.llada_v.get_input_embeddings().weight.dtype
            if dtype in {torch.float16, torch.bfloat16}:
                loading["torch_dtype"] = str(dtype).removeprefix("torch.")
        return _plain_config({
            "model_name_or_path": str(base_path),
            "model_base": self.model_config.get("model_base"),
            "model_name": self.model_config.get("model_name", "llava_llada"),
            "loader_kwargs": loading,
            "llada_config": llada_config.to_dict(),
            "canvas": asdict(self.model.canvas_builder.config),
            "audio_adapter": _audio_adapter_config(self.model),
            "context_length": self.model.context_length,
            "max_visual_tokens": self.model.max_visual_tokens,
            "max_text_tokens": self.model.max_text_tokens,
            "verifier": {
                "present": heads is not None,
                "class": (
                    f"{type(heads).__module__}.{type(heads).__qualname__}"
                    if heads is not None else None
                ),
                "config": head_config,
                "is_trained": self.verifier_is_trained,
                "weights_included": bool(head_state),
                "trained_weights_included": self.verifier_is_trained and bool(head_state),
            },
        })

    def _tokenizer_snapshot(self, output_dir: Path) -> dict[str, Any]:
        tokenizer = self.model.tokenizer
        with TemporaryDirectory(prefix="tokenizer-", dir=output_dir) as directory:
            tokenizer.save_pretrained(directory)
            root = Path(directory)
            files = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in root.rglob("*") if path.is_file()
            }
        vocabulary = tokenizer.get_vocab()
        return {
            "files": files,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
            "vocabulary": vocabulary,
            "special_token_mapping": {
                token: int(vocabulary[token])
                for token in (*DIFFUMER_SPECIAL_TOKENS, MASK_TOKEN)
            },
            "canvas_token_ids": asdict(self.model.canvas_builder.token_ids),
            "input_embedding_rows": self.model.llada_v.get_input_embeddings().weight.shape[0],
            "lm_head_rows": self.model.llada_v.lm_head.weight.shape[0],
        }

    def _special_token_rows(self) -> dict[str, Any]:
        tokenizer = self.model.tokenizer
        vocabulary = tokenizer.get_vocab()
        canvas_ids = self.model.canvas_builder.token_ids
        token_ids = torch.tensor(
            sorted(
                set(tokenizer.all_special_ids)
                | {vocabulary[token] for token in DIFFUMER_SPECIAL_TOKENS}
                | {canvas_ids.mask, canvas_ids.eos, canvas_ids.pad}
            ),
            dtype=torch.long,
        )
        input_weight = self.model.llada_v.get_input_embeddings().weight
        head = self.model.llada_v.lm_head
        return {
            "token_ids": token_ids,
            "input_embeddings": input_weight.detach().index_select(
                0, token_ids.to(input_weight.device)
            ).cpu().clone(),
            "lm_head": head.weight.detach().index_select(
                0, token_ids.to(head.weight.device)
            ).cpu().clone(),
            "lm_head_bias": (
                head.bias.detach().index_select(
                    0, token_ids.to(head.bias.device)
                ).cpu().clone()
                if head.bias is not None else None
            ),
        }

    def _optimizer_parameter_names(self) -> list[list[str]]:
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        return [
            [names[id(parameter)] for parameter in group["params"]]
            for group in self.optimizer.param_groups
        ]

    def save_checkpoint(self, *, name: str | None = None) -> Path:
        output_dir = Path(self.config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = name or f"checkpoint-{self.state.global_step}.pt"
        if Path(filename).name != filename:
            raise ValueError("Checkpoint name must be a filename, not a path.")
        checkpoint_path = output_dir / filename
        temporary_path = output_dir / f".{filename}.tmp"

        payload: dict[str, Any] = {
            "format_version": 2,
            "model_config": self._checkpoint_model_config(),
            # Embed tokenizer artifacts so copying the .pt preserves token IDs.
            "tokenizer": self._tokenizer_snapshot(output_dir),
            "special_token_rows": self._special_token_rows(),
            "model": self._model_state_for_checkpoint(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_parameter_names": self._optimizer_parameter_names(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "trainer_state": asdict(self.state),
            "trainer_config": asdict(self.config),
            "trainable_only": self.config.save_trainable_only,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
        return checkpoint_path

    @staticmethod
    def _read_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "checkpoint-final.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 2:
            raise RuntimeError(
                "Legacy checkpoint lacks reconstruction/tokenizer data and "
                "cannot guarantee restoration of frozen special-token weights."
            )
        return payload

    @staticmethod
    def _load_tokenizer(snapshot: dict[str, Any]) -> Any:
        from transformers import AutoTokenizer

        with TemporaryDirectory(prefix="diffumer-tokenizer-") as directory:
            root = Path(directory).resolve()
            for relative_path, contents in snapshot["files"].items():
                target = (root / relative_path).resolve()
                if not target.is_relative_to(root):
                    raise ValueError("Invalid tokenizer artifact path in checkpoint.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(contents)
            tokenizer = AutoTokenizer.from_pretrained(
                root,
                use_fast=snapshot["is_fast"],
                local_files_only=True,
                trust_remote_code=False,
            )
        tokenizer.name_or_path = snapshot["name_or_path"]
        vocabulary = tokenizer.get_vocab()
        if vocabulary != snapshot["vocabulary"]:
            raise ValueError("Restored tokenizer vocabulary/token IDs differ.")
        if any(
            vocabulary.get(token) != index
            for token, index in snapshot["special_token_mapping"].items()
        ):
            raise ValueError("Restored special-token mapping differs.")
        return tokenizer

    @classmethod
    def _restore_model(cls, model: DiffuMERModel, payload: dict[str, Any]) -> None:
        metadata = payload["model_config"]
        if asdict(model.canvas_builder.config) != metadata["canvas"]:
            raise ValueError("Canvas construction differs from the checkpoint.")
        if _audio_adapter_config(model) != metadata["audio_adapter"]:
            raise ValueError("Audio adapter construction differs from the checkpoint.")
        verifier = metadata["verifier"]
        if verifier["present"] != (model.verifier_heads is not None):
            raise ValueError("Checkpoint and model disagree about verifier presence.")
        if verifier["present"]:
            head_class = (
                f"{type(model.verifier_heads).__module__}."
                f"{type(model.verifier_heads).__qualname__}"
            )
            if head_class != verifier["class"]:
                raise ValueError("Verifier class differs from the checkpoint.")
        if verifier["is_trained"] and not verifier["trained_weights_included"]:
            raise ValueError("Checkpoint marks a trained verifier without its weights.")

        snapshot = payload["tokenizer"]
        input_weight = model.llada_v.get_input_embeddings().weight
        head = model.llada_v.lm_head
        if (
            input_weight.shape[0] != snapshot["input_embedding_rows"]
            or head.weight.shape[0] != snapshot["lm_head_rows"]
        ):
            raise ValueError(
                "Vocabulary table size differs. Use load_model_from_checkpoint() "
                "before constructing an optimizer."
            )
        tokenizer = cls._load_tokenizer(snapshot)
        builder = CanvasBuilder(tokenizer, config=CanvasConfig(**metadata["canvas"]))
        if asdict(builder.token_ids) != snapshot["canvas_token_ids"]:
            raise ValueError("Restored canvas token IDs differ from the checkpoint.")

        expected_keys = set(model.state_dict())
        saved_keys = set(payload["model"])
        unexpected = sorted(saved_keys - expected_keys)
        required_keys = {
            key for key in expected_keys
            if key.startswith(("audio_adapter.", "verifier_heads."))
        }
        if not payload["trainable_only"]:
            required_keys = expected_keys
        missing = sorted(required_keys - saved_keys)
        if unexpected or missing:
            raise RuntimeError(
                f"Checkpoint keys differ: unexpected={unexpected[:10]}, "
                f"missing={missing[:10]}."
            )

        rows = payload["special_token_rows"]
        token_ids = rows["token_ids"]
        row_pairs = (
            (input_weight, rows["input_embeddings"]),
            (head.weight, rows["lm_head"]),
            (head.bias, rows["lm_head_bias"]),
        )
        for parameter, values in row_pairs:
            if (parameter is None) != (values is None):
                raise ValueError("LM head bias differs from the checkpoint.")
            if parameter is not None:
                if tuple(values.shape) != (token_ids.numel(), *parameter.shape[1:]):
                    raise ValueError("Special-token row shape differs from the model.")
                if torch.any((token_ids < 0) | (token_ids >= parameter.shape[0])):
                    raise ValueError("Special-token row index is outside the model.")
        model.load_state_dict(payload["model"], strict=False)
        with torch.no_grad():
            for parameter, values in row_pairs:
                if parameter is not None:
                    parameter.index_copy_(
                        0, token_ids.to(parameter.device),
                        values.to(device=parameter.device, dtype=parameter.dtype),
                    )
        model.tokenizer = tokenizer
        model.canvas_builder = builder
        model.context_length = metadata["context_length"]
        model.max_visual_tokens = metadata["max_visual_tokens"]
        model.max_text_tokens = metadata["max_text_tokens"]
        model.verifier_is_trained = verifier["is_trained"]
        model.model_construction_config = metadata

    @classmethod
    def load_model_from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        loader_kwargs: Mapping[str, Any] | None = None,
        verifier_factory: Callable[[dict[str, Any]], nn.Module] | None = None,
    ) -> DiffuMERModel:
        """Rebuild a model without an optimizer or DataLoader.

        verifier_factory receives saved verifier class/config metadata. Callers
        choose model.eval() and torch.inference_mode() themselves.
        """
        payload = cls._read_checkpoint(checkpoint)
        metadata = payload["model_config"]
        verifier = metadata["verifier"]
        if verifier["present"] and verifier_factory is None:
            raise ValueError("Checkpoint contains a head; provide verifier_factory.")
        loading = dict(metadata["loader_kwargs"])
        loading.update(loader_kwargs or {})
        saved_config = dict(metadata["llada_config"])
        # The base weights still have the ORIGINAL vocabulary table shape.
        # Restore the resized table only after loading those base weights.
        for key in ("vocab_size", "_name_or_path"):
            saved_config.pop(key, None)
        overrides = dict(loading.get("overwrite_config") or {})
        loading["overwrite_config"] = {**saved_config, **overrides}
        model = DiffuMERModel.from_pretrained(
            model_name_or_path=metadata["model_name_or_path"],
            model_base=metadata["model_base"],
            model_name=metadata["model_name"],
            canvas_config=CanvasConfig(**metadata["canvas"]),
            audio_adapter=AudioAdapter(**metadata["audio_adapter"]),
            max_visual_tokens=metadata["max_visual_tokens"],
            max_text_tokens=metadata["max_text_tokens"],
            **loading,
        )
        if verifier["present"]:
            model.verifier_heads = verifier_factory(verifier)
            if not isinstance(model.verifier_heads, nn.Module):
                raise TypeError("verifier_factory must return an nn.Module.")
            weight = model.llada_v.get_input_embeddings().weight
            model.verifier_heads.to(device=weight.device, dtype=weight.dtype)
        input_rows = payload["tokenizer"]["input_embedding_rows"]
        if model.llada_v.get_input_embeddings().weight.shape[0] != input_rows:
            model.llada_v.resize_token_embeddings(input_rows)
        cls._restore_model(model, payload)
        return model

    def load_checkpoint(
        self,
        checkpoint: str | Path,
        *,
        restore_training_state: bool = True,
    ) -> None:
        payload = self._read_checkpoint(checkpoint)
        if restore_training_state:
            if payload["trainer_config"]["train_mode"] != self.config.train_mode:
                raise ValueError(
                    "Training mode changed; use restore_training_state=False "
                    "to start a new stage with a fresh optimizer."
                )
            if payload["optimizer_parameter_names"] != self._optimizer_parameter_names():
                raise ValueError("Optimizer parameter groups differ from the checkpoint.")
        self._restore_model(self.model, payload)
        self.model_config = dict(payload["model_config"])
        self.verifier_config = payload["model_config"]["verifier"]["config"]
        self.verifier_is_trained = self.model.verifier_is_trained
        if hasattr(self.collator, "canvas_builder"):
            self.collator.canvas_builder = self.model.canvas_builder
        if not restore_training_state:
            return

        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.scaler.load_state_dict(payload.get("scaler", {}))
        self.state = TrainerState(**payload["trainer_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        random.setstate(payload["python_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in payload:
            torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
