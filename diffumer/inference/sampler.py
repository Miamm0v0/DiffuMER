"""Iterative masked-diffusion sampling for DiffuMER."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from diffumer.data.schema import TARGET_BLOCKS
from diffumer.model.canvas import CanvasBatch, CanvasBuilder
from diffumer.model.diffumer_model import ConditionBatch, DiffuMERModel

from .policy import InferencePolicy, PositionSelection


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Execution settings that are separate from position-selection policy."""

    temperature: float = 0.0
    use_verifier: bool = False
    verifier_start_step: int | None = None
    verifier_every_steps: int = 1
    quality_scores_are_logits: bool = True
    quality_score_key: str = "quality"
    extra_forbidden_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative.")
        if self.verifier_start_step is not None and self.verifier_start_step < 0:
            raise ValueError("verifier_start_step must be non-negative or None.")
        if self.verifier_every_steps <= 0:
            raise ValueError("verifier_every_steps must be positive.")
        if not self.quality_score_key:
            raise ValueError("quality_score_key cannot be empty.")
        if any(token_id < 0 for token_id in self.extra_forbidden_token_ids):
            raise ValueError("extra_forbidden_token_ids cannot contain negatives.")
        if len(set(self.extra_forbidden_token_ids)) != len(
            self.extra_forbidden_token_ids
        ):
            raise ValueError("extra_forbidden_token_ids must be unique.")


@dataclass(slots=True)
class SamplerState:
    """Canvas and revision history required to resume at ``next_step``."""

    canvas: CanvasBatch
    remask_counts: torch.Tensor
    last_remask_steps: torch.Tensor
    next_step: int = 0

    @classmethod
    def from_canvas(cls, canvas: CanvasBatch) -> "SamplerState":
        shape = canvas.input_ids.shape
        device = canvas.input_ids.device
        return cls(
            canvas=canvas.clone(),
            remask_counts=torch.zeros(shape, dtype=torch.long, device=device),
            last_remask_steps=torch.full(
                shape,
                -1,
                dtype=torch.long,
                device=device,
            ),
            next_step=0,
        )


@dataclass(frozen=True, slots=True)
class StepTrace:
    """Independent CPU snapshot of one diffusion update."""

    step: int
    unmasked: PositionSelection
    remasked: PositionSelection
    eos_tail_mask: torch.Tensor
    masked_before: torch.Tensor
    masked_after: torch.Tensor
    verification_scores: torch.Tensor | dict[str, torch.Tensor] | None
    canvas: CanvasBatch
    finished: torch.Tensor
    stop_reason: str | None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Final canvas, decoded E-R-Y blocks, and the complete sampling trace."""

    canvas: CanvasBatch
    decoded: list[dict[str, str]]
    traces: list[StepTrace]
    stop_reason: str


def _masked_positions(canvas: CanvasBatch, mask_token_id: int) -> torch.Tensor:
    return canvas.output_mask.to(dtype=torch.bool) & canvas.input_ids.eq(mask_token_id)


def _empty_selection(reference: torch.Tensor) -> PositionSelection:
    return PositionSelection.from_mask(
        torch.zeros_like(reference, dtype=torch.bool)
    )


def _selection_to_cpu(selection: PositionSelection) -> PositionSelection:
    return PositionSelection.from_mask(selection.mask.detach().cpu())


def _scores_to_cpu(
    scores: torch.Tensor | Mapping[str, torch.Tensor] | None,
) -> torch.Tensor | dict[str, torch.Tensor] | None:
    if scores is None:
        return None
    if isinstance(scores, torch.Tensor):
        return scores.detach().cpu().clone()
    if not isinstance(scores, Mapping):
        raise TypeError("verification_scores must be a tensor or tensor mapping.")
    snapshot: dict[str, torch.Tensor] = {}
    for name, value in scores.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError("verification_scores mapping must contain tensor values.")
        snapshot[name] = value.detach().cpu().clone()
    return snapshot


def _validate_state(state: SamplerState, policy: InferencePolicy) -> None:
    if state.canvas.input_ids.ndim != 2:
        raise ValueError("Sampler canvas must have shape [batch, canvas_length].")
    shape = state.canvas.input_ids.shape
    for tensor, name in (
        (state.remask_counts, "remask_counts"),
        (state.last_remask_steps, "last_remask_steps"),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} must match the canvas shape.")
        if tensor.is_floating_point() or tensor.is_complex():
            raise TypeError(f"{name} must use an integer dtype.")
    if torch.any(state.remask_counts < 0):
        raise ValueError("remask_counts cannot be negative.")
    if state.next_step < 0 or state.next_step >= policy.config.num_steps:
        raise ValueError("Sampler state has no valid next diffusion step.")


def _forbidden_token_ids(
    builder: CanvasBuilder,
    config: SamplerConfig,
    vocabulary_size: int,
) -> list[int]:
    token_ids = builder.token_ids
    forbidden = {
        token_ids.mask,
        token_ids.canvas_start,
        token_ids.canvas_end,
        *token_ids.condition_start.values(),
        *token_ids.condition_end.values(),
        *token_ids.block_start.values(),
        *token_ids.block_end.values(),
        *config.extra_forbidden_token_ids,
    }
    # PAD inside content is ambiguous; EOS is the only generated terminator.
    if token_ids.pad != token_ids.eos:
        forbidden.add(token_ids.pad)
    invalid = sorted(
        token_id
        for token_id in forbidden
        if token_id < 0 or token_id >= vocabulary_size
    )
    if invalid:
        raise ValueError(f"Forbidden token IDs are outside the vocabulary: {invalid}")
    if len(forbidden) >= vocabulary_size:
        raise ValueError("All vocabulary entries are forbidden for sampling.")
    return sorted(forbidden)


def _predict_masked_tokens(
    logits: torch.Tensor,
    masked_positions: torch.Tensor,
    *,
    builder: CanvasBuilder,
    config: SamplerConfig,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or logits.shape[:2] != masked_positions.shape:
        raise ValueError("Model logits must align with the canvas.")
    masked = masked_positions.to(device=logits.device, dtype=torch.bool)
    predicted_ids = torch.full(
        masked.shape,
        builder.token_ids.mask,
        dtype=torch.long,
        device=logits.device,
    )
    confidences = torch.zeros(
        masked.shape,
        dtype=torch.float32,
        device=logits.device,
    )
    if not torch.any(masked):
        return predicted_ids, confidences

    candidate_logits = logits[masked].float().clone()
    if torch.any(~torch.isfinite(candidate_logits)):
        raise ValueError("Model produced non-finite logits at masked positions.")
    forbidden = _forbidden_token_ids(
        builder,
        config,
        candidate_logits.shape[-1],
    )
    candidate_logits[:, forbidden] = -torch.inf

    if config.temperature == 0.0:
        chosen_logits, chosen_ids = candidate_logits.max(dim=-1)
        normalizers = torch.logsumexp(candidate_logits, dim=-1)
        chosen_confidences = torch.exp(chosen_logits - normalizers)
    else:
        probabilities = torch.softmax(
            candidate_logits / config.temperature,
            dim=-1,
        )
        chosen_ids = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        ).squeeze(-1)
        chosen_confidences = probabilities.gather(
            1,
            chosen_ids.unsqueeze(1),
        ).squeeze(1)

    predicted_ids[masked] = chosen_ids
    confidences[masked] = chosen_confidences
    return predicted_ids, confidences


def _apply_eos_tails(
    canvas: CanvasBatch,
    builder: CanvasBuilder,
) -> torch.Tensor:
    """Replace positions after the first EOS in each block with inactive PAD."""
    tail_mask = torch.zeros_like(canvas.input_ids, dtype=torch.bool)
    eos_id = builder.token_ids.eos
    for batch_index in range(canvas.input_ids.shape[0]):
        for block_name in TARGET_BLOCKS:
            block_slice = builder.layout.block_slices[block_name]
            content = canvas.input_ids[batch_index, block_slice]
            eos_offsets = content.eq(eos_id).nonzero(as_tuple=False).flatten()
            if eos_offsets.numel() == 0:
                continue
            first_eos = block_slice.start + int(eos_offsets[0].item())
            if first_eos + 1 < block_slice.stop:
                tail_mask[batch_index, first_eos + 1:block_slice.stop] = True

    if torch.any(tail_mask):
        canvas.input_ids[tail_mask] = builder.token_ids.pad
        canvas.attention_mask[tail_mask] = False
        canvas.labels[tail_mask] = -100
        canvas.supervision_mask[tail_mask] = False
    return tail_mask


def _verification_is_due(
    step_index: int,
    policy: InferencePolicy,
    config: SamplerConfig,
) -> bool:
    if not config.use_verifier:
        return False
    start = (
        policy.config.evidence_only_steps
        if config.verifier_start_step is None
        else config.verifier_start_step
    )
    if start >= policy.config.num_steps:
        raise ValueError("verifier_start_step must be smaller than num_steps.")
    return (
        step_index >= start
        and (step_index - start) % config.verifier_every_steps == 0
    )


def _remasking_enabled(policy: InferencePolicy) -> bool:
    return (
        policy.config.remask_score_threshold is not None
        and policy.config.remask_budget_per_step > 0
        and policy.config.max_remasks_per_position > 0
    )


def _quality_keep_scores(
    scores: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    config: SamplerConfig,
    canvas_shape: torch.Size,
) -> torch.Tensor:
    if isinstance(scores, Mapping):
        if config.quality_score_key not in scores:
            raise KeyError(
                "Verifier remasking requires a canvas-aligned quality score "
                f"named {config.quality_score_key!r}."
            )
        quality = scores[config.quality_score_key]
    else:
        quality = scores
    if not isinstance(quality, torch.Tensor):
        raise TypeError("Verifier quality scores must be a tensor.")
    if quality.ndim == 3 and quality.shape[-1] == 1:
        quality = quality.squeeze(-1)
    if quality.shape != canvas_shape:
        raise ValueError(
            "Verifier quality scores must have shape [batch, canvas_length]."
        )
    quality = quality.float()
    if config.quality_scores_are_logits:
        return quality.sigmoid()
    if torch.any(~torch.isfinite(quality)) or torch.any((quality < 0) | (quality > 1)):
        raise ValueError("Verifier keep scores must be finite probabilities in [0, 1].")
    return quality


def _visible_quality_positions(
    canvas: CanvasBatch,
    builder: CanvasBuilder,
) -> torch.Tensor:
    token_ids = canvas.input_ids
    return (
        canvas.output_mask.to(dtype=torch.bool)
        & canvas.attention_mask.to(dtype=torch.bool)
        & token_ids.ne(builder.token_ids.mask)
        & token_ids.ne(builder.token_ids.eos)
        & token_ids.ne(builder.token_ids.pad)
    )


def _step_impl(
    model: DiffuMERModel,
    conditions: ConditionBatch,
    state: SamplerState,
    *,
    policy: InferencePolicy,
    config: SamplerConfig,
    generator: torch.Generator | None,
) -> tuple[SamplerState, StepTrace]:
    _validate_state(state, policy)
    step_index = state.next_step
    builder = model.canvas_builder
    model_device = model.llada_v.get_input_embeddings().weight.device
    canvas = state.canvas.to(model_device).clone()
    remask_counts = state.remask_counts.to(model_device).clone()
    last_remask_steps = state.last_remask_steps.to(model_device).clone()

    masked_before_mask = _masked_positions(canvas, builder.token_ids.mask)
    if not torch.any(masked_before_mask):
        raise RuntimeError("The canvas is already complete; no diffusion step is needed.")
    masked_before = masked_before_mask.sum(dim=1, dtype=torch.long)

    output = model.forward_canvas(conditions, canvas)
    predicted_ids, confidences = _predict_masked_tokens(
        output.logits,
        masked_before_mask,
        builder=builder,
        config=config,
        generator=generator,
    )
    unmasked = policy.select_unmask(
        confidences,
        masked_before_mask,
        step=step_index,
        block_ids=canvas.block_ids,
    )
    if torch.any(unmasked.mask):
        canvas.input_ids[unmasked.mask] = predicted_ids[unmasked.mask]
        canvas.attention_mask[unmasked.mask] = True
    eos_tail_mask = _apply_eos_tails(canvas, builder)

    verification_scores: torch.Tensor | Mapping[str, torch.Tensor] | None = None
    remasked = _empty_selection(canvas.input_ids)
    final_step = step_index == policy.config.num_steps - 1
    if _verification_is_due(step_index, policy, config):
        verified = model.verify_canvas(conditions, canvas)
        verification_scores = verified.verification_scores
        if verification_scores is None:
            raise RuntimeError("Verifier returned no scores.")
        # Never create fresh MASKs after the final forward pass.
        if not final_step and _remasking_enabled(policy):
            keep_scores = _quality_keep_scores(
                verification_scores,
                config=config,
                canvas_shape=canvas.input_ids.shape,
            )
            remasked = policy.select_remask(
                keep_scores,
                _visible_quality_positions(canvas, builder),
                step=step_index,
                remask_counts=remask_counts,
                last_remask_steps=last_remask_steps,
                protected_positions=eos_tail_mask,
            )
            if torch.any(remasked.mask):
                canvas.input_ids[remasked.mask] = builder.token_ids.mask
                canvas.attention_mask[remasked.mask] = True
                canvas.labels[remasked.mask] = -100
                canvas.supervision_mask[remasked.mask] = False
                remask_counts[remasked.mask] += 1
                last_remask_steps[remasked.mask] = step_index

    masked_after_mask = _masked_positions(canvas, builder.token_ids.mask)
    masked_after = masked_after_mask.sum(dim=1, dtype=torch.long)
    finished = masked_after.eq(0)
    stop_reason = None
    if bool(finished.all().item()):
        stop_reason = "complete"
    elif final_step:
        stop_reason = "max_steps"

    next_state = SamplerState(
        canvas=canvas,
        remask_counts=remask_counts,
        last_remask_steps=last_remask_steps,
        next_step=step_index + 1,
    )
    trace = StepTrace(
        step=step_index,
        unmasked=_selection_to_cpu(unmasked),
        remasked=_selection_to_cpu(remasked),
        eos_tail_mask=eos_tail_mask.detach().cpu().clone(),
        masked_before=masked_before.detach().cpu().clone(),
        masked_after=masked_after.detach().cpu().clone(),
        verification_scores=_scores_to_cpu(verification_scores),
        canvas=canvas.to("cpu").clone(),
        finished=finished.detach().cpu().clone(),
        stop_reason=stop_reason,
    )
    return next_state, trace


@torch.inference_mode()
def step(
    model: DiffuMERModel,
    conditions: ConditionBatch,
    state: SamplerState,
    *,
    policy: InferencePolicy | None = None,
    config: SamplerConfig | None = None,
    generator: torch.Generator | None = None,
) -> tuple[SamplerState, StepTrace]:
    """Execute one diffusion update using already cached AVT conditions."""
    active_policy = policy or InferencePolicy()
    active_config = config or SamplerConfig()
    was_training = model.training
    model.eval()
    try:
        return _step_impl(
            model,
            conditions,
            state,
            policy=active_policy,
            config=active_config,
            generator=generator,
        )
    finally:
        model.train(was_training)


@torch.inference_mode()
def generate(
    model: DiffuMERModel,
    *,
    video_frames: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
    audio_features: torch.Tensor | None = None,
    audio_attention_mask: torch.Tensor | None = None,
    text_input_ids: torch.Tensor | None = None,
    text_attention_mask: torch.Tensor | None = None,
    transcripts: Sequence[str] | None = None,
    batch_size: int | None = None,
    initial_canvas: CanvasBatch | None = None,
    policy: InferencePolicy | None = None,
    config: SamplerConfig | None = None,
    generator: torch.Generator | None = None,
) -> GenerationResult:
    """Run the complete diffusion loop with one cached AVT encoding.

    If ``initial_canvas`` is omitted, a fully masked canvas is created. A
    supplied canvas may resume a partially visible draft, but remask history
    starts empty; use :func:`step` with an existing :class:`SamplerState` for
    exact mid-trajectory continuation.
    """
    active_policy = policy or InferencePolicy()
    active_config = config or SamplerConfig()
    if initial_canvas is not None:
        canvas_batch_size = initial_canvas.input_ids.shape[0]
        if batch_size is not None and batch_size != canvas_batch_size:
            raise ValueError("batch_size differs from initial_canvas batch size.")
        batch_size = canvas_batch_size

    was_training = model.training
    model.eval()
    try:
        # Cache only the pre-backbone condition embeddings. Canvas-dependent
        # backbone hidden states are deliberately recomputed in every step.
        conditions = model.encode_conditions(
            video_frames=video_frames,
            audio_features=audio_features,
            audio_attention_mask=audio_attention_mask,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            transcripts=transcripts,
            batch_size=batch_size,
        )
        resolved_batch_size = conditions.embeddings.shape[0]
        device = model.llada_v.get_input_embeddings().weight.device
        if initial_canvas is None:
            canvas = model.build_generation_canvas(
                resolved_batch_size,
                device=device,
            )
        else:
            if initial_canvas.input_ids.shape[0] != resolved_batch_size:
                raise ValueError(
                    "Condition batch size differs from initial_canvas batch size."
                )
            canvas = initial_canvas.to(device).clone()
        _apply_eos_tails(canvas, model.canvas_builder)
        state = SamplerState.from_canvas(canvas)
        traces: list[StepTrace] = []

        initial_masked = _masked_positions(
            state.canvas,
            model.canvas_builder.token_ids.mask,
        )
        if not torch.any(initial_masked):
            stop_reason = "complete"
        else:
            stop_reason = "max_steps"
            while state.next_step < active_policy.config.num_steps:
                state, trace = step(
                    model,
                    conditions,
                    state,
                    policy=active_policy,
                    config=active_config,
                    generator=generator,
                )
                traces.append(trace)
                if trace.stop_reason is not None:
                    stop_reason = trace.stop_reason
                    break

        decoded = model.canvas_builder.decode_blocks(state.canvas.input_ids)
        return GenerationResult(
            canvas=state.canvas,
            decoded=decoded,
            traces=traces,
            stop_reason=stop_reason,
        )
    finally:
        model.train(was_training)
