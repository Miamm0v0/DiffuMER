"""Training-time corruption for DiffuMER's structured diffusion canvas.

This module implements the forward masking process used to construct
training states S_t from clean target canvases S_0.

Inference-time confidence decoding, consistency checking, and selective
re-masking should live in the future diffusion sampler rather than here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffumer.model.canvas import CanvasBatch


@dataclass(frozen=True, slots=True)
class MaskingConfig:
    """Configuration for continuous-time masked diffusion corruption.

    The default schedule follows LLaDA-V training:

        t ~ Uniform(0, 1)
        p_mask(t) = eps + (1 - eps) * t

    Only supervised output positions are eligible for masking.
    """

    eps: float = 1e-3
    ensure_at_least_one_mask: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.eps < 1.0:
            raise ValueError(f"eps must be in [0, 1), got {self.eps}.")


@dataclass(slots=True)
class MaskingResult:
    """Noisy canvas together with diffusion metadata used by the loss."""

    canvas: CanvasBatch
    timesteps: torch.Tensor
    mask_probabilities: torch.Tensor
    corruption_mask: torch.Tensor


def linear_mask_probability(
    timesteps: torch.Tensor,
    *,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Map continuous diffusion times in [0, 1] to masking probabilities."""
    if not 0.0 <= eps < 1.0:
        raise ValueError(f"eps must be in [0, 1), got {eps}.")
    if torch.any((timesteps < 0) | (timesteps > 1)):
        raise ValueError("timesteps must lie in [0, 1].")
    return eps + (1.0 - eps) * timesteps


def sample_timesteps(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one continuous diffusion time independently for each sample."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return torch.rand(
        batch_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )


def _clone_canvas(canvas: CanvasBatch) -> CanvasBatch:
    return CanvasBatch(
        input_ids=canvas.input_ids.clone(),
        attention_mask=canvas.attention_mask.clone(),
        labels=canvas.labels.clone(),
        output_mask=canvas.output_mask.clone(),
        supervision_mask=canvas.supervision_mask.clone(),
        block_ids=canvas.block_ids.clone(),
    )


def _normalize_timesteps(
    timesteps: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    timesteps = torch.as_tensor(
        timesteps,
        dtype=torch.float32,
        device=device,
    )
    if timesteps.ndim == 0:
        timesteps = timesteps.expand(batch_size)
    if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
        raise ValueError(
            "timesteps must be scalar or have shape [batch], got "
            f"{tuple(timesteps.shape)} for batch={batch_size}."
        )
    if torch.any((timesteps < 0) | (timesteps > 1)):
        raise ValueError("timesteps must lie in [0, 1].")
    return timesteps


def _ensure_one_mask_per_supervised_sample(
    corruption_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> None:
    """Ensure every supervised sample contributes at least one masked token."""
    needs_mask = candidate_mask.any(dim=1) & ~corruption_mask.any(dim=1)

    for batch_index in torch.nonzero(
        needs_mask,
        as_tuple=False,
    ).flatten().tolist():
        candidates = torch.nonzero(
            candidate_mask[batch_index],
            as_tuple=False,
        ).flatten()

        choice = torch.randint(
            candidates.numel(),
            (1,),
            device=candidates.device,
            generator=generator,
        )
        corruption_mask[batch_index, candidates[choice]] = True


def corrupt_canvas(
    clean_canvas: CanvasBatch,
    *,
    mask_token_id: int,
    config: MaskingConfig | None = None,
    timesteps: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> MaskingResult:
    """Construct S_t by replacing supervised target tokens with MDM masks.

    Boundary tokens, unsupervised blocks, and unused canvas slots are never
    corrupted. ``labels`` retain clean S_0 targets so the training loss can be
    computed only on ``corruption_mask`` positions.

    Args:
        clean_canvas:
            Clean target canvas from ``CanvasBuilder.encode_targets``.
        mask_token_id:
            Token id corresponding to ``<|mdm_mask|>``.
        config:
            Masking schedule configuration.
        timesteps:
            Optional scalar or ``[batch]`` tensor. If omitted, one continuous
            diffusion time is sampled independently for each sample.
        generator:
            Optional PyTorch random generator.

    Returns:
        ``MaskingResult`` containing the noisy canvas and diffusion metadata.
    """
    config = config or MaskingConfig()

    if clean_canvas.input_ids.ndim != 2:
        raise ValueError(
            "clean_canvas.input_ids must have shape [batch, length], got "
            f"{tuple(clean_canvas.input_ids.shape)}."
        )

    reference_shape = clean_canvas.input_ids.shape

    for name, tensor in (
        ("attention_mask", clean_canvas.attention_mask),
        ("labels", clean_canvas.labels),
        ("output_mask", clean_canvas.output_mask),
        ("supervision_mask", clean_canvas.supervision_mask),
        ("block_ids", clean_canvas.block_ids),
    ):
        if tensor.shape != reference_shape:
            raise ValueError(
                f"clean_canvas.{name} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(reference_shape)}."
            )

    batch_size = reference_shape[0]
    device = clean_canvas.input_ids.device

    if timesteps is None:
        timesteps = sample_timesteps(
            batch_size,
            device=device,
            generator=generator,
        )
    else:
        timesteps = _normalize_timesteps(
            timesteps,
            batch_size=batch_size,
            device=device,
        )

    mask_probabilities = linear_mask_probability(
        timesteps,
        eps=config.eps,
    )

    # Only clean target tokens with trusted supervision are corruptible.
    # This is important because current MER-Caption+ data supervises R and Y
    # but leaves E_v/E_a/E_t unannotated.
    candidate_mask = (
        clean_canvas.output_mask.bool()
        & clean_canvas.supervision_mask.bool()
        & clean_canvas.attention_mask.bool()
        & clean_canvas.labels.ne(-100)
    )

    random_values = torch.rand(
        reference_shape,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )

    corruption_mask = (
        random_values < mask_probabilities.unsqueeze(1)
    ) & candidate_mask

    if config.ensure_at_least_one_mask:
        _ensure_one_mask_per_supervised_sample(
            corruption_mask,
            candidate_mask,
            generator=generator,
        )

    noisy_canvas = _clone_canvas(clean_canvas)
    noisy_canvas.input_ids[corruption_mask] = int(mask_token_id)

    return MaskingResult(
        canvas=noisy_canvas,
        timesteps=timesteps,
        mask_probabilities=mask_probabilities,
        corruption_mask=corruption_mask,
    )
