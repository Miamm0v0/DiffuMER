"""Training objectives for masked diffusion over the DiffuMER canvas."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


def _expand_mask_probabilities(
    mask_probabilities: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    probabilities = torch.as_tensor(
        mask_probabilities,
        dtype=torch.float32,
        device=device,
    )
    if probabilities.ndim == 1:
        if probabilities.shape[0] != batch_size:
            raise ValueError(
                "mask_probabilities must have shape [batch] or [batch, length]; "
                f"got {tuple(probabilities.shape)}."
            )
        probabilities = probabilities.unsqueeze(1).expand(
            batch_size,
            sequence_length,
        )
    elif tuple(probabilities.shape) != (batch_size, sequence_length):
        raise ValueError(
            "mask_probabilities must have shape [batch] or [batch, length]; "
            f"got {tuple(probabilities.shape)}."
        )
    return probabilities


def masked_diffusion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    corruption_mask: torch.Tensor,
    mask_probabilities: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the LLaDA-style masked-token diffusion objective.

    Cross entropy is evaluated strictly at positions selected by
    ``corruption_mask``.  Each selected token is importance weighted by
    ``1 / p_mask`` and normalized by the number of supervised tokens in its
    sample, matching LLaDA's continuous-time masked diffusion objective.

    The importance weight is kept token-shaped internally so block-wise loss
    factors can be multiplied into it later without changing the reduction.
    """
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [batch, length, vocab], got "
            f"{tuple(logits.shape)}."
        )

    batch_size, sequence_length, _ = logits.shape
    expected_shape = (batch_size, sequence_length)
    if tuple(labels.shape) != expected_shape:
        raise ValueError(
            f"labels must have shape {expected_shape}, got {tuple(labels.shape)}."
        )
    if tuple(corruption_mask.shape) != expected_shape:
        raise ValueError(
            "corruption_mask must match [batch, length]; expected "
            f"{expected_shape}, got {tuple(corruption_mask.shape)}."
        )
    if batch_size == 0:
        raise ValueError("masked_diffusion_loss received an empty batch.")

    labels = labels.to(device=logits.device, dtype=torch.long)
    corruption_mask = corruption_mask.to(device=logits.device, dtype=torch.bool)
    if torch.any(corruption_mask & labels.eq(ignore_index)):
        raise ValueError(
            "corruption_mask selects positions whose labels equal ignore_index."
        )
    if not torch.any(corruption_mask):
        raise ValueError("corruption_mask contains no selected tokens.")

    probabilities = _expand_mask_probabilities(
        mask_probabilities,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=logits.device,
    )
    selected_probabilities = probabilities[corruption_mask]
    if torch.any((selected_probabilities <= 0) | (selected_probabilities > 1)):
        raise ValueError("Selected mask probabilities must lie in (0, 1].")

    selected_logits = logits[corruption_mask]
    selected_labels = labels[corruption_mask]
    token_losses = F.cross_entropy(
        selected_logits,
        selected_labels,
        reduction="none",
    )

    supervised_counts = labels.ne(ignore_index).sum(dim=1)
    selected_batch_indices = torch.nonzero(
        corruption_mask,
        as_tuple=False,
    )[:, 0]
    sample_denominators = supervised_counts[selected_batch_indices]
    if torch.any(sample_denominators == 0):
        raise ValueError("A corrupted sample contains no supervised tokens.")

    importance_weights = selected_probabilities.reciprocal()
    weighted_losses = (
        token_losses
        * importance_weights
        / sample_denominators.to(token_losses.dtype)
    )
    return weighted_losses.sum() / batch_size


def verifier_loss(
    predictions: Mapping[str, torch.Tensor | None],
    targets: Mapping[str, torch.Tensor | None],
    valid_masks: Mapping[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    """Return quality, grounding, adequacy, and relation losses independently.

    Predictions come from semantic verifier heads, not the language-model
    vocabulary logits. Quality uses raw binary logits shaped like its targets
    (an extra final singleton dimension is accepted) and labels in [0, 1]:
    1 means retain a visible token, 0 means discard it. Other logits have shape
    [*target_shape, 3], with integer class IDs in the following order:

        grounding: 0 supported, 1 contradicted, 2 undetermined
        adequacy: 0 adequate, 1 inadequate, 2 undetermined
        relation: 0 compatible, 1 conflicting, 2 undetermined

    Each annotated task requires its own valid mask with the target shape.
    Quality masks must exclude MASK, fixed boundary tokens, and post-EOS/PAD
    tails. Span/relation masks must exclude unfinished or invalid spans.
    Callers supply these masks and independent semantic annotations; no label
    is inferred from ordinary token logits or equality with gold wording.

    Missing target keys, None, and -100 labels represent missing annotations.
    Undetermined (class 2) is supervised normally. Missing/empty supervision
    returns a differentiable zero on that head's device, without evaluating
    ignored score values. A missing head uses the first available prediction
    tensor as its device reference; at least one such tensor is required.
    """
    reference = next(
        (value for value in predictions.values() if isinstance(value, torch.Tensor)),
        None,
    )
    if reference is None:
        raise ValueError("predictions must contain at least one verifier tensor.")
    if not reference.is_floating_point():
        raise TypeError("Verifier predictions must be floating-point logits.")

    losses: dict[str, torch.Tensor] = {}
    for name in ("quality", "grounding", "adequacy", "relation"):
        prediction = predictions.get(name)
        if prediction is not None and (
            not isinstance(prediction, torch.Tensor)
            or not prediction.is_floating_point()
        ):
            raise TypeError(f"{name} predictions must be floating-point logits.")
        anchor = prediction if prediction is not None else reference
        # An empty sum is zero even when ignored predictions contain NaN/Inf.
        # prediction.sum() * 0 would contaminate the result in that case.
        losses[name] = anchor.reshape(-1)[:0].float().sum()

        target = targets.get(name)
        if target is None:
            continue
        valid_mask = valid_masks.get(name)
        if valid_mask is None:
            raise ValueError(f"{name} annotations require a separate valid mask.")
        target = torch.as_tensor(target, device=anchor.device)
        valid_mask = torch.as_tensor(
            valid_mask,
            device=anchor.device,
            dtype=torch.bool,
        )
        if tuple(valid_mask.shape) != tuple(target.shape):
            raise ValueError(f"{name} valid mask must have the same shape as targets.")
        valid_mask = valid_mask & target.ne(-100)
        if not torch.any(valid_mask):
            continue
        if prediction is None:
            raise ValueError(f"{name} has valid supervision but no predictions.")

        if name == "quality":
            if tuple(prediction.shape) == (*target.shape, 1):
                prediction = prediction.squeeze(-1)
            if tuple(prediction.shape) != tuple(target.shape):
                raise ValueError("quality logits must have the same shape as targets.")
            selected_targets = target[valid_mask].float()
            if torch.any(
                ~torch.isfinite(selected_targets)
                | (selected_targets < 0)
                | (selected_targets > 1)
            ):
                raise ValueError("Valid quality targets must lie in [0, 1].")
            losses[name] = F.binary_cross_entropy_with_logits(
                prediction[valid_mask].float(),
                selected_targets,
            )
        else:
            if tuple(prediction.shape) != (*target.shape, 3):
                raise ValueError(f"{name} logits must have shape [*target_shape, 3].")
            selected_targets = target[valid_mask]
            if selected_targets.is_floating_point() or selected_targets.is_complex():
                raise TypeError(f"{name} targets must contain integer class IDs.")
            if torch.any((selected_targets < 0) | (selected_targets > 2)):
                raise ValueError(f"Valid {name} targets must lie in {{0, 1, 2}}.")
            losses[name] = F.cross_entropy(
                prediction[valid_mask].float(),
                selected_targets.to(torch.long),
            )

    return losses
