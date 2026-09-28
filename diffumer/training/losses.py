"""Training objectives for masked diffusion over the DiffuMER canvas."""

from __future__ import annotations

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
