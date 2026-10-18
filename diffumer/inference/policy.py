"""Pure selection rules for iterative DiffuMER inference.

This module decides *where* a sampler may unmask or remask. It never runs the
model, samples token IDs, mutates a canvas, or updates remask history in place.
All returned token indices are relative to the canvas, excluding any AVT
condition prefix.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diffumer.data.schema import TARGET_BLOCKS


EVIDENCE_BLOCK_NAMES = (
    "visual_evidence",
    "audio_evidence",
    "text_evidence",
)
EVIDENCE_BLOCK_IDS = tuple(
    TARGET_BLOCKS.index(block_name) + 1
    for block_name in EVIDENCE_BLOCK_NAMES
)


@dataclass(frozen=True, slots=True)
class InferencePolicyConfig:
    """Configuration for confidence decoding and bounded semantic revision.

    ``tokens_per_step=None`` divides the remaining masked positions evenly
    over the remaining steps. The last step always receives the full remaining
    budget. During ``evidence_only_steps``, only positions in the three Evidence
    blocks may be unmasked.

    Remasking expects a normalized keep/quality score in ``[0, 1]``: lower
    scores are worse and positions strictly below the threshold are candidates.
    Set the threshold to ``None``, the budget to zero, or the maximum count to
    zero to disable remasking.
    """

    num_steps: int = 32
    tokens_per_step: int | None = None
    evidence_only_steps: int = 4
    evidence_block_ids: tuple[int, ...] = EVIDENCE_BLOCK_IDS
    remask_score_threshold: float | None = 0.35
    remask_budget_per_step: int = 4
    remask_cooldown_steps: int = 2
    max_remasks_per_position: int = 2

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.tokens_per_step is not None and self.tokens_per_step <= 0:
            raise ValueError("tokens_per_step must be positive or None.")
        if not 0 <= self.evidence_only_steps < self.num_steps:
            raise ValueError(
                "evidence_only_steps must lie in [0, num_steps)."
            )
        if not self.evidence_block_ids:
            raise ValueError("evidence_block_ids cannot be empty.")
        if any(block_id <= 0 for block_id in self.evidence_block_ids):
            raise ValueError("Evidence block IDs must be positive.")
        if len(set(self.evidence_block_ids)) != len(self.evidence_block_ids):
            raise ValueError("evidence_block_ids must be unique.")
        if (
            self.remask_score_threshold is not None
            and not 0.0 <= self.remask_score_threshold <= 1.0
        ):
            raise ValueError("remask_score_threshold must lie in [0, 1] or be None.")
        if self.remask_budget_per_step < 0:
            raise ValueError("remask_budget_per_step must be non-negative.")
        if self.remask_cooldown_steps < 0:
            raise ValueError("remask_cooldown_steps must be non-negative.")
        if self.max_remasks_per_position < 0:
            raise ValueError("max_remasks_per_position must be non-negative.")


@dataclass(frozen=True, slots=True)
class PositionSelection:
    """Selected canvas-local positions for one policy decision.

    ``mask`` has shape ``[batch, canvas_length]``. ``indices`` is an ``[N, 2]``
    tensor containing ``(batch_index, canvas_index)`` pairs, and ``counts`` is
    the number selected for each batch item.
    """

    mask: torch.Tensor
    indices: torch.Tensor
    counts: torch.Tensor

    @classmethod
    def from_mask(cls, mask: torch.Tensor) -> "PositionSelection":
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("Selection mask must be a boolean [batch, length] tensor.")
        # Clone so later caller-side edits to an intermediate mask cannot alter
        # a stored trajectory decision.
        result = mask.clone()
        return cls(
            mask=result,
            indices=result.nonzero(as_tuple=False),
            counts=result.sum(dim=1, dtype=torch.long),
        )


class InferencePolicy:
    """Stateless policy for selecting iterative unmask/remask positions."""

    def __init__(self, config: InferencePolicyConfig | None = None) -> None:
        self.config = config or InferencePolicyConfig()

    def _validate_step(self, step: int) -> None:
        if not isinstance(step, int):
            raise TypeError("step must be an integer.")
        if not 0 <= step < self.config.num_steps:
            raise ValueError(
                f"step must lie in [0, {self.config.num_steps}), got {step}."
            )

    @staticmethod
    def _validate_matrix(tensor: torch.Tensor, name: str) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, canvas_length].")

    @classmethod
    def _validate_same_shape(
        cls,
        reference: torch.Tensor,
        tensor: torch.Tensor,
        name: str,
    ) -> None:
        cls._validate_matrix(tensor, name)
        if tensor.shape != reference.shape:
            raise ValueError(f"{name} must have the same shape as the score tensor.")

    def tokens_for_step(
        self,
        masked_counts: torch.Tensor,
        *,
        step: int,
    ) -> torch.Tensor:
        """Return the unmask budget for each sample without changing state."""
        self._validate_step(step)
        if not isinstance(masked_counts, torch.Tensor) or masked_counts.ndim != 1:
            raise ValueError("masked_counts must have shape [batch].")
        if masked_counts.is_floating_point() or masked_counts.is_complex():
            raise TypeError("masked_counts must use an integer dtype.")
        if torch.any(masked_counts < 0):
            raise ValueError("masked_counts cannot be negative.")

        remaining = masked_counts.to(dtype=torch.long)
        if step == self.config.num_steps - 1:
            return remaining.clone()
        if self.config.tokens_per_step is not None:
            fixed = torch.full_like(remaining, self.config.tokens_per_step)
            return torch.minimum(remaining, fixed)

        remaining_steps = self.config.num_steps - step
        # Integer ceil division evenly drains the current number of MASKs.
        return torch.div(
            remaining + remaining_steps - 1,
            remaining_steps,
            rounding_mode="floor",
        )

    def gate_unmask_candidates(
        self,
        candidate_mask: torch.Tensor,
        *,
        step: int,
        block_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the initial Evidence-only gate to a candidate mask."""
        self._validate_step(step)
        self._validate_matrix(candidate_mask, "candidate_mask")
        gated = candidate_mask.to(dtype=torch.bool).clone()
        if step >= self.config.evidence_only_steps:
            return gated
        if block_ids is None:
            raise ValueError("block_ids are required during the Evidence-only gate.")
        if block_ids.ndim == 1:
            if block_ids.shape[0] != gated.shape[1]:
                raise ValueError("One-dimensional block_ids must match canvas length.")
            block_ids = block_ids.unsqueeze(0).expand(gated.shape[0], -1)
        elif block_ids.shape != gated.shape:
            raise ValueError(
                "block_ids must be [canvas_length] or [batch, canvas_length]."
            )
        evidence_ids = torch.tensor(
            self.config.evidence_block_ids,
            device=block_ids.device,
            dtype=block_ids.dtype,
        )
        evidence_mask = torch.isin(block_ids, evidence_ids)
        return gated & evidence_mask.to(device=gated.device)

    @staticmethod
    def _select_by_score(
        scores: torch.Tensor,
        candidates: torch.Tensor,
        budgets: torch.Tensor,
        *,
        largest: bool,
    ) -> PositionSelection:
        selected = torch.zeros_like(candidates, dtype=torch.bool)
        for batch_index in range(scores.shape[0]):
            positions = candidates[batch_index].nonzero(as_tuple=False).flatten()
            count = min(int(budgets[batch_index].item()), positions.numel())
            if count == 0:
                continue
            candidate_scores = scores[batch_index].index_select(0, positions)
            chosen = torch.topk(
                candidate_scores,
                k=count,
                largest=largest,
                sorted=False,
            ).indices
            selected[batch_index, positions.index_select(0, chosen)] = True
        return PositionSelection.from_mask(selected)

    def select_unmask(
        self,
        confidences: torch.Tensor,
        masked_positions: torch.Tensor,
        *,
        step: int,
        block_ids: torch.Tensor | None = None,
    ) -> PositionSelection:
        """Select the highest-confidence currently masked canvas positions.

        ``masked_positions`` must already exclude fixed boundaries and invalid
        tail positions. Consequently, this method cannot accidentally select
        positions outside the mutable canvas content.
        """
        self._validate_step(step)
        self._validate_matrix(confidences, "confidences")
        if not confidences.is_floating_point():
            raise TypeError("confidences must use a floating-point dtype.")
        self._validate_same_shape(confidences, masked_positions, "masked_positions")
        masked = masked_positions.to(device=confidences.device, dtype=torch.bool)
        if torch.any(~torch.isfinite(confidences[masked])):
            raise ValueError("Masked positions contain non-finite confidence values.")

        candidates = self.gate_unmask_candidates(
            masked,
            step=step,
            block_ids=block_ids,
        )
        budgets = self.tokens_for_step(masked.sum(dim=1), step=step)
        return self._select_by_score(
            confidences,
            candidates,
            budgets,
            largest=True,
        )

    def select_remask(
        self,
        keep_scores: torch.Tensor,
        visible_positions: torch.Tensor,
        *,
        step: int,
        remask_counts: torch.Tensor,
        last_remask_steps: torch.Tensor,
        protected_positions: torch.Tensor | None = None,
    ) -> PositionSelection:
        """Select low-scoring visible positions within all revision limits.

        The caller owns ``remask_counts`` and ``last_remask_steps`` and updates
        them after accepting this result. A negative last-remask step means the
        position has never been remasked. Inputs are only read and never edited.
        """
        self._validate_step(step)
        self._validate_matrix(keep_scores, "keep_scores")
        if not keep_scores.is_floating_point():
            raise TypeError("keep_scores must use a floating-point dtype.")
        for tensor, name in (
            (visible_positions, "visible_positions"),
            (remask_counts, "remask_counts"),
            (last_remask_steps, "last_remask_steps"),
        ):
            self._validate_same_shape(keep_scores, tensor, name)
        if remask_counts.is_floating_point() or remask_counts.is_complex():
            raise TypeError("remask_counts must use an integer dtype.")
        if last_remask_steps.is_floating_point() or last_remask_steps.is_complex():
            raise TypeError("last_remask_steps must use an integer dtype.")
        if torch.any(remask_counts < 0):
            raise ValueError("remask_counts cannot be negative.")
        if torch.any(last_remask_steps > step):
            raise ValueError("last_remask_steps cannot refer to a future step.")
        if protected_positions is not None:
            self._validate_same_shape(
                keep_scores,
                protected_positions,
                "protected_positions",
            )

        disabled = (
            self.config.remask_score_threshold is None
            or self.config.remask_budget_per_step == 0
            or self.config.max_remasks_per_position == 0
        )
        if disabled:
            return PositionSelection.from_mask(
                torch.zeros_like(keep_scores, dtype=torch.bool)
            )

        visible = visible_positions.to(device=keep_scores.device, dtype=torch.bool)
        counts = remask_counts.to(device=keep_scores.device)
        last_steps = last_remask_steps.to(device=keep_scores.device)
        never_remasked = last_steps < 0
        cooldown_elapsed = (step - last_steps) > self.config.remask_cooldown_steps
        candidates = (
            visible
            & torch.isfinite(keep_scores)
            & (keep_scores < self.config.remask_score_threshold)
            & (counts < self.config.max_remasks_per_position)
            & (never_remasked | cooldown_elapsed)
        )
        if protected_positions is not None:
            protected = protected_positions.to(
                device=keep_scores.device,
                dtype=torch.bool,
            )
            candidates &= ~protected

        budgets = torch.full(
            (keep_scores.shape[0],),
            self.config.remask_budget_per_step,
            dtype=torch.long,
            device=keep_scores.device,
        )
        return self._select_by_score(
            keep_scores,
            candidates,
            budgets,
            largest=False,
        )
