"""Structured E-R-Y output canvas and its special-token protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from diffumer.data.schema import EMERTarget, TARGET_BLOCKS


MASK_TOKEN = "<|mdm_mask|>"
CANVAS_START_TOKEN = "<|diffumer_canvas_start|>"
CANVAS_END_TOKEN = "<|diffumer_canvas_end|>"

CONDITION_BOUNDARY_TOKENS: dict[str, tuple[str, str]] = {
    "visual": ("<|visual_start|>", "<|visual_end|>"),
    "audio": ("<|audio_start|>", "<|audio_end|>"),
    "text": ("<|text_start|>", "<|text_end|>"),
}

BLOCK_BOUNDARY_TOKENS: dict[str, tuple[str, str]] = {
    "visual_evidence": ("<|visual_evidence_start|>", "<|visual_evidence_end|>"),
    "audio_evidence": ("<|audio_evidence_start|>", "<|audio_evidence_end|>"),
    "text_evidence": ("<|text_evidence_start|>", "<|text_evidence_end|>"),
    "reasoning": ("<|reasoning_start|>", "<|reasoning_end|>"),
    "emotion": ("<|emotion_start|>", "<|emotion_end|>"),
}


def _all_new_special_tokens() -> tuple[str, ...]:
    tokens = [CANVAS_START_TOKEN, CANVAS_END_TOKEN]
    for start_token, end_token in CONDITION_BOUNDARY_TOKENS.values():
        tokens.extend((start_token, end_token))
    for block_name in TARGET_BLOCKS:
        tokens.extend(BLOCK_BOUNDARY_TOKENS[block_name])
    return tuple(tokens)


DIFFUMER_SPECIAL_TOKENS = _all_new_special_tokens()


@dataclass(frozen=True, slots=True)
class CanvasConfig:
    """Maximum number of generated tokens allocated to each output block."""

    visual_evidence_length: int = 32
    audio_evidence_length: int = 32
    text_evidence_length: int = 32
    reasoning_length: int = 96
    emotion_length: int = 8
    append_eos: bool = True

    def block_lengths(self) -> dict[str, int]:
        lengths = {
            "visual_evidence": self.visual_evidence_length,
            "audio_evidence": self.audio_evidence_length,
            "text_evidence": self.text_evidence_length,
            "reasoning": self.reasoning_length,
            "emotion": self.emotion_length,
        }
        for block_name, length in lengths.items():
            if length <= 0:
                raise ValueError(
                    f"Canvas length for {block_name} must be positive, got {length}."
                )
        return lengths


@dataclass(frozen=True, slots=True)
class CanvasLayout:
    """Absolute positions and content slices in one fixed-size canvas."""

    sequence_length: int
    canvas_start_position: int
    canvas_end_position: int
    block_start_positions: dict[str, int]
    block_end_positions: dict[str, int]
    block_slices: dict[str, slice]

    @classmethod
    def from_config(cls, config: CanvasConfig) -> "CanvasLayout":
        cursor = 0
        canvas_start_position = cursor
        cursor += 1

        block_start_positions: dict[str, int] = {}
        block_end_positions: dict[str, int] = {}
        block_slices: dict[str, slice] = {}
        lengths = config.block_lengths()

        for block_name in TARGET_BLOCKS:
            block_start_positions[block_name] = cursor
            cursor += 1
            block_slices[block_name] = slice(cursor, cursor + lengths[block_name])
            cursor += lengths[block_name]
            block_end_positions[block_name] = cursor
            cursor += 1

        canvas_end_position = cursor
        cursor += 1
        return cls(
            sequence_length=cursor,
            canvas_start_position=canvas_start_position,
            canvas_end_position=canvas_end_position,
            block_start_positions=block_start_positions,
            block_end_positions=block_end_positions,
            block_slices=block_slices,
        )


@dataclass(frozen=True, slots=True)
class CanvasTokenIds:
    """Resolved tokenizer ids used by the multimodal prefix and canvas."""

    mask: int
    pad: int
    eos: int
    canvas_start: int
    canvas_end: int
    condition_start: dict[str, int]
    condition_end: dict[str, int]
    block_start: dict[str, int]
    block_end: dict[str, int]


@dataclass(slots=True)
class CanvasBatch:
    """A batched clean or fully masked E-R-Y canvas."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    output_mask: torch.Tensor
    supervision_mask: torch.Tensor
    block_ids: torch.Tensor

    def to(self, device: torch.device | str) -> "CanvasBatch":
        return CanvasBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device),
            output_mask=self.output_mask.to(device),
            supervision_mask=self.supervision_mask.to(device),
            block_ids=self.block_ids.to(device),
        )


def _token_id(tokenizer: Any, token: str) -> int:
    vocabulary = tokenizer.get_vocab()
    if token not in vocabulary:
        raise ValueError(f"Tokenizer does not contain required token {token!r}.")
    token_id = int(vocabulary[token])
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [token_id]:
        raise ValueError(
            f"Special token {token!r} must encode to one id, got {encoded}."
        )
    return token_id


def register_canvas_special_tokens(
    tokenizer: Any,
    model: Any | None = None,
) -> CanvasTokenIds:
    """Register DiffuMER tokens and optionally resize LLaDA input/output tables."""
    vocabulary = tokenizer.get_vocab()
    missing_tokens = [
        token for token in DIFFUMER_SPECIAL_TOKENS if token not in vocabulary
    ]
    if missing_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": missing_tokens},
            replace_additional_special_tokens=False,
        )

    if model is not None:
        embedding_count = model.get_input_embeddings().num_embeddings
        if embedding_count < len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))

    vocabulary = tokenizer.get_vocab()
    if MASK_TOKEN not in vocabulary:
        raise ValueError(
            f"LLaDA mask token {MASK_TOKEN!r} is missing from the tokenizer."
        )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id for canvas termination.")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    return CanvasTokenIds(
        mask=_token_id(tokenizer, MASK_TOKEN),
        pad=int(pad_token_id),
        eos=int(eos_token_id),
        canvas_start=_token_id(tokenizer, CANVAS_START_TOKEN),
        canvas_end=_token_id(tokenizer, CANVAS_END_TOKEN),
        condition_start={
            name: _token_id(tokenizer, tokens[0])
            for name, tokens in CONDITION_BOUNDARY_TOKENS.items()
        },
        condition_end={
            name: _token_id(tokenizer, tokens[1])
            for name, tokens in CONDITION_BOUNDARY_TOKENS.items()
        },
        block_start={
            name: _token_id(tokenizer, BLOCK_BOUNDARY_TOKENS[name][0])
            for name in TARGET_BLOCKS
        },
        block_end={
            name: _token_id(tokenizer, BLOCK_BOUNDARY_TOKENS[name][1])
            for name in TARGET_BLOCKS
        },
    )


class CanvasBuilder:
    """Build fixed-layout canvases for diffusion training and generation."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        config: CanvasConfig | None = None,
        model: Any | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or CanvasConfig()
        self.layout = CanvasLayout.from_config(self.config)
        self.token_ids = register_canvas_special_tokens(tokenizer, model=model)
        self._template_ids, self._output_mask, self._block_ids = (
            self._build_template()
        )

    def _build_template(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = self.layout.sequence_length
        input_ids = torch.full((length,), self.token_ids.pad, dtype=torch.long)
        output_mask = torch.zeros(length, dtype=torch.bool)
        block_ids = torch.zeros(length, dtype=torch.long)

        input_ids[self.layout.canvas_start_position] = self.token_ids.canvas_start
        input_ids[self.layout.canvas_end_position] = self.token_ids.canvas_end

        for block_index, block_name in enumerate(TARGET_BLOCKS, start=1):
            input_ids[self.layout.block_start_positions[block_name]] = (
                self.token_ids.block_start[block_name]
            )
            input_ids[self.layout.block_end_positions[block_name]] = (
                self.token_ids.block_end[block_name]
            )
            block_slice = self.layout.block_slices[block_name]
            output_mask[block_slice] = True
            block_ids[block_slice] = block_index

        return input_ids, output_mask, block_ids

    def _new_batch(self, batch_size: int) -> CanvasBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        input_ids = self._template_ids.unsqueeze(0).repeat(batch_size, 1)
        output_mask = self._output_mask.unsqueeze(0).repeat(batch_size, 1)
        block_ids = self._block_ids.unsqueeze(0).repeat(batch_size, 1)
        return CanvasBatch(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
            labels=torch.full_like(input_ids, -100),
            output_mask=output_mask,
            supervision_mask=torch.zeros_like(input_ids, dtype=torch.bool),
            block_ids=block_ids,
        )

    @staticmethod
    def _target_value(
        target: EMERTarget | Mapping[str, str | None],
        block_name: str,
    ) -> str | None:
        if isinstance(target, Mapping):
            value = target.get(block_name)
        else:
            value = getattr(target, block_name)
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"Target block {block_name} must be str or None, got "
                f"{type(value).__name__}."
            )
        return value

    def _encode_block(self, text: str, maximum_length: int) -> list[int]:
        token_ids = list(
            self.tokenizer.encode(text, add_special_tokens=False)
        )
        if self.config.append_eos:
            token_ids = token_ids[: max(0, maximum_length - 1)]
            token_ids.append(self.token_ids.eos)
        else:
            token_ids = token_ids[:maximum_length]
        return token_ids

    def encode_targets(
        self,
        targets: Sequence[EMERTarget | Mapping[str, str | None]],
        *,
        device: torch.device | str | None = None,
    ) -> CanvasBatch:
        """Create clean canvases; only annotated content positions get labels."""
        canvas = self._new_batch(len(targets))
        block_lengths = self.config.block_lengths()

        for batch_index, target in enumerate(targets):
            for block_name in TARGET_BLOCKS:
                value = self._target_value(target, block_name)
                if value is None:
                    continue
                token_ids = self._encode_block(
                    value,
                    maximum_length=block_lengths[block_name],
                )
                if not token_ids:
                    continue
                block_slice = self.layout.block_slices[block_name]
                start = block_slice.start
                end = start + len(token_ids)
                encoded = torch.tensor(token_ids, dtype=torch.long)
                canvas.input_ids[batch_index, start:end] = encoded
                canvas.labels[batch_index, start:end] = encoded
                canvas.supervision_mask[batch_index, start:end] = True

        return canvas.to(device) if device is not None else canvas

    def build_masked(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> CanvasBatch:
        """Create inference canvases with every output slot set to MDM mask."""
        canvas = self._new_batch(batch_size)
        canvas.input_ids[canvas.output_mask] = self.token_ids.mask
        return canvas.to(device) if device is not None else canvas

    def block_mask(self, block_name: str, batch_size: int = 1) -> torch.Tensor:
        if block_name not in self.layout.block_slices:
            raise KeyError(f"Unknown canvas block: {block_name}")
        mask = torch.zeros(
            (batch_size, self.layout.sequence_length),
            dtype=torch.bool,
        )
        mask[:, self.layout.block_slices[block_name]] = True
        return mask

    def decode_blocks(self, token_ids: torch.Tensor) -> list[dict[str, str]]:
        """Decode canvases, stopping each block at EOS, PAD, or remaining MASK."""
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2 or token_ids.shape[1] != self.layout.sequence_length:
            raise ValueError(
                "token_ids must have shape [batch, canvas_length], got "
                f"{tuple(token_ids.shape)}."
            )

        stop_ids = {self.token_ids.eos, self.token_ids.pad, self.token_ids.mask}
        decoded: list[dict[str, str]] = []
        for row in token_ids.detach().cpu().tolist():
            blocks: dict[str, str] = {}
            for block_name in TARGET_BLOCKS:
                block_slice = self.layout.block_slices[block_name]
                content: list[int] = []
                for token_id in row[block_slice]:
                    if token_id in stop_ids:
                        break
                    content.append(token_id)
                blocks[block_name] = self.tokenizer.decode(
                    content,
                    skip_special_tokens=True,
                ).strip()
            decoded.append(blocks)
        return decoded
