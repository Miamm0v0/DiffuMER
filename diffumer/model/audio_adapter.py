"""Trainable projection from frozen HuBERT features to LLaDA embeddings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class AudioAdapter(nn.Module):
    """Map every HuBERT frame into the LLaDA token embedding space.

    The module deliberately keeps the temporal length unchanged.  Padding or
    temporal resampling can therefore be handled by the future canvas/collator
    without hiding sequence-length changes inside this projection.

    Args:
        hubert_hidden_size: Last dimension of the cached HuBERT features.  It is
            768 for ``facebook/hubert-base-ls960``.
        llada_hidden_size: LLaDA embedding dimension (4096 for LLaDA-V 8B).
        projector_hidden_size: Width of the MLP.  Defaults to the LLaDA hidden
            size.
        num_layers: Number of linear layers in the projector.  Must be >= 1.
        dropout: Dropout after each hidden activation.
        add_modality_embedding: Add one learned audio-type embedding to every
            valid audio position.
    """

    def __init__(
        self,
        hubert_hidden_size: int = 768,
        llada_hidden_size: int = 4096,
        *,
        projector_hidden_size: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        add_modality_embedding: bool = True,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if hubert_hidden_size <= 0 or llada_hidden_size <= 0:
            raise ValueError("Input and output hidden sizes must be positive.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        projector_hidden_size = projector_hidden_size or llada_hidden_size
        self.hubert_hidden_size = hubert_hidden_size
        self.llada_hidden_size = llada_hidden_size
        self.input_norm = nn.LayerNorm(hubert_hidden_size)

        layers: list[nn.Module] = []
        input_size = hubert_hidden_size
        for layer_index in range(num_layers):
            output_size = (
                llada_hidden_size
                if layer_index == num_layers - 1
                else projector_hidden_size
            )
            layers.append(nn.Linear(input_size, output_size))
            if layer_index != num_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
            input_size = output_size
        self.projector = nn.Sequential(*layers)
        self.output_norm = nn.LayerNorm(llada_hidden_size)

        if add_modality_embedding:
            self.audio_embedding = nn.Parameter(
                torch.empty(1, 1, llada_hidden_size)
            )
        else:
            self.register_parameter("audio_embedding", None)

        self.initializer_range = initializer_range
        self.reset_parameters()

    @classmethod
    def from_llada_config(
        cls,
        llada_config: Mapping[str, Any] | Any,
        *,
        hubert_hidden_size: int = 768,
        **kwargs: Any,
    ) -> "AudioAdapter":
        """Build an adapter from a config dict or a Transformers config object."""
        if isinstance(llada_config, Mapping):
            hidden_size = llada_config.get("hidden_size")
            initializer_range = llada_config.get("initializer_range", 0.02)
        else:
            hidden_size = getattr(llada_config, "hidden_size", None)
            initializer_range = getattr(llada_config, "initializer_range", 0.02)
        if hidden_size is None:
            raise ValueError("LLaDA config does not define hidden_size.")

        kwargs.setdefault("initializer_range", float(initializer_range))
        return cls(
            hubert_hidden_size=hubert_hidden_size,
            llada_hidden_size=int(hidden_size),
            **kwargs,
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=self.initializer_range,
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.audio_embedding is not None:
            nn.init.normal_(
                self.audio_embedding,
                mean=0.0,
                std=self.initializer_range,
            )

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project ``[..., time, hubert_dim]`` to LLaDA-sized embeddings.

        Masked positions are forced to zero after projection so biases and the
        modality embedding cannot leak padded audio into the LLaDA canvas.
        """
        if audio_features.ndim not in {2, 3}:
            raise ValueError(
                "audio_features must have shape [time, dim] or [batch, time, dim], "
                f"got {tuple(audio_features.shape)}."
            )
        if audio_features.shape[-1] != self.hubert_hidden_size:
            raise ValueError(
                f"Expected HuBERT dimension {self.hubert_hidden_size}, got "
                f"{audio_features.shape[-1]}."
            )

        squeeze_batch = audio_features.ndim == 2
        if squeeze_batch:
            audio_features = audio_features.unsqueeze(0)

        embeddings = self.input_norm(audio_features)
        embeddings = self.projector(embeddings)
        embeddings = self.output_norm(embeddings)
        if self.audio_embedding is not None:
            embeddings = embeddings + self.audio_embedding

        if audio_attention_mask is not None:
            if audio_attention_mask.ndim == 1 and squeeze_batch:
                audio_attention_mask = audio_attention_mask.unsqueeze(0)
            expected_shape = embeddings.shape[:2]
            if tuple(audio_attention_mask.shape) != tuple(expected_shape):
                raise ValueError(
                    "audio_attention_mask must match [batch, time]; expected "
                    f"{tuple(expected_shape)}, got {tuple(audio_attention_mask.shape)}."
                )
            mask = audio_attention_mask.to(
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            embeddings = embeddings * mask.unsqueeze(-1)

        return embeddings.squeeze(0) if squeeze_batch else embeddings
