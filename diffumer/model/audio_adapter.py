"""Resample frozen HuBERT features into a small set of LLaDA audio tokens."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class AudioAdapter(nn.Module):
    """Resample HuBERT frames, then project the audio latents into LLaDA space.

    Learned latent queries cross-attend to the HuBERT timeline, reducing a
    variable number of frames to a fixed, compact sequence before the MLP
    projector.  In practice ``num_audio_latents`` is usually 16 or 32.

    Args:
        hubert_hidden_size: Last dimension of the cached HuBERT features.  It is
            768 for ``facebook/hubert-base-ls960``.
        llada_hidden_size: LLaDA embedding dimension (4096 for LLaDA-V 8B).
        num_audio_latents: Number of resampled audio tokens (typically 16/32).
        num_attention_heads: Attention heads in the temporal resampler.
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
        num_audio_latents: int = 32,
        num_attention_heads: int = 8,
        projector_hidden_size: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        add_modality_embedding: bool = True,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if hubert_hidden_size <= 0 or llada_hidden_size <= 0:
            raise ValueError("Input and output hidden sizes must be positive.")
        if num_audio_latents <= 0:
            raise ValueError("num_audio_latents must be positive.")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if hubert_hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hubert_hidden_size must be divisible by num_attention_heads."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        projector_hidden_size = projector_hidden_size or llada_hidden_size
        self.hubert_hidden_size = hubert_hidden_size
        self.llada_hidden_size = llada_hidden_size
        self.num_audio_latents = num_audio_latents
        self.input_norm = nn.LayerNorm(hubert_hidden_size)
        self.latent_queries = nn.Parameter(
            torch.empty(1, num_audio_latents, hubert_hidden_size)
        )
        self.latent_norm = nn.LayerNorm(hubert_hidden_size)
        self.temporal_resampler = nn.MultiheadAttention(
            hubert_hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.resampler_output_norm = nn.LayerNorm(hubert_hidden_size)

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
        nn.init.normal_(
            self.latent_queries,
            mean=0.0,
            std=self.initializer_range,
        )
        nn.init.normal_(
            self.temporal_resampler.in_proj_weight,
            mean=0.0,
            std=self.initializer_range,
        )
        if self.temporal_resampler.in_proj_bias is not None:
            nn.init.zeros_(self.temporal_resampler.in_proj_bias)

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
        *,
        return_attention_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Resample ``[..., time, hubert_dim]`` into LLaDA-sized audio tokens.

        When requested, the returned mask describes the resampled latent
        sequence rather than the original HuBERT frames.  Samples containing
        no valid frames produce zero latents and an all-false latent mask.
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
        if audio_features.shape[-2] == 0:
            raise ValueError("audio_features must contain at least one frame.")

        squeeze_batch = audio_features.ndim == 2
        if squeeze_batch:
            audio_features = audio_features.unsqueeze(0)

        batch_size, num_frames = audio_features.shape[:2]
        if audio_attention_mask is None:
            frame_mask = torch.ones(
                (batch_size, num_frames),
                dtype=torch.bool,
                device=audio_features.device,
            )
        else:
            if audio_attention_mask.ndim == 1 and squeeze_batch:
                audio_attention_mask = audio_attention_mask.unsqueeze(0)
            expected_shape = (batch_size, num_frames)
            if tuple(audio_attention_mask.shape) != expected_shape:
                raise ValueError(
                    "audio_attention_mask must match [batch, time]; expected "
                    f"{expected_shape}, got {tuple(audio_attention_mask.shape)}."
                )
            frame_mask = audio_attention_mask.to(
                device=audio_features.device,
                dtype=torch.bool,
            )

        has_audio = frame_mask.any(dim=1)
        safe_frame_mask = frame_mask.clone()
        safe_frame_mask[~has_audio, 0] = True

        frames = self.input_norm(audio_features)
        frames = frames.masked_fill(~frame_mask.unsqueeze(-1), 0.0)
        latents = self.latent_queries.expand(batch_size, -1, -1)
        resampled, _ = self.temporal_resampler(
            query=self.latent_norm(latents),
            key=frames,
            value=frames,
            key_padding_mask=~safe_frame_mask,
            need_weights=False,
        )
        embeddings = self.resampler_output_norm(latents + resampled)
        embeddings = self.projector(embeddings)
        embeddings = self.output_norm(embeddings)
        if self.audio_embedding is not None:
            embeddings = embeddings + self.audio_embedding

        latent_mask = has_audio.unsqueeze(1).expand(-1, self.num_audio_latents)
        embeddings = embeddings * latent_mask.unsqueeze(-1).to(embeddings.dtype)

        if squeeze_batch:
            embeddings = embeddings.squeeze(0)
            latent_mask = latent_mask.squeeze(0)
        if return_attention_mask:
            return embeddings, latent_mask
        return embeddings
