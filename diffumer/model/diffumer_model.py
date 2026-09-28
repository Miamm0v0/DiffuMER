"""DiffuMER wrapper that joins visual, audio, text, and E-R-Y canvas tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .audio_adapter import AudioAdapter
from .canvas import CanvasBatch, CanvasBuilder, CanvasConfig


@dataclass(slots=True)
class DiffuMEROutput:
    """Outputs aligned to the fixed E-R-Y canvas."""

    logits: torch.Tensor
    canvas_hidden_states: torch.Tensor
    canvas_attention_mask: torch.Tensor
    labels: torch.Tensor
    output_mask: torch.Tensor
    supervision_mask: torch.Tensor
    full_attention_mask: torch.Tensor
    modality_spans: dict[str, slice]
    full_hidden_states: torch.Tensor | None = None
    backbone_outputs: Any | None = None


class DiffuMERModel(nn.Module):
    """Insert A/V/T conditions before a structured diffusion canvas.

    LLaDA-V's public training ``forward`` performs its own random masking.  This
    wrapper instead calls the underlying bidirectional LLaDA backbone directly,
    because DiffuMER's masking time step is constructed by the data collator.
    """

    def __init__(
        self,
        llada_v: nn.Module,
        tokenizer: Any,
        *,
        canvas_config: CanvasConfig | None = None,
        audio_adapter: AudioAdapter | None = None,
        hubert_hidden_size: int = 768,
        image_processor: Any | None = None,
        context_length: int | None = None,
        max_visual_tokens: int | None = None,
        max_text_tokens: int = 512,
        freeze_llada_v: bool = False,
    ) -> None:
        super().__init__()
        self.llada_v = llada_v
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_visual_tokens = max_visual_tokens
        self.max_text_tokens = max_text_tokens

        if max_visual_tokens is not None and max_visual_tokens <= 0:
            raise ValueError("max_visual_tokens must be positive or None.")
        if max_text_tokens <= 0:
            raise ValueError("max_text_tokens must be positive.")

        # Registration may resize both the input embedding and LM head.
        self.canvas_builder = CanvasBuilder(
            tokenizer,
            config=canvas_config,
            model=llada_v,
        )

        hidden_size = int(getattr(llada_v.config, "hidden_size"))
        self.audio_adapter = audio_adapter or AudioAdapter.from_llada_config(
            llada_v.config,
            hubert_hidden_size=hubert_hidden_size,
        )
        if self.audio_adapter.llada_hidden_size != hidden_size:
            raise ValueError(
                "Audio adapter output dimension must equal LLaDA hidden_size: "
                f"{self.audio_adapter.llada_hidden_size} != {hidden_size}."
            )

        inferred_context_length = (
            getattr(llada_v.config, "max_position_embeddings", None)
            or getattr(llada_v.config, "tokenizer_model_max_length", None)
        )
        self.context_length = context_length or inferred_context_length
        self._llada_is_frozen = False
        self.set_llada_trainable(not freeze_llada_v)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "GSAI-ML/LLaDA-V",
        *,
        model_base: str | None = None,
        model_name: str = "llava_llada",
        canvas_config: CanvasConfig | None = None,
        audio_adapter: AudioAdapter | None = None,
        hubert_hidden_size: int = 768,
        max_visual_tokens: int | None = None,
        max_text_tokens: int = 512,
        freeze_llada_v: bool = False,
        **loader_kwargs: Any,
    ) -> "DiffuMERModel":
        """Load LLaDA-V with its official builder, without modifying it."""
        try:
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:
            raise ImportError(
                "Could not import the LLaDA-V 'llava' package. Install "
                "third_party/LLaDA-V/train in editable mode first."
            ) from exc

        tokenizer, llada_v, image_processor, context_length = (
            load_pretrained_model(
                model_name_or_path,
                model_base,
                model_name,
                **loader_kwargs,
            )
        )
        model = cls(
            llada_v,
            tokenizer,
            canvas_config=canvas_config,
            audio_adapter=audio_adapter,
            hubert_hidden_size=hubert_hidden_size,
            image_processor=image_processor,
            context_length=context_length,
            max_visual_tokens=max_visual_tokens,
            max_text_tokens=max_text_tokens,
            freeze_llada_v=freeze_llada_v,
        )
        model.align_audio_adapter_to_llada()
        return model

    def set_llada_trainable(self, trainable: bool) -> None:
        self.llada_v.requires_grad_(trainable)
        self._llada_is_frozen = not trainable
        if self._llada_is_frozen:
            self.llada_v.eval()

    def train(self, mode: bool = True) -> "DiffuMERModel":
        super().train(mode)
        if self._llada_is_frozen:
            self.llada_v.eval()
        return self

    def align_audio_adapter_to_llada(self) -> None:
        """Move the newly created adapter beside LLaDA's token embeddings."""
        embedding_weight = self.llada_v.get_input_embeddings().weight
        if embedding_weight.device.type == "meta":
            return
        self.audio_adapter.to(
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )

    def build_training_canvas(
        self,
        targets: Sequence[Any],
        *,
        device: torch.device | str | None = None,
    ) -> CanvasBatch:
        return self.canvas_builder.encode_targets(targets, device=device)

    def build_generation_canvas(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> CanvasBatch:
        return self.canvas_builder.build_masked(batch_size, device=device)

    def _embedding_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        weight = self.llada_v.get_input_embeddings().weight
        return weight.device, weight.dtype

    def _embed_token_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        device, _ = self._embedding_device_and_dtype()
        return self.llada_v.get_input_embeddings()(input_ids.to(device))

    def _vision_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        vision_tower = self.llada_v.get_vision_tower()
        try:
            parameter = next(vision_tower.parameters())
        except StopIteration:
            return self._embedding_device_and_dtype()
        return parameter.device, parameter.dtype

    def _limit_visual_tokens(self, features: torch.Tensor) -> torch.Tensor:
        if (
            self.max_visual_tokens is None
            or features.shape[0] <= self.max_visual_tokens
        ):
            return features
        pooled = F.adaptive_avg_pool1d(
            features.transpose(0, 1).unsqueeze(0),
            self.max_visual_tokens,
        )
        return pooled.squeeze(0).transpose(0, 1).contiguous()

    def _encode_one_visual_sample(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(
                "Each visual sample must be [frames, channels, height, width], "
                f"got {tuple(frames.shape)}."
            )
        vision_device, vision_dtype = self._vision_device_and_dtype()
        frames = frames.to(device=vision_device, dtype=vision_dtype)
        features = self.llada_v.encode_images(frames)
        if not isinstance(features, torch.Tensor) or features.ndim < 2:
            raise TypeError("LLaDA-V encode_images returned an invalid tensor.")
        if features.ndim == 3 and features.shape[0] > 1:
            pool_stride = int(
                getattr(self.llada_v.config, "mm_spatial_pool_stride", 1)
            )
            if pool_stride > 1 and hasattr(self.llada_v, "get_2dPool"):
                features = self.llada_v.get_2dPool(
                    features,
                    stride=pool_stride,
                )
        features = features.reshape(-1, features.shape[-1])
        features = self._limit_visual_tokens(features)
        embedding_device, embedding_dtype = self._embedding_device_and_dtype()
        return features.to(device=embedding_device, dtype=embedding_dtype)

    def _normalize_visual_batch(
        self,
        video_frames: torch.Tensor | Sequence[torch.Tensor | None],
        batch_size: int,
    ) -> list[torch.Tensor | None]:
        if isinstance(video_frames, torch.Tensor):
            if video_frames.ndim == 5:
                if video_frames.shape[0] != batch_size:
                    raise ValueError("video_frames batch size does not match canvas.")
                return list(video_frames.unbind(0))
            if video_frames.ndim == 4:
                if video_frames.shape[0] == batch_size:
                    return [frame.unsqueeze(0) for frame in video_frames]
                if batch_size == 1:
                    return [video_frames]
            if video_frames.ndim == 3 and batch_size == 1:
                return [video_frames.unsqueeze(0)]
            raise ValueError(
                "video_frames tensor must be [B,F,C,H,W], [B,C,H,W], or "
                "[F,C,H,W] for batch size 1."
            )

        samples = list(video_frames)
        if len(samples) != batch_size:
            raise ValueError("video_frames list length does not match canvas batch.")
        for sample in samples:
            if sample is not None and not isinstance(sample, torch.Tensor):
                raise TypeError(
                    "video_frames must be processed into tensors before the model."
                )
        return samples

    def _pad_embedding_sequences(
        self,
        sequences: Sequence[torch.Tensor | None],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding_device, embedding_dtype = self._embedding_device_and_dtype()
        maximum_length = max(
            (sequence.shape[0] for sequence in sequences if sequence is not None),
            default=0,
        )
        hidden_size = int(self.llada_v.config.hidden_size)
        embeddings = torch.zeros(
            (batch_size, maximum_length, hidden_size),
            device=embedding_device,
            dtype=embedding_dtype,
        )
        attention_mask = torch.zeros(
            (batch_size, maximum_length),
            device=embedding_device,
            dtype=torch.bool,
        )
        for batch_index, sequence in enumerate(sequences):
            if sequence is None:
                continue
            length = sequence.shape[0]
            embeddings[batch_index, :length] = sequence
            attention_mask[batch_index, :length] = True
        return embeddings, attention_mask

    def _encode_visual_batch(
        self,
        video_frames: torch.Tensor | Sequence[torch.Tensor | None] | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_frames is None:
            return self._pad_embedding_sequences([None] * batch_size, batch_size)
        samples = self._normalize_visual_batch(video_frames, batch_size)
        encoded = [
            self._encode_one_visual_sample(sample) if sample is not None else None
            for sample in samples
        ]
        return self._pad_embedding_sequences(encoded, batch_size)

    def _encode_audio(
        self,
        audio_features: torch.Tensor | None,
        audio_attention_mask: torch.Tensor | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio_features is None:
            return self._pad_embedding_sequences([None] * batch_size, batch_size)
        if audio_features.ndim == 2 and batch_size == 1:
            audio_features = audio_features.unsqueeze(0)
        if audio_features.ndim != 3 or audio_features.shape[0] != batch_size:
            raise ValueError(
                "audio_features must have shape [batch, time, hubert_dim]."
            )
        if audio_attention_mask is None:
            audio_attention_mask = torch.ones(
                audio_features.shape[:2],
                dtype=torch.bool,
                device=audio_features.device,
            )
        elif audio_attention_mask.ndim == 1 and batch_size == 1:
            audio_attention_mask = audio_attention_mask.unsqueeze(0)
        if tuple(audio_attention_mask.shape) != tuple(audio_features.shape[:2]):
            raise ValueError(
                "audio_attention_mask must have shape [batch, audio_time]."
            )

        adapter_parameter = next(self.audio_adapter.parameters())
        audio_features = audio_features.to(
            device=adapter_parameter.device,
            dtype=adapter_parameter.dtype,
        )
        audio_attention_mask = audio_attention_mask.to(adapter_parameter.device)
        embeddings, audio_attention_mask = self.audio_adapter(
            audio_features,
            audio_attention_mask=audio_attention_mask,
            return_attention_mask=True,
        )
        embedding_device, embedding_dtype = self._embedding_device_and_dtype()
        return (
            embeddings.to(device=embedding_device, dtype=embedding_dtype),
            audio_attention_mask.to(device=embedding_device, dtype=torch.bool),
        )

    def _encode_text(
        self,
        *,
        text_input_ids: torch.Tensor | None,
        text_attention_mask: torch.Tensor | None,
        transcripts: Sequence[str] | None,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_input_ids is not None and transcripts is not None:
            raise ValueError("Pass text_input_ids or transcripts, not both.")

        if transcripts is not None:
            if len(transcripts) != batch_size:
                raise ValueError("transcripts length does not match canvas batch.")
            encoded = self.tokenizer(
                list(transcripts),
                add_special_tokens=False,
                padding=True,
                truncation=True,
                max_length=self.max_text_tokens,
                return_tensors="pt",
            )
            text_input_ids = encoded["input_ids"]
            text_attention_mask = encoded.get("attention_mask")

        if text_input_ids is None:
            return self._pad_embedding_sequences([None] * batch_size, batch_size)
        if text_input_ids.ndim == 1 and batch_size == 1:
            text_input_ids = text_input_ids.unsqueeze(0)
        if text_input_ids.ndim != 2 or text_input_ids.shape[0] != batch_size:
            raise ValueError("text_input_ids must have shape [batch, text_time].")

        if text_attention_mask is None:
            pad_id = self.canvas_builder.token_ids.pad
            text_attention_mask = text_input_ids.ne(pad_id)
        elif text_attention_mask.ndim == 1 and batch_size == 1:
            text_attention_mask = text_attention_mask.unsqueeze(0)
        if tuple(text_attention_mask.shape) != tuple(text_input_ids.shape):
            raise ValueError(
                "text_attention_mask must have the same shape as text_input_ids."
            )

        embeddings = self._embed_token_ids(text_input_ids)
        return embeddings, text_attention_mask.to(
            device=embeddings.device,
            dtype=torch.bool,
        )

    def _wrap_condition_segment(
        self,
        modality: str,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if embeddings.shape[1] == 0:
            return embeddings, attention_mask
        batch_size = embeddings.shape[0]
        present = attention_mask.any(dim=1, keepdim=True)
        start_ids = torch.full(
            (batch_size, 1),
            self.canvas_builder.token_ids.condition_start[modality],
            dtype=torch.long,
            device=embeddings.device,
        )
        end_ids = torch.full(
            (batch_size, 1),
            self.canvas_builder.token_ids.condition_end[modality],
            dtype=torch.long,
            device=embeddings.device,
        )
        wrapped_embeddings = torch.cat(
            (self._embed_token_ids(start_ids), embeddings, self._embed_token_ids(end_ids)),
            dim=1,
        )
        wrapped_mask = torch.cat((present, attention_mask, present), dim=1)
        return wrapped_embeddings, wrapped_mask

    def forward(
        self,
        canvas: CanvasBatch,
        *,
        video_frames: torch.Tensor | Sequence[torch.Tensor | None] | None = None,
        audio_features: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
        text_input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
        transcripts: Sequence[str] | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_full_hidden_states: bool = False,
        return_backbone_outputs: bool = False,
    ) -> DiffuMEROutput:
        if canvas.input_ids.ndim != 2:
            raise ValueError("canvas.input_ids must have shape [batch, canvas_length].")
        expected_canvas_length = self.canvas_builder.layout.sequence_length
        if canvas.input_ids.shape[1] != expected_canvas_length:
            raise ValueError(
                f"Expected canvas length {expected_canvas_length}, got "
                f"{canvas.input_ids.shape[1]}."
            )
        batch_size = canvas.input_ids.shape[0]

        visual_embeddings, visual_mask = self._encode_visual_batch(
            video_frames,
            batch_size,
        )
        audio_embeddings, audio_mask = self._encode_audio(
            audio_features,
            audio_attention_mask,
            batch_size,
        )
        text_embeddings, text_mask = self._encode_text(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            transcripts=transcripts,
            batch_size=batch_size,
        )

        condition_segments = (
            ("visual", visual_embeddings, visual_mask),
            ("audio", audio_embeddings, audio_mask),
            ("text", text_embeddings, text_mask),
        )
        all_embeddings: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        modality_spans: dict[str, slice] = {}
        cursor = 0

        for modality, embeddings, attention_mask in condition_segments:
            wrapped_embeddings, wrapped_mask = self._wrap_condition_segment(
                modality,
                embeddings,
                attention_mask,
            )
            if wrapped_embeddings.shape[1] == 0:
                continue
            segment_length = wrapped_embeddings.shape[1]
            modality_spans[modality] = slice(cursor, cursor + segment_length)
            cursor += segment_length
            all_embeddings.append(wrapped_embeddings)
            all_masks.append(wrapped_mask)

        canvas_input_ids = canvas.input_ids.to(self._embedding_device_and_dtype()[0])
        canvas_embeddings = self._embed_token_ids(canvas_input_ids)
        canvas_mask = canvas.attention_mask.to(
            device=canvas_embeddings.device,
            dtype=torch.bool,
        )
        canvas_start = cursor
        canvas_end = cursor + canvas_embeddings.shape[1]
        modality_spans["canvas"] = slice(canvas_start, canvas_end)
        all_embeddings.append(canvas_embeddings)
        all_masks.append(canvas_mask)

        inputs_embeds = torch.cat(all_embeddings, dim=1)
        attention_mask = torch.cat(all_masks, dim=1)
        if self.context_length is not None and inputs_embeds.shape[1] > self.context_length:
            raise ValueError(
                f"Combined multimodal sequence has length {inputs_embeds.shape[1]}, "
                f"which exceeds context length {self.context_length}."
            )

        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids.masked_fill_(~attention_mask, 0)

        backbone = self.llada_v.get_model()
        backbone_outputs = backbone(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state
        canvas_hidden_states = hidden_states[:, canvas_start:canvas_end]
        logits = self.llada_v.lm_head(canvas_hidden_states).float()

        output = DiffuMEROutput(
            logits=logits,
            canvas_hidden_states=canvas_hidden_states,
            canvas_attention_mask=canvas_mask,
            labels=canvas.labels.to(logits.device),
            output_mask=canvas.output_mask.to(logits.device),
            supervision_mask=canvas.supervision_mask.to(logits.device),
            full_attention_mask=attention_mask,
            modality_spans=modality_spans,
            full_hidden_states=hidden_states if return_full_hidden_states else None,
            backbone_outputs=(
                backbone_outputs if return_backbone_outputs else None
            ),
        )
        return output
