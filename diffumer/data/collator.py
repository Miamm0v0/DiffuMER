"""Batch collation for DiffuMER multimodal diffusion training."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from diffumer.model.canvas import CanvasBuilder

from .masking import MaskingConfig, corrupt_canvas


class DiffuMERDataCollator:
    """Convert ``DiffuMERDataset`` samples into one model-ready batch.

    Training flow:

        dataset samples
            -> gather/pad multimodal conditions
            -> encode clean E_v/E_a/E_t/R/Y canvas S_0
            -> corrupt supervised canvas tokens to S_t
            -> return model inputs plus diffusion metadata

    ``canvas_builder`` should normally be ``model.canvas_builder`` so tokenizer,
    special-token ids, and canvas layout exactly match the model.
    """

    def __init__(
        self,
        canvas_builder: CanvasBuilder,
        *,
        image_processor: Any | None = None,
        masking_config: MaskingConfig | None = None,
        training: bool = True,
    ) -> None:
        self.canvas_builder = canvas_builder
        self.image_processor = image_processor
        self.masking_config = masking_config or MaskingConfig()
        self.training = training

    def _process_video_frames(
        self,
        value: Any,
    ) -> torch.Tensor | None:
        """Accept already processed tensors or process a list of PIL frames."""
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            return value

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes),
        ):
            frames = list(value)
            if not frames:
                return None
        else:
            raise TypeError(
                "video_frames must be a tensor, a sequence of PIL images, "
                "or None."
            )

        if self.image_processor is None:
            raise TypeError(
                "Received unprocessed video frames but no image_processor was "
                "provided. Pass model.image_processor to DiffuMERDataCollator "
                "or preprocess frames inside DiffuMERDataset."
            )

        if hasattr(self.image_processor, "preprocess"):
            processed = self.image_processor.preprocess(
                frames,
                return_tensors="pt",
            )
        else:
            try:
                processed = self.image_processor(
                    images=frames,
                    return_tensors="pt",
                )
            except TypeError:
                processed = self.image_processor(frames)

        if isinstance(processed, Mapping):
            if "pixel_values" not in processed:
                raise KeyError(
                    "image_processor returned a mapping without pixel_values."
                )
            processed = processed["pixel_values"]
        elif hasattr(processed, "pixel_values"):
            processed = processed.pixel_values

        if not isinstance(processed, torch.Tensor):
            raise TypeError(
                "image_processor must return a tensor or an object containing "
                "pixel_values."
            )

        return processed

    @staticmethod
    def _collate_audio(
        samples: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Pad variable-length HuBERT features along their time dimension."""
        features_per_sample = [
            sample.get("audio_features")
            for sample in samples
        ]

        present_features = [
            features
            for features in features_per_sample
            if features is not None
        ]

        if not present_features:
            return None, None

        for features in present_features:
            if not isinstance(features, torch.Tensor):
                raise TypeError("audio_features must be torch.Tensor or None.")
            if features.ndim != 2:
                raise ValueError(
                    "Each audio_features tensor must have shape [time, dim], "
                    f"got {tuple(features.shape)}."
                )

        feature_dim = present_features[0].shape[-1]
        dtype = present_features[0].dtype

        for features in present_features:
            if features.shape[-1] != feature_dim:
                raise ValueError(
                    "All audio features in a batch must share the same feature "
                    f"dimension; expected {feature_dim}, got "
                    f"{features.shape[-1]}."
                )

        batch_size = len(samples)
        max_length = max(
            features.shape[0]
            for features in present_features
        )

        audio_features = torch.zeros(
            (batch_size, max_length, feature_dim),
            dtype=dtype,
        )
        audio_attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.bool,
        )

        for batch_index, sample in enumerate(samples):
            features = sample.get("audio_features")
            if features is None:
                continue

            length = features.shape[0]
            audio_features[batch_index, :length] = features

            sample_mask = sample.get("audio_attention_mask")

            if sample_mask is None:
                audio_attention_mask[batch_index, :length] = True
                continue

            if not isinstance(sample_mask, torch.Tensor):
                sample_mask = torch.as_tensor(sample_mask)

            sample_mask = sample_mask.squeeze().to(torch.bool)

            if (
                sample_mask.ndim != 1
                or sample_mask.shape[0] != length
            ):
                raise ValueError(
                    "audio_attention_mask must have shape [time] matching "
                    f"audio_features; got {tuple(sample_mask.shape)} for "
                    f"length={length}."
                )

            audio_attention_mask[batch_index, :length] = sample_mask

        return audio_features, audio_attention_mask

    @staticmethod
    def _collect_targets(
        samples: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, str | None] | None]:
        targets: list[Mapping[str, str | None] | None] = []

        for sample in samples:
            target = sample.get("target")

            if target is not None and not isinstance(target, Mapping):
                raise TypeError(
                    "sample['target'] must be a mapping or None, got "
                    f"{type(target).__name__}."
                )

            targets.append(target)

        return targets

    def __call__(
        self,
        samples: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch.")

        # Keep video samples as a list because different examples may contain
        # different numbers of frames. DiffuMERModel already supports this form.
        video_frames_list = [
            self._process_video_frames(
                sample.get("video_frames")
            )
            for sample in samples
        ]

        video_frames: list[torch.Tensor | None] | None = (
            None
            if all(
                frames is None
                for frames in video_frames_list
            )
            else video_frames_list
        )

        audio_features, audio_attention_mask = self._collate_audio(
            samples
        )

        transcripts = [
            str(sample.get("transcript") or "")
            for sample in samples
        ]

        targets = self._collect_targets(samples)

        timesteps: torch.Tensor | None = None
        mask_probabilities: torch.Tensor | None = None
        corruption_mask: torch.Tensor | None = None

        if self.training:
            if any(target is None for target in targets):
                missing = [
                    str(samples[index].get("sample_id", index))
                    for index, target in enumerate(targets)
                    if target is None
                ]
                raise ValueError(
                    "Training requires targets for every sample; missing target "
                    f"for {missing[:5]}."
                )

            clean_canvas = self.canvas_builder.encode_targets(
                [
                    target
                    for target in targets
                    if target is not None
                ]
            )

            masking_result = corrupt_canvas(
                clean_canvas,
                mask_token_id=self.canvas_builder.token_ids.mask,
                config=self.masking_config,
            )

            canvas = masking_result.canvas
            timesteps = masking_result.timesteps
            mask_probabilities = masking_result.mask_probabilities
            corruption_mask = masking_result.corruption_mask

        else:
            has_targets = [
                target is not None
                for target in targets
            ]

            if all(has_targets):
                # Validation with gold targets: return clean S_0 canvas.
                canvas = self.canvas_builder.encode_targets(
                    [
                        target
                        for target in targets
                        if target is not None
                    ]
                )
            elif not any(has_targets):
                # Pure generation/test batch: begin from fully masked canvas.
                canvas = self.canvas_builder.build_masked(
                    len(samples)
                )
            else:
                raise ValueError(
                    "Evaluation/inference batches cannot mix samples with "
                    "and without targets."
                )

        # Keys used by DiffuMERModel.forward are kept at the top level.
        # timesteps/mask_probabilities/corruption_mask are consumed by the
        # training objective rather than by the backbone forward pass.
        return {
            "canvas": canvas,
            "video_frames": video_frames,
            "audio_features": audio_features,
            "audio_attention_mask": audio_attention_mask,
            "transcripts": transcripts,
            "timesteps": timesteps,
            "mask_probabilities": mask_probabilities,
            "corruption_mask": corruption_mask,
            "sample_ids": [
                str(sample.get("sample_id", index))
                for index, sample in enumerate(samples)
            ],
            "datasets": [
                str(sample.get("dataset", ""))
                for sample in samples
            ],
            "frame_timestamps": [
                sample.get("frame_timestamps")
                for sample in samples
            ],
            "metadata": [
                sample.get("metadata", {})
                for sample in samples
            ],
        }
