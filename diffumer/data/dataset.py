"""Dataset for unified DiffuMER multimodal JSONL manifests."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .schema import EMERSample, Split


ImageProcessor = Callable[..., Any]


def _uniform_indices(length: int, count: int | None) -> list[int]:
    if length <= 0:
        return []
    if count is None or count >= length:
        return list(range(length))
    if count <= 0:
        raise ValueError("num_video_frames must be positive or None.")
    if count == 1:
        return [length // 2]
    return [round(i * (length - 1) / (count - 1)) for i in range(count)]


def _safe_torch_load(path: Path) -> Any:
    """Load tensor-only feature files without allowing arbitrary pickle objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "Loading HuBERT features safely requires a PyTorch version that "
            "supports torch.load(..., weights_only=True)."
        ) from exc


class DiffuMERDataset(Dataset[dict[str, Any]]):
    """Read frames, cached HuBERT features, transcript, and E-R-Y targets.

    Relative paths in the JSONL file are resolved from ``data_root``.  By
    default, ``data_root`` is the manifest's parent directory.

    ``image_processor`` may be a Hugging Face image processor (with a
    ``preprocess`` method) or any callable accepting a list of RGB PIL images.
    Without a processor, ``video_frames`` is returned as a list of PIL images;
    this is useful when visual preprocessing is deferred to the collator.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        data_root: str | Path | None = None,
        split: Split | None = None,
        num_video_frames: int | None = 8,
        image_processor: ImageProcessor | None = None,
        require_audio_features: bool = True,
        require_target: bool = False,
        validate_samples: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest_path}")

        self.data_root = (
            Path(data_root).expanduser().resolve()
            if data_root is not None
            else self.manifest_path.parent
        )
        self.num_video_frames = num_video_frames
        self.image_processor = image_processor
        self.require_audio_features = require_audio_features
        self.require_target = require_target
        self.samples = self._read_manifest(
            split=split,
            validate_samples=validate_samples,
        )

        if not self.samples:
            split_message = f" for split={split}" if split is not None else ""
            raise ValueError(
                f"No samples found in {self.manifest_path}{split_message}."
            )

    def _read_manifest(
        self,
        *,
        split: Split | None,
        validate_samples: bool,
    ) -> list[EMERSample]:
        samples: list[EMERSample] = []
        sample_ids: set[str] = set()
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_sample = json.loads(line)
                    sample = EMERSample.from_dict(raw_sample)
                    if validate_samples:
                        sample.validate(require_target=self.require_target)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid sample in {self.manifest_path} at line "
                        f"{line_number}: {exc}"
                    ) from exc

                if split is not None and sample.split != split:
                    continue
                if sample.sample_id in sample_ids:
                    raise ValueError(
                        f"Duplicate sample_id {sample.sample_id!r} in "
                        f"{self.manifest_path}."
                    )
                sample_ids.add(sample.sample_id)
                samples.append(sample)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = self.data_root / path
        return path.resolve()

    def _load_explicit_frames(
        self,
        frame_paths: Sequence[str],
        frame_timestamps: Sequence[float],
    ) -> tuple[list[Image.Image], list[float]]:
        selected_indices = _uniform_indices(
            len(frame_paths),
            self.num_video_frames,
        )
        frames: list[Image.Image] = []
        timestamps: list[float] = []
        for frame_index in selected_indices:
            frame_path = self._resolve_path(frame_paths[frame_index])
            if not frame_path.is_file():
                raise FileNotFoundError(f"Video frame does not exist: {frame_path}")
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB").copy())
            if frame_timestamps:
                timestamps.append(float(frame_timestamps[frame_index]))
        return frames, timestamps

    def _decode_video(
        self,
        video_path_value: str,
    ) -> tuple[list[Image.Image], list[float]]:
        try:
            from decord import VideoReader, cpu
        except ImportError as exc:
            raise ImportError(
                "Reading inputs.video_path requires decord. Install the "
                "LLaDA-V training dependencies or provide inputs.frame_paths."
            ) from exc

        video_path = self._resolve_path(video_path_value)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")

        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        selected_indices = _uniform_indices(len(reader), self.num_video_frames)
        if not selected_indices:
            raise ValueError(f"Video contains no decodable frames: {video_path}")

        frame_array = reader.get_batch(selected_indices).asnumpy()
        frames = [Image.fromarray(frame).convert("RGB") for frame in frame_array]
        fps = float(reader.get_avg_fps())
        timestamps = (
            [index / fps for index in selected_indices]
            if fps > 0
            else []
        )
        return frames, timestamps

    def _process_frames(self, frames: list[Image.Image]) -> Any:
        if not frames or self.image_processor is None:
            return frames

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

        if isinstance(processed, dict):
            if "pixel_values" not in processed:
                raise KeyError(
                    "image_processor returned a dictionary without pixel_values."
                )
            return processed["pixel_values"]
        if hasattr(processed, "pixel_values"):
            return processed.pixel_values
        return processed

    def _load_audio_features(
        self,
        feature_path_value: str | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
        if not feature_path_value:
            if self.require_audio_features:
                raise ValueError("Sample has no inputs.audio_feature_path.")
            return None, None, {}

        feature_path = self._resolve_path(feature_path_value)
        if not feature_path.is_file():
            raise FileNotFoundError(
                f"HuBERT feature file does not exist: {feature_path}"
            )

        payload = _safe_torch_load(feature_path)
        if isinstance(payload, torch.Tensor):
            features = payload
            attention_mask = None
            feature_metadata: dict[str, Any] = {}
        elif isinstance(payload, dict):
            if "features" not in payload:
                raise KeyError(
                    f"HuBERT feature file has no 'features' key: {feature_path}"
                )
            features = payload["features"]
            attention_mask = payload.get("attention_mask")
            feature_metadata = {
                key: value
                for key, value in payload.items()
                if key not in {"features", "attention_mask"}
            }
        else:
            raise TypeError(
                f"Unsupported HuBERT feature payload in {feature_path}: "
                f"{type(payload).__name__}"
            )

        if not isinstance(features, torch.Tensor):
            raise TypeError(f"'features' is not a tensor in {feature_path}.")
        if features.ndim == 3 and features.shape[0] == 1:
            features = features.squeeze(0)
        if features.ndim != 2:
            raise ValueError(
                f"Expected HuBERT features [time, dim], got {features.shape} "
                f"in {feature_path}."
            )
        if features.shape[0] == 0:
            raise ValueError(f"HuBERT features are empty: {feature_path}")

        features = features.to(torch.float32).contiguous()
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[0], dtype=torch.bool)
        else:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.as_tensor(attention_mask)
            attention_mask = attention_mask.squeeze().to(torch.bool).contiguous()
            if attention_mask.ndim != 1 or attention_mask.shape[0] != features.shape[0]:
                raise ValueError(
                    "HuBERT attention_mask must have shape [time], but got "
                    f"{attention_mask.shape} for features {features.shape} in "
                    f"{feature_path}."
                )

        feature_metadata["path"] = str(feature_path)
        return features, attention_mask, feature_metadata

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        inputs = sample.inputs

        if inputs.frame_paths:
            frames, timestamps = self._load_explicit_frames(
                inputs.frame_paths,
                inputs.frame_timestamps,
            )
        elif inputs.video_path:
            frames, timestamps = self._decode_video(inputs.video_path)
        else:
            frames, timestamps = [], []

        video_frames = self._process_frames(frames)
        audio_features, audio_attention_mask, audio_feature_metadata = (
            self._load_audio_features(inputs.audio_feature_path)
        )
        target = asdict(sample.target) if sample.target is not None else None
        target_supervision_mask = (
            sample.target.supervision_mask()
            if sample.target is not None
            else None
        )

        return {
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "split": sample.split,
            "video_frames": video_frames,
            "frame_timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "audio_features": audio_features,
            "audio_attention_mask": audio_attention_mask,
            "audio_feature_metadata": audio_feature_metadata,
            "transcript": inputs.transcript,
            "language": inputs.language,
            "target": target,
            "target_supervision_mask": target_supervision_mask,
            "metadata": sample.metadata,
        }
