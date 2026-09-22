# diffumer/data/schema.py
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Split = Literal["train", "val", "test"]

TARGET_BLOCKS = (
    "visual_evidence",   # E_v
    "audio_evidence",    # E_a
    "text_evidence",     # E_t
    "reasoning",         # R
    "emotion",           # Y
)


@dataclass(slots=True)
class MultimodalInput:
    # Visual
    video_path: str | None = None
    frame_paths: list[str] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)

    # Audio
    audio_path: str | None = None
    audio_feature_path: str | None = None

    # Text
    transcript: str = ""
    language: str | None = None

    def available_modalities(self) -> dict[str, bool]:
        return {
            "visual": bool(self.video_path or self.frame_paths),
            "audio": bool(self.audio_path or self.audio_feature_path),
            "text": bool(self.transcript.strip()),
        }

    def validate(self) -> None:
        modalities = self.available_modalities()

        if not any(modalities.values()):
            raise ValueError("Sample contains no available input modality.")

        if (
            self.frame_timestamps
            and len(self.frame_paths) != len(self.frame_timestamps)
        ):
            raise ValueError(
                "frame_paths and frame_timestamps must have the same length."
            )


@dataclass(slots=True)
class EMERTarget:
    visual_evidence: str | None = None
    audio_evidence: str | None = None
    text_evidence: str | None = None
    reasoning: str | None = None
    emotion: str | None = None

    def supervision_mask(self) -> dict[str, bool]:
        """
        True 表示该 block 有人工/可信监督。
        None 表示数据集没有该标注，不应计算该 block 的监督损失。
        """
        return {
            name: getattr(self, name) is not None
            for name in TARGET_BLOCKS
        }

    def has_supervision(self) -> bool:
        return any(self.supervision_mask().values())


@dataclass(slots=True)
class EMERSample:
    schema_version: str
    sample_id: str
    dataset: str
    split: Split
    inputs: MultimodalInput

    # 测试集可能没有 target
    target: EMERTarget | None = None

    # 原始ID、时长、fps、标签空间等非核心信息
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, require_target: bool = False) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty.")

        if not self.dataset.strip():
            raise ValueError("dataset must not be empty.")

        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {self.split}")

        self.inputs.validate()

        if require_target and self.target is None:
            raise ValueError(
                f"Target is required for sample {self.sample_id}."
            )

        if self.target is not None and not self.target.has_supervision():
            raise ValueError(
                f"Target of sample {self.sample_id} has no supervision."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EMERSample":
        input_data = data.get("inputs", {})
        target_data = data.get("target")

        return cls(
            schema_version=data.get("schema_version", "1.0"),
            sample_id=data["sample_id"],
            dataset=data["dataset"],
            split=data["split"],
            inputs=MultimodalInput(**input_data),
            target=(
                EMERTarget(**target_data)
                if target_data is not None
                else None
            ),
            metadata=data.get("metadata", {}),
        )