"""Convert supported emotion datasets to DiffuMER's unified JSONL schema.

The first supported dataset is MER2025/MER-Caption+.  Its annotation is built
from three official files:

* ``track2_train_mercaptionplus.csv``: open-vocabulary emotion labels;
* ``track3_train_mercaptionplus.csv``: multimodal emotion description;
* ``subtitle_chieng.csv``: Chinese and English transcripts.

MER-Caption+ does not provide separate E_v/E_a/E_t labels.  The official
description is therefore stored as R (``target.reasoning``), the open-set labels
as Y (``target.emotion``), and the three evidence blocks remain unsupervised.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffumer.data.schema import EMERSample, EMERTarget, MultimodalInput  # noqa: E402


LOGGER = logging.getLogger("diffumer.prepare_data")
DATASET_NAME = "MER2025/MER-Caption+"

TranscriptPreference = Literal[
    "english",
    "chinese",
    "prefer-english",
    "prefer-chinese",
]
MissingMediaPolicy = Literal["keep", "drop", "error"]


def _increase_csv_field_limit() -> None:
    """Allow long MER-Caption descriptions on all Python platforms."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def read_csv_index(
    path: Path,
    *,
    value_columns: tuple[str, ...],
    id_columns: tuple[str, ...] = ("name", "names"),
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Read a CSV keyed by sample id while preserving the source row order."""
    if not path.is_file():
        raise FileNotFoundError(f"Required annotation file does not exist: {path}")

    _increase_csv_field_limit()
    order: list[str] = []
    rows_by_id: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        id_column = next(
            (column for column in id_columns if column in fieldnames),
            None,
        )
        if id_column is None:
            raise ValueError(
                f"{path} must contain one of the id columns {id_columns}; "
                f"found {fieldnames}."
            )
        missing_columns = [
            column for column in value_columns if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"{path} is missing required columns: {missing_columns}."
            )

        for line_number, row in enumerate(reader, start=2):
            sample_id = (row.get(id_column) or "").strip()
            if not sample_id:
                raise ValueError(f"Empty sample id in {path} at line {line_number}.")
            if sample_id in rows_by_id:
                raise ValueError(
                    f"Duplicate sample id {sample_id!r} in {path} at "
                    f"line {line_number}."
                )
            rows_by_id[sample_id] = {
                column: (row.get(column) or "").strip()
                for column in value_columns
            }
            order.append(sample_id)
    return order, rows_by_id


def parse_openset_labels(raw_value: str) -> list[str]:
    """Parse both ``[happy, calm]`` and Python-list-style label strings."""
    value = raw_value.strip()
    if not value or value == "[]":
        return ["neutral"]

    labels: list[str]
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        labels = [str(label).strip() for label in parsed]
    else:
        unwrapped = value
        if unwrapped.startswith("[") and unwrapped.endswith("]"):
            unwrapped = unwrapped[1:-1]
        labels = [
            part.strip().strip("'\"")
            for part in unwrapped.split(",")
        ]

    unique_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if not label:
            continue
        comparison_key = label.casefold()
        if comparison_key not in seen:
            seen.add(comparison_key)
            unique_labels.append(label)
    return unique_labels or ["neutral"]


def choose_transcript(
    row: Mapping[str, str],
    preference: TranscriptPreference,
) -> tuple[str, str | None, str]:
    """Return transcript text, ISO-like language, and its source column."""
    english = (row.get("english") or "").strip()
    chinese = (row.get("chinese") or "").strip()

    candidates: tuple[tuple[str, str, str], ...]
    if preference == "english":
        candidates = ((english, "en", "english"),)
    elif preference == "chinese":
        candidates = ((chinese, "zh", "chinese"),)
    elif preference == "prefer-english":
        candidates = (
            (english, "en", "english"),
            (chinese, "zh", "chinese"),
        )
    elif preference == "prefer-chinese":
        candidates = (
            (chinese, "zh", "chinese"),
            (english, "en", "english"),
        )
    else:
        raise ValueError(f"Unsupported transcript preference: {preference}")

    for text, language, source_column in candidates:
        if text:
            return text, language, source_column
    return "", None, "none"


def stable_validation_assignment(
    sample_id: str,
    *,
    validation_ratio: float,
    split_seed: int,
) -> bool:
    """Assign validation samples without depending on input order or RNG state."""
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in [0, 1).")
    if validation_ratio == 0.0:
        return False
    digest = hashlib.sha256(f"{split_seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / 2**64 < validation_ratio


def path_for_manifest(path: Path, manifest_directory: Path) -> str:
    """Represent media paths relative to the generated manifests when possible."""
    try:
        relative_path = os.path.relpath(path, start=manifest_directory)
    except ValueError:
        return path.resolve().as_posix()
    return Path(relative_path).as_posix()


def apply_missing_media_policy(
    expected_path: Path,
    *,
    policy: MissingMediaPolicy,
) -> str | None:
    if expected_path.is_file() or policy == "keep":
        return str(expected_path)
    if policy == "drop":
        return None
    # Missing paths are collected and reported together by the caller in error mode.
    return str(expected_path)


def write_jsonl_atomic(samples: Iterable[EMERSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(sample.to_dict(), ensure_ascii=False) + "\n"
                )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _check_annotation_ids(
    label_ids: set[str],
    reasoning_ids: set[str],
    transcript_ids: set[str],
) -> None:
    without_reasoning = label_ids - reasoning_ids
    without_label = reasoning_ids - label_ids
    without_transcript = label_ids - transcript_ids
    if without_reasoning or without_label or without_transcript:
        examples = {
            "labels_without_reasoning": sorted(without_reasoning)[:5],
            "reasoning_without_labels": sorted(without_label)[:5],
            "labels_without_transcript": sorted(without_transcript)[:5],
        }
        raise ValueError(
            "MER-Caption+ annotation tables do not contain matching sample ids: "
            f"{examples}"
        )


def prepare_mer2025_mercaptionplus(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    label_path = source_root / "track2_train_mercaptionplus.csv"
    reasoning_path = source_root / "track3_train_mercaptionplus.csv"
    transcript_path = source_root / "subtitle_chieng.csv"

    sample_order, label_rows = read_csv_index(
        label_path,
        value_columns=("openset",),
    )
    _, reasoning_rows = read_csv_index(
        reasoning_path,
        value_columns=("reason",),
    )
    _, transcript_rows = read_csv_index(
        transcript_path,
        value_columns=("chinese", "english"),
    )
    _check_annotation_ids(
        set(label_rows),
        set(reasoning_rows),
        set(transcript_rows),
    )

    manifests = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
    }
    output_paths = [manifests["train"], output_dir / "dataset_info.json"]
    if args.validation_ratio > 0:
        output_paths.append(manifests["val"])
    existing_outputs = [path for path in output_paths if path.exists()]
    if existing_outputs and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    audio_root = source_root / "audio"
    video_root = source_root / "video" / "video"
    openface_root = source_root / "openface_face"
    manifest_directory = output_dir

    samples_by_split: dict[str, list[EMERSample]] = {"train": [], "val": []}
    missing_audio: list[str] = []
    missing_video: list[str] = []
    missing_openface: list[str] = []
    transcript_source_counts = {"english": 0, "chinese": 0, "none": 0}

    for sample_id in sample_order:
        labels = parse_openset_labels(label_rows[sample_id]["openset"])
        reasoning = reasoning_rows[sample_id]["reason"].strip()
        if not reasoning:
            raise ValueError(f"Sample {sample_id} has an empty reasoning target.")

        transcript, language, transcript_source = choose_transcript(
            transcript_rows[sample_id],
            args.transcript_language,
        )
        transcript_source_counts[transcript_source] += 1

        expected_audio_path = audio_root / f"{sample_id}.wav"
        expected_video_path = video_root / f"{sample_id}.mp4"
        expected_openface_path = (
            openface_root / sample_id / f"{sample_id}.npy"
        )
        audio_available = expected_audio_path.is_file()
        video_available = expected_video_path.is_file()
        openface_available = expected_openface_path.is_file()
        if not audio_available:
            missing_audio.append(sample_id)
        if args.visual_source == "video" and not video_available:
            missing_video.append(sample_id)
        if not openface_available:
            missing_openface.append(sample_id)

        audio_path_value = apply_missing_media_policy(
            expected_audio_path,
            policy=args.missing_media,
        )
        video_path_value = None
        if args.visual_source == "video":
            video_path_value = apply_missing_media_policy(
                expected_video_path,
                policy=args.missing_media,
            )

        is_validation = stable_validation_assignment(
            sample_id,
            validation_ratio=args.validation_ratio,
            split_seed=args.split_seed,
        )
        split = "val" if is_validation else "train"

        sample = EMERSample(
            schema_version="1.0",
            sample_id=sample_id,
            dataset=DATASET_NAME,
            split=split,
            inputs=MultimodalInput(
                video_path=(
                    path_for_manifest(Path(video_path_value), manifest_directory)
                    if video_path_value is not None
                    else None
                ),
                audio_path=(
                    path_for_manifest(Path(audio_path_value), manifest_directory)
                    if audio_path_value is not None
                    else None
                ),
                transcript=transcript,
                language=language,
            ),
            target=EMERTarget(
                visual_evidence=None,
                audio_evidence=None,
                text_evidence=None,
                reasoning=reasoning,
                emotion=", ".join(labels),
            ),
            metadata={
                "source_dataset": "MER2025",
                "source_subset": "MER-Caption+",
                "emotion_labels": labels,
                "transcript_source": transcript_source,
                "media_available": {
                    "audio": audio_available,
                    "video": video_available,
                    "openface": openface_available,
                },
                "openface_feature_path": path_for_manifest(
                    expected_openface_path,
                    manifest_directory,
                ),
                "split_method": "stable_sha256",
                "split_seed": args.split_seed,
            },
        )
        sample.validate(require_target=True)
        samples_by_split[split].append(sample)

    if args.missing_media == "error" and (missing_audio or missing_video):
        raise FileNotFoundError(
            "Required MER-Caption+ media is missing. "
            f"audio={len(missing_audio)} (examples: {missing_audio[:5]}), "
            f"video={len(missing_video)} (examples: {missing_video[:5]})."
        )

    summary: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "schema_version": "1.0",
        "source_root": str(source_root),
        "annotation_files": {
            "emotion": label_path.name,
            "reasoning": reasoning_path.name,
            "transcript": transcript_path.name,
        },
        "num_samples": len(sample_order),
        "split_counts": {
            split: len(samples)
            for split, samples in samples_by_split.items()
        },
        "validation_ratio": args.validation_ratio,
        "split_seed": args.split_seed,
        "transcript_preference": args.transcript_language,
        "transcript_source_counts": transcript_source_counts,
        "visual_source": args.visual_source,
        "missing_media_policy": args.missing_media,
        "missing_media_counts": {
            "audio": len(missing_audio),
            "video": len(missing_video),
            "openface": len(missing_openface),
        },
        "target_supervision": {
            "visual_evidence": False,
            "audio_evidence": False,
            "text_evidence": False,
            "reasoning": True,
            "emotion": True,
        },
    }

    if not args.dry_run:
        write_jsonl_atomic(samples_by_split["train"], manifests["train"])
        if samples_by_split["val"]:
            write_jsonl_atomic(samples_by_split["val"], manifests["val"])
        write_json_atomic(summary, output_dir / "dataset_info.json")

    LOGGER.info(
        "Prepared %s: train=%d, val=%d, missing audio=%d, "
        "missing video=%d, missing OpenFace=%d%s",
        DATASET_NAME,
        len(samples_by_split["train"]),
        len(samples_by_split["val"]),
        len(missing_audio),
        len(missing_video),
        len(missing_openface),
        " (dry run)" if args.dry_run else "",
    )
    if missing_audio or missing_video:
        LOGGER.warning(
            "Some referenced media files are unavailable. Extract the official "
            "audio/video archives before feature extraction or training."
        )
    if missing_openface:
        LOGGER.info(
            "OpenFace .npy files are incomplete or unavailable; they are optional "
            "and are not consumed by the current LLaDA-V visual pipeline."
        )
    return summary


DATASET_PREPARERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "mer2025-mercaptionplus": prepare_mer2025_mercaptionplus,
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert emotion datasets to DiffuMER JSONL manifests."
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_PREPARERS),
        default="mer2025-mercaptionplus",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "mer2025-dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "mer2025_mercaptionplus",
    )
    parser.add_argument(
        "--transcript-language",
        choices=("english", "chinese", "prefer-english", "prefer-chinese"),
        default="prefer-english",
        help="English is used by the official baseline; prefer-english falls back to Chinese.",
    )
    parser.add_argument(
        "--visual-source",
        choices=("video", "none"),
        default="video",
        help="OpenFace files are .npy features, not image frames, so they are metadata only.",
    )
    parser.add_argument(
        "--missing-media",
        choices=("keep", "drop", "error"),
        default="keep",
        help=(
            "keep writes expected paths, drop omits unavailable modalities, and "
            "error requires all selected media to exist."
        ),
    )
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    DATASET_PREPARERS[args.dataset](args)


if __name__ == "__main__":
    main()
