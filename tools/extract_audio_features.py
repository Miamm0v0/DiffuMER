"""Extract frozen HuBERT frame features for a DiffuMER JSONL manifest.

The input is the unified JSONL format defined in ``diffumer.data.schema``.  A
feature file is written for every sample that has ``inputs.audio_path`` and a
copy of the manifest is produced with ``inputs.audio_feature_path`` filled in.

Example:
    python tools/extract_audio_features.py \
        --manifest data/train.jsonl \
        --output-dir data/features/hubert \
        --model-name-or-path facebook/hubert-base-ls960
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, HubertModel


LOGGER = logging.getLogger("diffumer.extract_audio_features")

_MODEL_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

_SAVE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest and report malformed records with line numbers."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}."
                )
            records.append(record)
    return records


def write_jsonl_atomic(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write JSONL through a temporary file so interrupted runs stay readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def path_for_manifest(path: Path, manifest_path: Path) -> str:
    """Prefer a portable path relative to the generated manifest."""
    try:
        relative_path = os.path.relpath(path, start=manifest_path.parent)
    except ValueError:
        # Windows raises ValueError when the two paths live on different drives.
        return path.as_posix()
    return Path(relative_path).as_posix()


def safe_name(value: str) -> str:
    """Create a readable, collision-resistant filename from an arbitrary id."""
    normalized = re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("._")
    normalized = normalized[:96] or "sample"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{normalized}-{digest}"


def select_device(device: str) -> torch.device:
    if device != "auto":
        selected = torch.device(device)
    elif torch.cuda.is_available():
        selected = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        selected = torch.device("mps")
    else:
        selected = torch.device("cpu")

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return selected


def load_audio(path: Path, target_sample_rate: int) -> tuple[torch.Tensor, int]:
    """Load audio as a mono float tensor and resample it for HuBERT."""
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[-1] == 0:
        raise ValueError(f"Invalid or empty waveform in {path}: {waveform.shape}")

    waveform = waveform.to(torch.float32).mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        sample_rate = target_sample_rate
    return waveform.contiguous(), sample_rate


class FrozenHuBERTExtractor:
    """Thin inference-only wrapper around a Hugging Face HuBERT model."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: torch.device,
        model_dtype: torch.dtype = torch.float32,
        layer: int = -1,
        trust_remote_code: bool = False,
    ) -> None:
        if device.type == "cpu" and model_dtype != torch.float32:
            LOGGER.warning(
                "Half precision was requested on CPU; using float32 for HuBERT."
            )
            model_dtype = torch.float32

        self.device = device
        self.layer = layer
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = HubertModel.from_pretrained(
            model_name_or_path,
            torch_dtype=model_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)

        sampling_rate = getattr(self.feature_extractor, "sampling_rate", None)
        if sampling_rate is None:
            raise ValueError(
                "The HuBERT feature extractor does not define a sampling rate."
            )
        self.sampling_rate = int(sampling_rate)
        self.model_dtype = next(self.model.parameters()).dtype

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return a ``[num_audio_tokens, hubert_hidden_size]`` tensor."""
        if waveform.ndim != 1:
            raise ValueError(
                f"Expected a mono waveform with shape [samples], got {waveform.shape}."
            )

        inputs = self.feature_extractor(
            waveform.numpy(),
            sampling_rate=self.sampling_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(
            device=self.device,
            dtype=self.model_dtype,
        )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        need_hidden_states = self.layer != -1
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )

        if self.layer == -1:
            features = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("HuBERT did not return hidden states.")
            try:
                features = hidden_states[self.layer]
            except IndexError as exc:
                raise ValueError(
                    f"Requested HuBERT layer {self.layer}, but the model returned "
                    f"{len(hidden_states)} hidden-state tensors."
                ) from exc

        return features.squeeze(0).detach().cpu()


def feature_output_path(
    output_dir: Path,
    dataset_name: str,
    sample_id: str,
) -> Path:
    dataset_dir = safe_name(dataset_name)
    return output_dir / dataset_dir / f"{safe_name(sample_id)}.pt"


def save_feature_file(
    path: Path,
    *,
    features: torch.Tensor,
    sample_id: str,
    audio_path: Path,
    model_name_or_path: str,
    layer: int,
    sample_rate: int,
    num_audio_samples: int,
    save_dtype: torch.dtype,
    frame_stride_seconds: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features.to(dtype=save_dtype).contiguous(),
        "attention_mask": torch.ones(features.shape[0], dtype=torch.bool),
        "sample_id": sample_id,
        "source_audio_path": str(audio_path),
        "model_name_or_path": model_name_or_path,
        "layer": layer,
        "sample_rate": sample_rate,
        "num_audio_samples": num_audio_samples,
        "duration_seconds": num_audio_samples / sample_rate,
        "feature_length": features.shape[0],
        "feature_dim": features.shape[-1],
        "frame_stride_seconds": frame_stride_seconds,
    }
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def extract_manifest(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    audio_root = (
        args.audio_root.resolve()
        if args.audio_root is not None
        else manifest_path.parent
    )
    output_manifest = (
        args.output_manifest.resolve()
        if args.output_manifest is not None
        else manifest_path.with_name(f"{manifest_path.stem}.hubert.jsonl")
    )

    records = read_jsonl(manifest_path)
    device = select_device(args.device)
    extractor = FrozenHuBERTExtractor(
        args.model_name_or_path,
        device=device,
        model_dtype=_MODEL_DTYPES[args.model_dtype],
        layer=args.layer,
        trust_remote_code=args.trust_remote_code,
    )

    convolution_stride = getattr(extractor.model.config, "conv_stride", None)
    frame_stride_seconds = None
    if convolution_stride:
        stride_samples = 1
        for stride in convolution_stride:
            stride_samples *= int(stride)
        frame_stride_seconds = stride_samples / extractor.sampling_rate

    extracted = 0
    reused = 0
    missing_audio = 0
    failed = 0

    for index, record in enumerate(tqdm(records, desc="Extracting HuBERT")):
        sample_id = str(record.get("sample_id", "")).strip()
        dataset_name = str(record.get("dataset", "dataset")).strip() or "dataset"
        inputs = record.get("inputs")
        if not isinstance(inputs, dict):
            message = f"Record {index} ({sample_id or 'unknown'}) has no inputs object."
            if not args.skip_errors:
                raise ValueError(message)
            LOGGER.error(message)
            failed += 1
            continue

        audio_path_value = inputs.get("audio_path")
        if not audio_path_value:
            missing_audio += 1
            continue
        if not sample_id:
            message = f"Record {index} has audio but no sample_id."
            if not args.skip_errors:
                raise ValueError(message)
            LOGGER.error(message)
            failed += 1
            continue

        audio_path = resolve_path(str(audio_path_value), audio_root)
        output_path = feature_output_path(
            output_dir,
            dataset_name=dataset_name,
            sample_id=sample_id,
        )

        try:
            if output_path.exists() and not args.overwrite:
                reused += 1
            else:
                waveform, sample_rate = load_audio(
                    audio_path,
                    target_sample_rate=extractor.sampling_rate,
                )
                features = extractor(waveform)
                save_feature_file(
                    output_path,
                    features=features,
                    sample_id=sample_id,
                    audio_path=audio_path,
                    model_name_or_path=args.model_name_or_path,
                    layer=args.layer,
                    sample_rate=sample_rate,
                    num_audio_samples=waveform.numel(),
                    save_dtype=_SAVE_DTYPES[args.save_dtype],
                    frame_stride_seconds=frame_stride_seconds,
                )
                extracted += 1

            inputs["audio_feature_path"] = path_for_manifest(
                output_path,
                output_manifest,
            )
        except Exception as exc:
            failed += 1
            if not args.skip_errors:
                raise RuntimeError(
                    f"Failed to extract sample {sample_id} from {audio_path}."
                ) from exc
            LOGGER.exception("Failed sample %s: %s", sample_id, exc)

    write_jsonl_atomic(records, output_manifest)
    LOGGER.info(
        "Finished: extracted=%d, reused=%d, no_audio=%d, failed=%d, manifest=%s",
        extracted,
        reused,
        missing_audio,
        failed,
        output_manifest,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract frozen HuBERT features for a DiffuMER JSONL manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Default: <input-manifest-stem>.hubert.jsonl beside the input manifest.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=None,
        help="Base directory for relative audio paths; defaults to the manifest directory.",
    )
    parser.add_argument(
        "--model-name-or-path",
        default="facebook/hubert-base-ls960",
    )
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model-dtype",
        choices=tuple(_MODEL_DTYPES),
        default="float32",
    )
    parser.add_argument(
        "--save-dtype",
        choices=tuple(_SAVE_DTYPES),
        default="float32",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
    extract_manifest(args)


if __name__ == "__main__":
    main()
