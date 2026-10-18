"""Command-line entry point for DiffuMER inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LLADA_SOURCE = PROJECT_ROOT / "third_party" / "LLaDA-V" / "train"
if LLADA_SOURCE.is_dir() and str(LLADA_SOURCE) not in sys.path:
    sys.path.insert(0, str(LLADA_SOURCE))

from diffumer.data.collator import DiffuMERDataCollator
from diffumer.data.dataset import DiffuMERDataset
from diffumer.inference.policy import InferencePolicy, InferencePolicyConfig
from diffumer.inference.sampler import SamplerConfig, generate
from diffumer.model.loading import load_model_for_inference


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            config = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError(
                    "YAML config requires PyYAML; install it or use JSON."
                ) from exc
            config = yaml.safe_load(handle)
        else:
            raise ValueError("Config must be a .json, .yaml, or .yml file.")
    if not isinstance(config, dict):
        raise TypeError("The top-level inference config must be a mapping.")
    return config


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _reject_unknown(config: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {name}: {unknown}")


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _build_dataset_and_loader(
    raw_config: Mapping[str, Any],
    *,
    config_dir: Path,
    model: Any,
) -> tuple[DiffuMERDataset, DataLoader[Any]]:
    config = dict(raw_config)
    _reject_unknown(
        config,
        {
            "manifest_path",
            "data_root",
            "split",
            "num_video_frames",
            "require_audio_features",
            "validate_samples",
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
        },
        "data",
    )
    if "manifest_path" not in config:
        raise ValueError("data.manifest_path is required.")

    batch_size = int(config.pop("batch_size", 1))
    num_workers = int(config.pop("num_workers", 0))
    pin_memory = bool(config.pop("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(config.pop("persistent_workers", False))
    if batch_size <= 0:
        raise ValueError("data.batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("data.num_workers must be non-negative.")
    if num_workers == 0:
        persistent_workers = False

    manifest_path = _resolve_path(config.pop("manifest_path"), config_dir)
    data_root_value = config.pop("data_root", None)
    data_root = (
        _resolve_path(data_root_value, config_dir)
        if data_root_value is not None
        else None
    )
    dataset = DiffuMERDataset(
        manifest_path,
        data_root=data_root,
        split=config.pop("split", None),
        num_video_frames=config.pop("num_video_frames", 8),
        require_audio_features=bool(
            config.pop("require_audio_features", True)
        ),
        require_target=False,
        validate_samples=bool(config.pop("validate_samples", True)),
    )
    collator = DiffuMERDataCollator(
        model.canvas_builder,
        image_processor=model.image_processor,
        mode="generate",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return dataset, loader


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def _write_json_line(handle: TextIO, record: Mapping[str, Any]) -> None:
    handle.write(
        json.dumps(
            _json_safe(record),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    handle.write("\n")


def _sample_verification_scores(
    scores: torch.Tensor | Mapping[str, torch.Tensor] | None,
    *,
    sample_index: int,
    batch_size: int,
) -> Any:
    if scores is None:
        return None

    def select(value: torch.Tensor) -> torch.Tensor:
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[sample_index]
        return value

    if isinstance(scores, torch.Tensor):
        return _json_safe(select(scores))
    return {
        name: _json_safe(select(value))
        for name, value in scores.items()
    }


def _selected_canvas_indices(
    indices: torch.Tensor,
    sample_index: int,
) -> list[int]:
    if indices.numel() == 0:
        return []
    rows = indices[indices[:, 0].eq(sample_index)]
    return [int(index) for index in rows[:, 1].tolist()]


def _trace_record(
    *,
    result: Any,
    sample_index: int,
    sample_id: str,
    dataset_name: str,
    canvas_builder: Any,
) -> dict[str, Any]:
    batch_size = result.canvas.input_ids.shape[0]
    steps: list[dict[str, Any]] = []
    for trace in result.traces:
        visible_spans = [
            {
                "block_name": span.block_name,
                "token_indices": span.token_indices,
                "token_ids": span.token_ids,
                "text": span.text,
            }
            for span in canvas_builder.extract_visible_spans(trace.canvas)
            if span.batch_index == sample_index
        ]
        steps.append({
            "step": trace.step,
            "unmasked_token_indices": _selected_canvas_indices(
                trace.unmasked.indices,
                sample_index,
            ),
            "remasked_token_indices": _selected_canvas_indices(
                trace.remasked.indices,
                sample_index,
            ),
            "eos_tail_token_indices": trace.eos_tail_mask[
                sample_index
            ].nonzero(as_tuple=False).flatten().tolist(),
            "masked_before": int(trace.masked_before[sample_index].item()),
            "masked_after": int(trace.masked_after[sample_index].item()),
            "finished": bool(trace.finished[sample_index].item()),
            "stop_reason": trace.stop_reason,
            "verification_scores": _sample_verification_scores(
                trace.verification_scores,
                sample_index=sample_index,
                batch_size=batch_size,
            ),
            "canvas_token_ids": trace.canvas.input_ids[sample_index].tolist(),
            "visible_spans": visible_spans,
        })
    return {
        "sample_id": sample_id,
        "dataset": dataset_name,
        "stop_reason": result.stop_reason,
        "steps": steps,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DiffuMER inference from a JSON or YAML config.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Override model.checkpoint from the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output.output_dir from the config.",
    )
    parser.add_argument(
        "--save-traces",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override output.save_traces.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    _reject_unknown(
        config,
        {"model", "data", "policy", "sampler", "output", "seed"},
        "top level",
    )

    model_config = _mapping(config.get("model"), "model")
    _reject_unknown(model_config, {"checkpoint", "loader_kwargs"}, "model")
    checkpoint_value: str | Path | None = args.checkpoint
    if checkpoint_value is None:
        checkpoint_value = model_config.pop("checkpoint", None)
    else:
        model_config.pop("checkpoint", None)
    if checkpoint_value is None:
        raise ValueError("model.checkpoint or --checkpoint is required.")
    checkpoint = _resolve_path(checkpoint_value, config_path.parent)
    loader_kwargs = _mapping(
        model_config.pop("loader_kwargs", None),
        "model.loader_kwargs",
    )

    policy_values = _mapping(config.get("policy"), "policy")
    if "evidence_block_ids" in policy_values:
        policy_values["evidence_block_ids"] = tuple(
            policy_values["evidence_block_ids"]
        )
    policy = InferencePolicy(InferencePolicyConfig(**policy_values))

    sampler_values = _mapping(config.get("sampler"), "sampler")
    if "extra_forbidden_token_ids" in sampler_values:
        sampler_values["extra_forbidden_token_ids"] = tuple(
            sampler_values["extra_forbidden_token_ids"]
        )
    sampler_config = SamplerConfig(**sampler_values)

    output_config = _mapping(config.get("output"), "output")
    _reject_unknown(output_config, {"output_dir", "save_traces"}, "output")
    output_value: str | Path = (
        args.output_dir
        if args.output_dir is not None
        else output_config.pop("output_dir", "outputs/inference")
    )
    output_dir = _resolve_path(output_value, config_path.parent)
    save_traces = (
        bool(args.save_traces)
        if args.save_traces is not None
        else bool(output_config.pop("save_traces", False))
    )

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = load_model_for_inference(
        checkpoint,
        loader_kwargs=loader_kwargs,
    )
    if sampler_config.use_verifier:
        if model.verifier_heads is None:
            raise RuntimeError(
                "sampler.use_verifier=true, but the checkpoint has no verifier head."
            )
        if not bool(getattr(model, "verifier_is_trained", False)):
            raise RuntimeError(
                "sampler.use_verifier=true, but the checkpoint does not mark "
                "its verifier weights as trained."
            )

    dataset, loader = _build_dataset_and_loader(
        _mapping(config.get("data"), "data"),
        config_dir=config_path.parent,
        model=model,
    )
    model_device = model.llada_v.get_input_embeddings().weight.device
    generator = torch.Generator(device=model_device)
    generator.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    traces_path = output_dir / "traces.jsonl"
    trace_handle: TextIO | None = (
        traces_path.open("w", encoding="utf-8") if save_traces else None
    )
    processed = 0
    try:
        with predictions_path.open("w", encoding="utf-8") as prediction_handle:
            for batch in loader:
                # References deliberately remain outside the sampler/model call.
                result = generate(
                    model,
                    video_frames=batch["video_frames"],
                    audio_features=batch["audio_features"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    transcripts=batch["transcripts"],
                    initial_canvas=batch["canvas"],
                    policy=policy,
                    config=sampler_config,
                    generator=generator,
                )
                batch_size = len(batch["sample_ids"])
                for sample_index in range(batch_size):
                    prediction_record = {
                        "sample_id": batch["sample_ids"][sample_index],
                        "dataset": batch["datasets"][sample_index],
                        "prediction": result.decoded[sample_index],
                        "reference": batch["references"][sample_index],
                        "metadata": batch["metadata"][sample_index],
                        "stop_reason": result.stop_reason,
                        "num_steps": len(result.traces),
                    }
                    _write_json_line(prediction_handle, prediction_record)
                    if trace_handle is not None:
                        _write_json_line(
                            trace_handle,
                            _trace_record(
                                result=result,
                                sample_index=sample_index,
                                sample_id=batch["sample_ids"][sample_index],
                                dataset_name=batch["datasets"][sample_index],
                                canvas_builder=model.canvas_builder,
                            ),
                        )
                    processed += 1
                prediction_handle.flush()
                if trace_handle is not None:
                    trace_handle.flush()
    finally:
        if trace_handle is not None:
            trace_handle.close()

    print(
        json.dumps(
            {
                "samples": processed,
                "dataset_size": len(dataset),
                "predictions": str(predictions_path),
                "traces": str(traces_path) if save_traces else None,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
