"""Command-line entry point for DiffuMER training."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from diffumer.data.masking import MaskingConfig
from diffumer.model.audio_adapter import AudioAdapter
from diffumer.model.canvas import CanvasConfig
from diffumer.model.diffumer_model import DiffuMERModel
from diffumer.training2.trainer import (
    DiffuMERTrainer,
    TrainerConfig,
    seed_everything,
)


TRAIN_MODES = ("full", "freeze_llada", "audio_adapter", "verifier")


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
        raise TypeError("The top-level training config must be a mapping.")
    return config


def _reject_unknown(config: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {section}: {unknown}")


def _normalize_verifier_config(value: Any) -> dict[str, Any]:
    """A disabled verifier requires no head implementation or annotations."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("model.verifier must be a mapping.")
    config = dict(value)
    _reject_unknown(config, {"enabled", "class_path", "kwargs"}, "model.verifier")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("model.verifier.enabled must be a boolean.")
    class_path = config.get("class_path")
    kwargs = config.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise TypeError("model.verifier.kwargs must be a mapping.")
    if class_path is not None and not isinstance(class_path, str):
        raise TypeError("model.verifier.class_path must be a dotted class name.")
    if enabled and (not class_path or "." not in class_path):
        raise ValueError(
            "Enabled verifier requires model.verifier.class_path pointing to "
            "an implemented scoring nn.Module."
        )
    return {"enabled": enabled, "class_path": class_path, "kwargs": dict(kwargs)}


def _resolve_verifier_class(config: dict[str, Any]) -> type[torch.nn.Module] | None:
    if not config["enabled"]:
        return None
    module_name, class_name = config["class_path"].rsplit(".", 1)
    try:
        head_class = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Verifier head implementation is unavailable: {config['class_path']}."
        ) from exc
    if not inspect.isclass(head_class) or not issubclass(head_class, torch.nn.Module):
        raise TypeError("model.verifier.class_path must identify an nn.Module class.")
    signature = inspect.signature(head_class.forward)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs and not {"attention_mask", "block_ids"}.issubset(signature.parameters):
        raise TypeError(
            "Verifier forward must accept canvas hidden states plus keyword "
            "arguments attention_mask and block_ids, returning a score mapping."
        )
    return head_class


def _validate_verifier_supervision(
    dataset: DiffuMERDataset,
    collator: DiffuMERDataCollator,
) -> None:
    """Check the data interface; missing annotations remain valid per sample.

    Later batches are checked by the trainer/loss. Never synthesize semantic
    annotations from token logits or wording equality with the E-R-Y target.
    """
    # Validation must not advance the diffusion-corruption RNG for training.
    with torch.random.fork_rng(devices=[]):
        batch = collator([dataset[0]])
    targets = batch.get("verification_targets")
    masks = batch.get("verification_supervision_mask")
    if not isinstance(targets, Mapping) or not isinstance(masks, Mapping):
        raise ValueError(
            "Enabled verifier requires independent verification_targets and "
            "verification_supervision_mask mappings from the dataset/collator. "
            "Extend that data interface before enabling the head; ordinary "
            "E-R-Y targets are not semantic verifier annotations."
        )
    names = {"quality", "grounding", "adequacy", "relation"}
    _reject_unknown(dict(targets), names, "verification_targets")
    _reject_unknown(dict(masks), names, "verification_supervision_mask")
    for name in names:
        target = targets.get(name)
        if target is None:
            continue
        mask = masks.get(name)
        if mask is None:
            raise ValueError(f"Verifier {name} annotations require a separate valid mask.")
        target = torch.as_tensor(target)
        mask = torch.as_tensor(mask)
        if target.shape != mask.shape:
            raise ValueError(f"Verifier {name} supervision mask must match targets.")


def _resolve_required_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _resolve_local_model_path(value: str, base_dir: Path) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    relative_candidate = base_dir / candidate
    if relative_candidate.exists() or value.startswith("."):
        return str(relative_candidate.resolve())
    return value


def _build_model(
    raw_config: dict[str, Any],
    *,
    config_dir: Path,
    train_mode: str,
) -> DiffuMERModel:
    model_config = dict(raw_config)
    _reject_unknown(
        model_config,
        {
            "model_name_or_path",
            "model_base",
            "model_name",
            "hubert_hidden_size",
            "max_visual_tokens",
            "max_text_tokens",
            "canvas",
            "audio_adapter",
            "verifier",
            "loader_kwargs",
        },
        "model",
    )

    canvas_config = CanvasConfig(**model_config.pop("canvas", {}))
    adapter_config = dict(model_config.pop("audio_adapter", {}))
    verifier_config = _normalize_verifier_config(model_config.pop("verifier", {}))
    if train_mode not in TRAIN_MODES:
        raise ValueError(f"Unsupported train_mode: {train_mode}")
    if train_mode == "verifier" and not verifier_config["enabled"]:
        raise ValueError("train_mode='verifier' requires model.verifier.enabled=true.")
    # Fail before loading the base model when a requested head is unavailable.
    verifier_class = _resolve_verifier_class(verifier_config)
    loader_kwargs = dict(model_config.pop("loader_kwargs", {}))
    model_name_or_path = str(
        model_config.pop("model_name_or_path", "GSAI-ML/LLaDA-V")
    )
    model_name_or_path = _resolve_local_model_path(
        model_name_or_path,
        config_dir,
    )
    model_base = model_config.pop("model_base", None)
    if model_base is not None:
        model_base = _resolve_local_model_path(str(model_base), config_dir)

    hubert_hidden_size = int(model_config.pop("hubert_hidden_size", 768))
    model_name = str(model_config.pop("model_name", "llava_llada"))
    max_visual_tokens = model_config.pop("max_visual_tokens", None)
    max_text_tokens = int(model_config.pop("max_text_tokens", 512))
    model = DiffuMERModel.from_pretrained(
        model_name_or_path=model_name_or_path,
        model_base=model_base,
        model_name=model_name,
        canvas_config=canvas_config,
        hubert_hidden_size=hubert_hidden_size,
        max_visual_tokens=max_visual_tokens,
        max_text_tokens=max_text_tokens,
        freeze_llada_v=train_mode != "full",
        **loader_kwargs,
    )

    if adapter_config:
        model.audio_adapter = AudioAdapter.from_llada_config(
            model.llada_v.config,
            hubert_hidden_size=hubert_hidden_size,
            **adapter_config,
        )
        model.align_audio_adapter_to_llada()
    if verifier_class is not None:
        head_kwargs = dict(verifier_config["kwargs"])
        if "hidden_size" in inspect.signature(verifier_class).parameters:
            head_kwargs.setdefault("hidden_size", int(model.llada_v.config.hidden_size))
        heads = verifier_class(**head_kwargs)
        embedding_weight = model.llada_v.get_input_embeddings().weight
        model.verifier_heads = heads.to(
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )
        verifier_config["kwargs"] = head_kwargs

    # Save resolved paths/defaults and the actual adapter architecture. The
    # trainer also snapshots LLaDA config, tokenizer IDs/files, and token rows.
    adapter = model.audio_adapter
    linear_layers = [
        layer for layer in adapter.projector if isinstance(layer, torch.nn.Linear)
    ]
    model.model_construction_config = {
        "model_name_or_path": model_name_or_path,
        "model_base": model_base,
        "model_name": model_name,
        "hubert_hidden_size": adapter.hubert_hidden_size,
        "max_visual_tokens": max_visual_tokens,
        "max_text_tokens": max_text_tokens,
        "canvas": asdict(canvas_config),
        "audio_adapter": {
            "num_audio_latents": adapter.num_audio_latents,
            "num_attention_heads": adapter.temporal_resampler.num_heads,
            "projector_hidden_size": (
                linear_layers[0].out_features
                if len(linear_layers) > 1 else adapter.llada_hidden_size
            ),
            "num_layers": len(linear_layers),
            "dropout": adapter.temporal_resampler.dropout,
            "add_modality_embedding": adapter.audio_embedding is not None,
            "initializer_range": adapter.initializer_range,
        },
        "verifier": verifier_config,
        "loader_kwargs": dict(loader_kwargs),
    }
    return model


def _build_dataset_and_loader(
    raw_config: dict[str, Any],
    *,
    config_dir: Path,
    model: DiffuMERModel,
    masking_config: MaskingConfig,
    seed: int,
) -> tuple[DiffuMERDataset, DiffuMERDataCollator, DataLoader[Any]]:
    data_config = dict(raw_config)
    _reject_unknown(
        data_config,
        {
            "manifest_path",
            "data_root",
            "split",
            "num_video_frames",
            "require_audio_features",
            "validate_samples",
            "batch_size",
            "shuffle",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "drop_last",
        },
        "data",
    )
    if "manifest_path" not in data_config:
        raise ValueError("data.manifest_path is required.")

    batch_size = int(data_config.pop("batch_size", 1))
    shuffle = bool(data_config.pop("shuffle", True))
    num_workers = int(data_config.pop("num_workers", 0))
    pin_memory = bool(data_config.pop("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(data_config.pop("persistent_workers", False))
    drop_last = bool(data_config.pop("drop_last", False))
    if num_workers == 0:
        persistent_workers = False

    manifest_path = _resolve_required_path(
        str(data_config.pop("manifest_path")),
        config_dir,
    )
    data_root = data_config.pop("data_root", None)
    if data_root is not None:
        data_root = _resolve_required_path(str(data_root), config_dir)

    dataset = DiffuMERDataset(
        manifest_path,
        data_root=data_root,
        split=data_config.pop("split", "train"),
        num_video_frames=data_config.pop("num_video_frames", 8),
        require_audio_features=bool(
            data_config.pop("require_audio_features", True)
        ),
        require_target=True,
        validate_samples=bool(data_config.pop("validate_samples", True)),
    )
    collator = DiffuMERDataCollator(
        model.canvas_builder,
        image_processor=model.image_processor,
        masking_config=masking_config,
        mode="train",
    )
    if model.verifier_heads is not None:
        _validate_verifier_supervision(dataset, collator)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        generator=generator,
    )
    return dataset, collator, loader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DiffuMER from a JSON or YAML config.",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--train-mode",
        choices=TRAIN_MODES,
        help="Override training.train_mode from the config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override training.output_dir from the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    _reject_unknown(
        config,
        {"model", "data", "masking", "training"},
        "top level",
    )

    training_values = dict(config.get("training", {}))
    if args.train_mode is not None:
        training_values["train_mode"] = args.train_mode
    requested_mode = training_values.get("train_mode", "audio_adapter")
    if requested_mode not in TRAIN_MODES:
        raise ValueError(f"Unsupported train_mode: {requested_mode}")
    verifier_config = _normalize_verifier_config(
        dict(config.get("model", {})).get("verifier", {})
    )
    if requested_mode == "verifier" and not verifier_config["enabled"]:
        raise ValueError("train_mode='verifier' requires model.verifier.enabled=true.")
    if args.output_dir is not None:
        training_values["output_dir"] = str(args.output_dir.expanduser().resolve())
    elif "output_dir" in training_values:
        training_values["output_dir"] = _resolve_required_path(
            str(training_values["output_dir"]),
            config_path.parent,
        )
    trainer_config = TrainerConfig(**training_values)
    if not verifier_config["enabled"] and any((
        trainer_config.lambda_quality,
        trainer_config.lambda_grounding,
        trainer_config.lambda_adequacy,
        trainer_config.lambda_relation,
    )):
        raise ValueError("Positive verifier loss weights require model.verifier.enabled=true.")
    seed_everything(trainer_config.seed)

    model = _build_model(
        dict(config.get("model", {})),
        config_dir=config_path.parent,
        train_mode=trainer_config.train_mode,
    )
    masking_config = MaskingConfig(**dict(config.get("masking", {})))
    dataset, collator, train_loader = _build_dataset_and_loader(
        dict(config.get("data", {})),
        config_dir=config_path.parent,
        model=model,
        masking_config=masking_config,
        seed=trainer_config.seed,
    )

    trainer = DiffuMERTrainer(
        model,
        train_loader,
        collator,
        config=trainer_config,
        model_config=model.model_construction_config,
        verifier_config=(
            model.model_construction_config["verifier"]
            if model.verifier_heads is not None else None
        ),
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "samples": len(dataset),
                "batches": len(train_loader),
                "train_mode": trainer_config.train_mode,
                "trainable_parameters": trainable,
                "total_parameters": total,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
