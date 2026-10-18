"""Shared construction and inference loading for DiffuMER models."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .audio_adapter import AudioAdapter
from .canvas import CanvasConfig
from .diffumer_model import DiffuMERModel


VerifierFactory = Callable[[Mapping[str, Any]], nn.Module]

_MODEL_CONFIG_KEYS = {
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
}
_AUDIO_ADAPTER_KEYS = {
    "hubert_hidden_size",
    "llada_hidden_size",
    "num_audio_latents",
    "num_attention_heads",
    "projector_hidden_size",
    "num_layers",
    "dropout",
    "add_modality_embedding",
    "initializer_range",
}
_RESERVED_LOADER_KEYS = {
    "model_base",
    "model_name",
    "canvas_config",
    "audio_adapter",
    "hubert_hidden_size",
    "max_visual_tokens",
    "max_text_tokens",
    "freeze_llada_v",
}


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


def _resolve_model_reference(value: str | None, config_dir: Path | None) -> str | None:
    if value is None or config_dir is None:
        return value
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    relative = config_dir / candidate
    if relative.exists() or value.startswith("."):
        return str(relative.resolve())
    # Preserve repository/model IDs such as GSAI-ML/LLaDA-V.
    return value


def _qualified_class_name(module: nn.Module) -> str:
    return f"{type(module).__module__}.{type(module).__qualname__}"


def _import_module_class(class_path: str) -> type[nn.Module]:
    if not class_path or "." not in class_path:
        raise ValueError("Verifier class_path must be a dotted class name.")
    module_name, class_name = class_path.rsplit(".", 1)
    try:
        candidate = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Verifier head implementation is unavailable: {class_path}."
        ) from exc
    if not inspect.isclass(candidate) or not issubclass(candidate, nn.Module):
        raise TypeError("Verifier class_path must identify an nn.Module class.")
    return candidate


def _validate_verifier_forward(head: nn.Module) -> None:
    signature = inspect.signature(head.forward)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if not accepts_kwargs and not {
        "attention_mask",
        "block_ids",
    }.issubset(signature.parameters):
        raise TypeError(
            "Verifier forward must accept canvas hidden states plus keyword "
            "arguments attention_mask and block_ids."
        )


def _normalize_verifier_config(value: Any) -> dict[str, Any]:
    config = _mapping(value, "verifier")
    _reject_unknown(config, {"enabled", "class_path", "kwargs"}, "verifier")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("verifier.enabled must be a boolean.")
    class_path = config.get("class_path")
    if class_path is not None and not isinstance(class_path, str):
        raise TypeError("verifier.class_path must be a dotted class name.")
    kwargs = _mapping(config.get("kwargs"), "verifier.kwargs")
    return {
        "enabled": enabled,
        "class_path": class_path,
        "kwargs": kwargs,
    }


def _build_verifier_head(
    config: dict[str, Any],
    *,
    hidden_size: int,
    factory: VerifierFactory | None,
) -> nn.Module | None:
    if not config["enabled"]:
        if factory is not None:
            raise ValueError("verifier_factory was provided while verifier is disabled.")
        return None

    if factory is not None:
        head = factory(config)
    else:
        class_path = config["class_path"]
        if not class_path:
            raise ValueError(
                "Enabled verifier requires verifier.class_path or verifier_factory."
            )
        head_class = _import_module_class(class_path)
        kwargs = dict(config["kwargs"])
        if "hidden_size" in inspect.signature(head_class).parameters:
            kwargs.setdefault("hidden_size", hidden_size)
        head = head_class(**kwargs)
        config["kwargs"] = kwargs

    if not isinstance(head, nn.Module):
        raise TypeError("verifier_factory must return an nn.Module.")
    _validate_verifier_forward(head)
    # Store the actual import path so checkpoint reconstruction is unambiguous.
    config["class_path"] = _qualified_class_name(head)
    return head


def _audio_adapter_config(adapter: AudioAdapter) -> dict[str, Any]:
    linears = [layer for layer in adapter.projector if isinstance(layer, nn.Linear)]
    if not linears:
        raise ValueError("Audio adapter projector contains no linear layers.")
    return {
        "hubert_hidden_size": adapter.hubert_hidden_size,
        "llada_hidden_size": adapter.llada_hidden_size,
        "num_audio_latents": adapter.num_audio_latents,
        "num_attention_heads": adapter.temporal_resampler.num_heads,
        "projector_hidden_size": (
            linears[0].out_features
            if len(linears) > 1
            else adapter.llada_hidden_size
        ),
        "num_layers": len(linears),
        "dropout": adapter.temporal_resampler.dropout,
        "add_modality_embedding": adapter.audio_embedding is not None,
        "initializer_range": adapter.initializer_range,
    }


def build_model(
    config: Mapping[str, Any] | None = None,
    *,
    config_dir: str | Path | None = None,
    freeze_llada_v: bool = False,
    verifier_factory: VerifierFactory | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
) -> DiffuMERModel:
    """Build the base model and every DiffuMER-owned component.

    ``config`` follows the ``model`` section used by the training entry point.
    Values in ``loader_kwargs`` override its nested ``loader_kwargs`` mapping.
    A configured verifier may be imported through ``class_path`` or created by
    ``verifier_factory``. The returned construction metadata is suitable for
    passing to :class:`DiffuMERTrainer` when checkpoints are saved.
    """
    values = _mapping(config, "model config")
    _reject_unknown(values, _MODEL_CONFIG_KEYS, "model config")
    resolved_config_dir = (
        Path(config_dir).expanduser().resolve() if config_dir is not None else None
    )

    model_name_or_path = str(
        values.pop("model_name_or_path", "GSAI-ML/LLaDA-V")
    )
    model_name_or_path = _resolve_model_reference(
        model_name_or_path,
        resolved_config_dir,
    )
    model_base_value = values.pop("model_base", None)
    model_base = _resolve_model_reference(
        str(model_base_value) if model_base_value is not None else None,
        resolved_config_dir,
    )
    model_name = str(values.pop("model_name", "llava_llada"))
    max_visual_tokens = values.pop("max_visual_tokens", None)
    max_text_tokens = int(values.pop("max_text_tokens", 512))

    canvas_value = values.pop("canvas", None)
    if isinstance(canvas_value, CanvasConfig):
        canvas_config = canvas_value
    else:
        canvas_config = CanvasConfig(**_mapping(canvas_value, "canvas"))

    adapter_values = _mapping(values.pop("audio_adapter", None), "audio_adapter")
    _reject_unknown(adapter_values, _AUDIO_ADAPTER_KEYS, "audio_adapter")
    top_hubert_size = int(values.pop("hubert_hidden_size", 768))
    adapter_hubert_size = int(
        adapter_values.pop("hubert_hidden_size", top_hubert_size)
    )
    if adapter_hubert_size != top_hubert_size:
        raise ValueError(
            "hubert_hidden_size disagrees with audio_adapter.hubert_hidden_size."
        )
    expected_llada_size = adapter_values.pop("llada_hidden_size", None)

    verifier_config = _normalize_verifier_config(values.pop("verifier", None))
    loading = _mapping(values.pop("loader_kwargs", None), "loader_kwargs")
    loading.update(_mapping(loader_kwargs, "loader_kwargs override"))
    reserved = sorted(set(loading) & _RESERVED_LOADER_KEYS)
    if reserved:
        raise ValueError(f"Reserved keys cannot appear in loader_kwargs: {reserved}")

    model = DiffuMERModel.from_pretrained(
        model_name_or_path=model_name_or_path,
        model_base=model_base,
        model_name=model_name,
        canvas_config=canvas_config,
        hubert_hidden_size=adapter_hubert_size,
        max_visual_tokens=max_visual_tokens,
        max_text_tokens=max_text_tokens,
        freeze_llada_v=freeze_llada_v,
        **loading,
    )

    hidden_size = int(model.llada_v.config.hidden_size)
    if expected_llada_size is not None and int(expected_llada_size) != hidden_size:
        raise ValueError(
            "audio_adapter.llada_hidden_size differs from the loaded base model."
        )
    model.audio_adapter = AudioAdapter.from_llada_config(
        model.llada_v.config,
        hubert_hidden_size=adapter_hubert_size,
        **adapter_values,
    )
    model.align_audio_adapter_to_llada()

    verifier_head = _build_verifier_head(
        verifier_config,
        hidden_size=hidden_size,
        factory=verifier_factory,
    )
    if verifier_head is not None:
        embedding_weight = model.llada_v.get_input_embeddings().weight
        model.verifier_heads = verifier_head.to(
            device=embedding_weight.device,
            dtype=embedding_weight.dtype,
        )
    model.verifier_is_trained = False

    actual_adapter = _audio_adapter_config(model.audio_adapter)
    model.model_construction_config = {
        "model_name_or_path": model_name_or_path,
        "model_base": model_base,
        "model_name": model_name,
        "hubert_hidden_size": actual_adapter["hubert_hidden_size"],
        "max_visual_tokens": model.max_visual_tokens,
        "max_text_tokens": model.max_text_tokens,
        "canvas": asdict(model.canvas_builder.config),
        "audio_adapter": {
            key: value
            for key, value in actual_adapter.items()
            if key not in {"hubert_hidden_size", "llada_hidden_size"}
        },
        "verifier": verifier_config,
        "loader_kwargs": dict(loading),
    }
    return model


def _verifier_from_checkpoint(metadata: Mapping[str, Any]) -> nn.Module:
    class_path = metadata.get("class")
    if not isinstance(class_path, str):
        raise ValueError("Checkpoint verifier metadata lacks its class path.")
    saved_config = metadata.get("config")
    config = _mapping(saved_config, "checkpoint verifier config")
    configured_path = config.get("class_path")
    if configured_path is not None and configured_path != class_path:
        raise ValueError(
            "Checkpoint verifier class disagrees with its construction config."
        )
    if "kwargs" in config:
        kwargs = _mapping(config["kwargs"], "checkpoint verifier kwargs")
    else:
        # Also support heads whose saved config is directly their constructor args.
        kwargs = {
            key: value
            for key, value in config.items()
            if key not in {"enabled", "class_path"}
        }
    head = _import_module_class(class_path)(**kwargs)
    _validate_verifier_forward(head)
    return head


def load_model_for_inference(
    checkpoint: str | Path,
    *,
    loader_kwargs: Mapping[str, Any] | None = None,
    verifier_factory: VerifierFactory | None = None,
) -> DiffuMERModel:
    """Restore a checkpoint as an evaluation-only model.

    Model reconstruction is delegated to the trainer's checkpoint-v2 restore
    implementation. It verifies the saved canvas/audio/verifier construction,
    exact tokenizer vocabulary and special-token IDs, model state keys, table
    sizes, and the separately stored input-embedding/LM-head token rows.

    A custom ``verifier_factory`` receives the saved verifier metadata. When it
    is omitted, the saved class path and constructor kwargs are used. This
    function switches the result to eval mode and freezes parameters; callers
    should still wrap actual generation in ``torch.inference_mode()``.
    """
    # Lazy import keeps ordinary model construction independent of training
    # initialization while reusing the single authoritative restore path.
    from diffumer.training2.trainer import DiffuMERTrainer

    model = DiffuMERTrainer.load_model_from_checkpoint(
        checkpoint,
        loader_kwargs=loader_kwargs,
        verifier_factory=verifier_factory or _verifier_from_checkpoint,
    )
    model.requires_grad_(False)
    model.eval()
    return model
