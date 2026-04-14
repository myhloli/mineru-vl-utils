import atexit
import json
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger

_QWEN_VL_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl"}
_COMPAT_MODEL_DIRS: list[Path] = []
_LM_HEAD_WEIGHT_KEYS = {"lm_head.weight", "language_model.lm_head.weight"}


def _cleanup_compat_model_dirs() -> None:
    for compat_dir in _COMPAT_MODEL_DIRS:
        shutil.rmtree(compat_dir, ignore_errors=True)


atexit.register(_cleanup_compat_model_dirs)


def _build_mlx_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    patched_config = deepcopy(config)
    text_config = patched_config.get("text_config")
    if patched_config.get("model_type") not in _QWEN_VL_MODEL_TYPES or not isinstance(text_config, dict):
        return patched_config

    # mlx_vlm's Qwen2-VL config builder reconstructs text_config from root keys.
    # Mirror nested text_config fields to the root so tied-embedding models stay consistent.
    for key, value in text_config.items():
        patched_config[key] = value
    return patched_config


def _needs_mlx_config_patch(config: dict[str, Any]) -> bool:
    if config.get("model_type") not in _QWEN_VL_MODEL_TYPES:
        return False

    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or not text_config:
        return False

    return any(config.get(key) != value for key, value in text_config.items())


def _iter_safetensors_paths(model_path: Path) -> list[Path]:
    return sorted(
        path
        for path in model_path.glob("*.safetensors")
        if not path.name.startswith(".")
        and not path.name.startswith("._")
        and path.name != "consolidated.safetensors"
    )


def _model_has_explicit_lm_head(model_path: Path) -> bool:
    try:
        from safetensors import safe_open
    except ImportError:
        logger.debug("safetensors is unavailable; assuming no explicit lm_head for {}.", model_path)
        return False

    for weight_path in _iter_safetensors_paths(model_path):
        try:
            with safe_open(weight_path, framework="pt", device="cpu") as f:
                if any(key in _LM_HEAD_WEIGHT_KEYS for key in f.keys()):
                    return True
        except Exception as exc:
            logger.debug("Skipping unreadable safetensors candidate {}: {}", weight_path, exc)
    return False


def _prepare_mlx_model_path(model_path: Path) -> Path:
    with open(model_path / "config.json", encoding="utf-8") as f:
        config = json.load(f)

    if not _needs_mlx_config_patch(config):
        return model_path

    if _model_has_explicit_lm_head(model_path):
        logger.debug(
            "Keeping original MLX model dir for {} because weights already include an explicit lm_head.",
            model_path,
        )
        return model_path

    compat_dir = Path(tempfile.mkdtemp(prefix="mineru-mlx-compat-"))
    for child in model_path.iterdir():
        if child.name == "config.json":
            continue
        os.symlink(child, compat_dir / child.name, target_is_directory=child.is_dir())

    patched_config = _build_mlx_compatible_config(config)
    with open(compat_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(patched_config, f, ensure_ascii=False, indent=2)

    _COMPAT_MODEL_DIRS.append(compat_dir)
    logger.debug(
        "Prepared MLX compatibility model dir for {} at {}.",
        model_path,
        compat_dir,
    )
    return compat_dir


def load_mlx_model(path_or_hf_repo: str, **kwargs):
    try:
        from mlx_vlm import load as mlx_load
        from mlx_vlm.utils import get_model_path
    except ImportError:
        raise ImportError("Please install mlx-vlm to use the mlx-engine backend.")

    revision = kwargs.get("revision")
    force_download = kwargs.get("force_download", False)
    model_path = get_model_path(
        path_or_hf_repo,
        revision=revision,
        force_download=force_download,
    )
    prepared_path = _prepare_mlx_model_path(model_path)
    return mlx_load(str(prepared_path), **kwargs)
