import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger

_QWEN_VL_MODEL_TYPES = {"qwen2_vl", "qwen2_5_vl"}
_LM_HEAD_WEIGHT_KEYS = {"lm_head.weight", "language_model.lm_head.weight"}


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
        if not path.name.startswith(".") and not path.name.startswith("._") and path.name != "consolidated.safetensors"
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
    """构造独立配置目录，保留原始模型与权重文件。"""
    model_path = model_path.resolve()
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
    try:
        for child in model_path.iterdir():
            if child.name == "config.json":
                continue
            os.symlink(child, compat_dir / child.name, target_is_directory=child.is_dir())
        patched_config = _build_mlx_compatible_config(config)
        with open(compat_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(patched_config, f, ensure_ascii=False, indent=2)
    except BaseException:
        shutil.rmtree(compat_dir, ignore_errors=True)
        raise

    logger.debug(
        "Prepared MLX compatibility model dir for {} at {}.",
        model_path,
        compat_dir,
    )
    return compat_dir


@contextmanager
def prepare_mlx_model_path(
    path_or_hf_repo: str | Path,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Iterator[Path]:
    """解析模型路径并管理兼容目录；调用方须在使用模型期间保持上下文。"""
    from mlx_vlm.utils import get_model_path

    model_path = Path(get_model_path(str(path_or_hf_repo), revision=revision, force_download=force_download)).resolve()
    prepared_path = _prepare_mlx_model_path(model_path)
    try:
        yield prepared_path
    finally:
        if prepared_path != model_path:
            shutil.rmtree(prepared_path, ignore_errors=True)


def load_mlx_model(path_or_hf_repo: str, **kwargs: Any) -> Any:
    """通过共享路径准备接口加载模型，并及时清理临时配置目录。"""
    from mlx_vlm import load as mlx_load

    with prepare_mlx_model_path(
        path_or_hf_repo,
        revision=kwargs.get("revision"),
        force_download=kwargs.get("force_download", False),
    ) as prepared_path:
        return mlx_load(str(prepared_path), **kwargs)


__all__ = ["load_mlx_model", "prepare_mlx_model_path"]
