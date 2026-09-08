"""Transformers 5 多模态处理器的显式组件配置。"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ProcessorMixin, Qwen2VLForConditionalGeneration


def load_transformers_model(
    model_path: str | Path,
    *,
    device_map: str | dict[str, str] = "auto",
) -> "Qwen2VLForConditionalGeneration":
    """显式同步文本权重绑定配置，防止 5.10 将未单独保存的 lm_head 随机初始化。"""
    from transformers import AutoConfig, PretrainedConfig, Qwen2VLForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path)
    raw_config, _ = PretrainedConfig.get_config_dict(model_path)
    text_config = config.get_text_config(decoder=True)
    tied = raw_config.get("tie_word_embeddings")
    if tied is None:
        tied = text_config.tie_word_embeddings
    text_config.tie_word_embeddings = tied
    config.tie_word_embeddings = tied
    return Qwen2VLForConditionalGeneration.from_pretrained(model_path, config=config, device_map=device_map, dtype="auto")


def load_transformers_processor(model_path: str | Path) -> "ProcessorMixin":
    """仅为图像组件选择 TorchVision，避免 backend 参数传给只读的视频处理器属性。"""
    from transformers import AutoImageProcessor, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)
    processor.image_processor = AutoImageProcessor.from_pretrained(model_path, backend="torchvision")
    return processor


__all__ = ["load_transformers_model", "load_transformers_processor"]
