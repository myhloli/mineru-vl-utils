"""验证 Transformers 5 与 LMDeploy Pipeline 的调用、批量及异步契约。"""

from __future__ import annotations

import asyncio
import sys
import subprocess
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mineru_vl_utils.vlm_client.base_client import SamplingParams, ServerError
from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient


def test_importing_vllm_client_does_not_patch_or_import_engine() -> None:
    """客户端模块导入不得尝试加载 vLLM 或安装已失效的全局 logprobs 补丁。"""
    code = """
import importlib.abc
import sys
attempts = []

class CheckImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 记录被旧补丁吞掉的导入错误，确保导入期间完全没有引擎副作用。
        if fullname.split('.')[0] == 'vllm':
            attempts.append(fullname)
            raise ImportError(fullname)

sys.meta_path.insert(0, CheckImports())
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient
assert not attempts, attempts
"""
    process = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert process.returncode == 0, process.stderr


@pytest.fixture
def pipeline_type(monkeypatch: pytest.MonkeyPatch) -> type:
    """提供公开 Pipeline 契约替身，不需要 CUDA 或 LMDeploy 二进制安装。"""

    class GenerationConfig(SimpleNamespace):
        """保存真实客户端传给 LMDeploy 的生成参数。"""

    class Pipeline:
        """模拟可从多线程调用的 Pipeline，记录并发数和参数。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """构造已解析的 backend_config 和线程安全计数器。"""
            self.backend_config = SimpleNamespace(session_len=128)
            self.calls: list[dict[str, Any]] = []
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()
            self.error = False

        def infer(self, prompts: list[Any], *, gen_config: list[Any], **kwargs: Any) -> list[Any]:
            """模拟阻塞推理并返回保持输入顺序的 Response。"""
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.calls.append({"prompts": prompts, "gen_config": gen_config, "thread": threading.get_ident(), **kwargs})
            try:
                threading.Event().wait(0.03)
                if self.error:
                    return [SimpleNamespace(text="", finish_reason="error") for _ in prompts]
                return [
                    SimpleNamespace(text=prompt[0] if isinstance(prompt, tuple) else prompt, finish_reason="stop")
                    for prompt in prompts
                ]
            finally:
                with self.lock:
                    self.active -= 1

    module = ModuleType("lmdeploy")
    module.GenerationConfig = GenerationConfig
    module.pipeline = Pipeline
    pipeline_module = ModuleType("lmdeploy.pipeline")
    pipeline_module.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "lmdeploy", module)
    monkeypatch.setitem(sys.modules, "lmdeploy.pipeline", pipeline_module)
    return Pipeline


def test_pipeline_preserves_batch_order_priority_and_sampling(pipeline_type: type) -> None:
    """不同优先级请求可以拆批，但必须保持结果顺序及各自生成参数。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline, batch_size=3, use_tqdm=False)
    outputs = client.batch_predict(
        [None] * 4,
        ["a", "b", "c", "d"],
        [SamplingParams(max_new_tokens=i + 1) for i in range(4)],
        priority=[3, 3, 1, 3],
    )
    assert outputs == ["a", "b", "c", "d"]
    assert [call["priority"] for call in pipeline.calls] == [3, 1, 3]
    assert [config.max_new_tokens for call in pipeline.calls for config in call["gen_config"]] == [1, 2, 3, 4]
    assert all(not config.skip_special_tokens for call in pipeline.calls for config in call["gen_config"])


def test_pipeline_async_calls_use_threads_and_limit_concurrency(pipeline_type: type) -> None:
    """异步批量接口不得阻塞事件循环，并保持并发限制和请求顺序。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline, max_concurrency=2, use_tqdm=False)
    main_thread = threading.get_ident()

    async def run() -> list[str]:
        """使用真实 asyncio 调度检查非阻塞适配。"""
        return await client.aio_batch_predict([None] * 5, [str(i) for i in range(5)], priority=4)

    assert asyncio.run(run()) == [str(i) for i in range(5)]
    assert pipeline.peak == 2
    assert all(call["thread"] != main_thread and call["priority"] == 4 for call in pipeline.calls)


def test_pipeline_errors_reach_sync_and_async_callers(pipeline_type: type) -> None:
    """后端错误响应不能被转换成成功的空字符串。"""
    pipeline = pipeline_type()
    pipeline.error = True
    client = LmdeployEngineVlmClient(pipeline, use_tqdm=False)
    with pytest.raises(ServerError, match="LMDeploy inference failed"):
        client.predict(None, "x")
    with pytest.raises(ServerError, match="LMDeploy inference failed"):
        asyncio.run(client.aio_predict(None, "x"))


def test_mineru_client_constructs_public_pipeline(pipeline_type: type) -> None:
    """仅传模型路径的公共入口也必须构造新版 Pipeline。"""
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="lmdeploy-engine", model_path="local-model")
    assert isinstance(client.client.lmdeploy_engine, pipeline_type)


def test_pipeline_cancellation_waits_for_inflight_worker(pipeline_type: type) -> None:
    """传播取消前必须完成不可中断的同步调用，防止引擎卸载后仍有线程使用它。"""
    pipeline = pipeline_type()
    client = LmdeployEngineVlmClient(pipeline, max_concurrency=1, use_tqdm=False)

    async def cancel() -> None:
        """在工作线程执行期间取消，并检查没有遗留在途调用。"""
        task = asyncio.create_task(client.aio_predict(None, "x"))
        while pipeline.active == 0:
            await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert pipeline.active == 0

    asyncio.run(cancel())


def test_transformers_uses_text_config_and_all_special_token_ids() -> None:
    """使用实际 Qwen2-VL 配置验证嵌套参数和多 EOS 过滤，再经过完整客户端批量调用。"""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers", minversion="5.10.1")
    from transformers import Qwen2VLConfig
    from transformers.feature_extraction_utils import BatchFeature
    from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient

    class Model:
        """保留真实配置的生成替身，记录送入 generate 的参数。"""

        def __init__(self) -> None:
            """让所有特殊 token 来源不同，避免测试因字段重复而漏检。"""
            self.config = Qwen2VLConfig(
                text_config={"max_position_embeddings": 32, "bos_token_id": 10, "eos_token_id": [11, 12], "pad_token_id": 13}
            )
            self.generation_config = SimpleNamespace(eos_token_id=[14, 15], bos_token_id=16, pad_token_id=18)
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.kwargs: dict[str, Any] = {}

        def generate(self, **kwargs: Any) -> Any:
            """返回包含普通 token 和所有特殊 token 的生成序列。"""
            self.kwargs = kwargs
            suffix = torch.tensor([[17, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22]])
            return torch.cat([kwargs["input_ids"], suffix.repeat(kwargs["input_ids"].shape[0], 1)], dim=1)

    class Processor:
        """提供真实 BatchFeature 的处理器替身，保留整数 input_ids 的搬运语义。"""

        tokenizer = SimpleNamespace(bos_token_id=20, eos_token_id=21, pad_token_id=22)

        def apply_chat_template(self, messages: list[Any], **kwargs: Any) -> str:
            """返回占位 prompt，实际参数流仍经过客户端。"""
            return "prompt"

        def __call__(self, *, text: list[str], **kwargs: Any) -> Any:
            """构造两个输入 token，检验客户端对生成前缀的截取。"""
            return BatchFeature(
                {"input_ids": torch.tensor([[1, 2]] * len(text)), "attention_mask": torch.ones(len(text), 2, dtype=torch.long)}
            )

        def batch_decode(self, ids: list[list[int]], **kwargs: Any) -> list[str]:
            """仅把保留的 token 转为字符串，方便检查过滤后的精确序列。"""
            return [",".join(str(value) for value in row) for row in ids]

    model = Model()
    client = TransformersVlmClient(model, Processor(), batch_size=2, use_tqdm=False)
    assert not hasattr(model.config, "max_position_embeddings")
    assert client.model_max_length == 32
    assert client.batch_predict([None, None], ["a", "b"]) == ["17", "17"]
    assert model.kwargs["use_cache"] is True
    assert model.kwargs["max_length"] == 32
    assert model.kwargs["input_ids"].dtype == torch.long


def test_processor_backend_is_configured_only_on_image_component(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 backend 不得传给 AutoProcessor 的视频组件，同时保留 tokenizer 和 chat template。"""
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    from mineru_vl_utils.transformers_loading import load_transformers_processor

    processor = SimpleNamespace(image_processor=None, tokenizer=object(), chat_template="template", video_processor=object())
    image_processor = object()
    calls = []

    def load_processor(path: str, **kwargs: Any) -> Any:
        """模拟会拒绝全局 backend 的复合处理器加载。"""
        assert "backend" not in kwargs
        return processor

    def load_image_processor(path: str, **kwargs: Any) -> Any:
        """记录只传给图像组件的后端配置。"""
        calls.append((path, kwargs))
        return image_processor

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", load_processor)
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", load_image_processor)
    assert load_transformers_processor("model") is processor
    assert processor.image_processor is image_processor
    assert processor.chat_template == "template"
    assert calls == [("model", {"backend": "torchvision"})]


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("raw_config", [{}, {"tie_word_embeddings": None}])
def test_model_loader_preserves_nested_embedding_tying(
    monkeypatch: pytest.MonkeyPatch, tied: bool, raw_config: dict[str, Any]
) -> None:
    """两个方向的绑定配置都由文本模型决定，避免缺失 lm_head 或意外合并独立权重。"""
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    from mineru_vl_utils.transformers_loading import load_transformers_model

    config = transformers.Qwen2VLConfig(text_config={"tie_word_embeddings": tied})
    config.tie_word_embeddings = not tied
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", lambda path: (raw_config, {}))
    seen = {}

    def load(path: str, **kwargs: Any) -> Any:
        """记录传入真实模型加载器的配置及设备，不创建大模型。"""
        seen.update(kwargs)
        return config

    monkeypatch.setattr(transformers.Qwen2VLForConditionalGeneration, "from_pretrained", load)
    assert load_transformers_model("model", device_map={"": "cpu"}) is config
    assert seen["config"].tie_word_embeddings is tied
    assert seen["dtype"] == "auto"
    assert seen["device_map"] == {"": "cpu"}


def test_model_loader_preserves_explicit_untied_root_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式配置的独立 lm_head 优先于文本子配置，不能静默覆盖调用方权重语义。"""
    transformers = pytest.importorskip("transformers", minversion="5.10.1")
    from mineru_vl_utils.transformers_loading import load_transformers_model

    config = transformers.Qwen2VLConfig(text_config={"tie_word_embeddings": True}, tie_word_embeddings=False)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda path: config)
    monkeypatch.setattr(transformers.PretrainedConfig, "get_config_dict", lambda path: ({"tie_word_embeddings": False}, {}))
    monkeypatch.setattr(transformers.Qwen2VLForConditionalGeneration, "from_pretrained", lambda path, **kwargs: config)
    assert load_transformers_model("model").tie_word_embeddings is False
