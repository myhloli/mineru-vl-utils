"""验证四类任务的本地防重复配置、实际适配参数与跨后端隔离。"""

from __future__ import annotations

import sys
from dataclasses import asdict
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from mineru_vl_utils import MinerUClient, MinerUSamplingParams
from mineru_vl_utils.mineru_client import DEFAULT_SAMPLING_PARAMS
from mineru_vl_utils.structs import ContentBlock
from mineru_vl_utils.vlm_client.base_client import SamplingParams, VlmClient
from mineru_vl_utils.vlm_client.http_client import HttpVlmClient

TARGET_KEYS = ("image", "chart", "equation", "[default]")
OTHER_KEYS = ("table", "[layout]", "[cross_page_table_merge]")


@pytest.fixture
def lmdeploy_types(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """使用公开配置类型和 Pipeline 替身验证分支选择，不加载 CUDA。"""

    class TurbomindEngineConfig:
        """模拟已解析的 TurboMind 配置类型。"""

        session_len = 8192

    class PytorchEngineConfig:
        """模拟已解析的 PyTorch 配置类型。"""

        session_len = 8192

    class Pipeline:
        """只提供适配器需要的实际后端配置。"""

        def __init__(self, backend_config: object) -> None:
            """保存传入的具体引擎配置，避免用平台或名称推断后端。"""
            self.backend_config = backend_config

    module = ModuleType("lmdeploy")
    module.GenerationConfig = SimpleNamespace
    module.TurbomindEngineConfig = TurbomindEngineConfig
    module.PytorchEngineConfig = PytorchEngineConfig
    pipeline_module = ModuleType("lmdeploy.pipeline")
    pipeline_module.Pipeline = Pipeline
    monkeypatch.setitem(sys.modules, "lmdeploy", module)
    monkeypatch.setitem(sys.modules, "lmdeploy.pipeline", pipeline_module)
    return SimpleNamespace(pipeline=Pipeline, turbomind=TurbomindEngineConfig, pytorch=PytorchEngineConfig)


def test_llama_defaults_reach_native_sampler_and_default_block_fallback() -> None:
    """四类与正文回退均传入 512 窗口，表格和布局等仍使用原始配置。"""
    # 基础安装不包含可选原生后端，仅跳过当前用例，保留其他后端的配置检查。
    Engine = pytest.importorskip("mineru_llama_cpp").Engine

    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=object.__new__(Engine), use_tqdm=False)
    for key in TARGET_KEYS:
        native = client.client.build_llama_cpp_sampling_params(client.sampling_params[key])
        assert native.repeat_last_n == 512
        assert native.repeat_penalty == 1.0
        assert native.presence_penalty == 1.0
        assert native.frequency_penalty == 0.05
        assert (native.temperature, native.top_k, native.top_p) == (0.0, 1, 0.01)
    for key in OTHER_KEYS:
        native = client.client.build_llama_cpp_sampling_params(client.sampling_params[key])
        assert native.repeat_last_n is None
        assert client.sampling_params[key] == DEFAULT_SAMPLING_PARAMS[key]

    for block_type in ("image", "chart", "equation", "text", "code", "title"):
        _, _, params, indices = client.helper.prepare_for_extract(
            Image.new("RGB", (100, 100)), [ContentBlock(block_type, [0, 0, 1, 1])], image_analysis=True
        )
        assert indices == [0]
        assert params[0].repeat_last_n == 512


@pytest.mark.parametrize("backend", ["turbomind", "pytorch"])
def test_lmdeploy_defaults_use_actual_backend_and_reach_generation_config(
    lmdeploy_types: SimpleNamespace, backend: str
) -> None:
    """只有真实 TurboMind 配置启用惩罚，并确保 top-1 不覆盖显式采样开关。"""
    config_type = lmdeploy_types.turbomind if backend == "turbomind" else lmdeploy_types.pytorch
    client = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=lmdeploy_types.pipeline(config_type()), use_tqdm=False)
    for key, params in client.sampling_params.items():
        config = client.client.build_lmdeploy_generation_config(params)
        if backend == "turbomind" and key in TARGET_KEYS:
            assert config.repetition_penalty == 1.05
            assert config.do_sample is True
            assert (config.top_k, config.top_p, config.temperature) == (1, 1.0, 1.0)
            assert config.random_seed == 0
        else:
            assert params == DEFAULT_SAMPLING_PARAMS[key]
            assert config.repetition_penalty == 1.0
            assert config.do_sample is False
            assert not hasattr(config, "random_seed")


def test_explicit_sampling_map_and_instances_remain_isolated(lmdeploy_types: SimpleNamespace) -> None:
    """显式配置完整覆盖默认策略，调用者、其他实例和模块默认值互不污染。"""
    pipeline = lmdeploy_types.pipeline(lmdeploy_types.turbomind())
    explicit = {"[default]": MinerUSamplingParams(repetition_penalty=1.0, do_sample=False, random_seed=7)}
    first = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=pipeline, sampling_params=explicit)
    second = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=pipeline)
    third = MinerUClient(backend="lmdeploy-engine", lmdeploy_engine=pipeline)
    assert set(first.sampling_params) == {"[default]"}
    assert first.helper.sampling_params is first.sampling_params
    config = first.client.build_lmdeploy_generation_config(first.sampling_params["[default]"])
    assert (config.repetition_penalty, config.do_sample, config.random_seed) == (1.0, False, 7)
    first.sampling_params["[default]"].random_seed = 99
    second.sampling_params["image"].repetition_penalty = 1.2
    assert explicit["[default]"].random_seed == 7
    assert third.sampling_params["image"].repetition_penalty == 1.05
    assert DEFAULT_SAMPLING_PARAMS["image"].repetition_penalty == 1.0
    assert DEFAULT_SAMPLING_PARAMS["image"].repeat_last_n is None


def test_sampling_merge_preserves_zero_false_and_inherited_values() -> None:
    """新增字段逐项合并，0 和 False 是有效覆盖值，None 则保留客户端设置。"""
    client = VlmClient(sampling_params=SamplingParams(repeat_last_n=512, do_sample=True, random_seed=7))
    assert client.build_sampling_params(None).repeat_last_n == 512
    merged = client.build_sampling_params(SamplingParams(repeat_last_n=0, do_sample=False, random_seed=0))
    assert (merged.repeat_last_n, merged.do_sample, merged.random_seed) == (0, False, 0)
    assert client.build_sampling_params(SamplingParams()).random_seed == 7
    assert VlmClient().build_sampling_params(None) == SamplingParams()


@pytest.mark.parametrize("explicit", [None, False, True])
def test_lmdeploy_sampling_switch_preserves_explicit_choice(lmdeploy_types: SimpleNamespace, explicit: bool | None) -> None:
    """显式开关优先，未提供时保留原有 temperature 与 top_k 推导。"""
    from mineru_vl_utils.vlm_client.lmdeploy_engine_client import LmdeployEngineVlmClient

    client = LmdeployEngineVlmClient(lmdeploy_types.pipeline(lmdeploy_types.turbomind()))
    for top_k in (1, 8):
        config = client.build_lmdeploy_generation_config(SamplingParams(temperature=1.0, top_k=top_k, do_sample=explicit))
        assert config.do_sample is (top_k > 1 if explicit is None else explicit)


@pytest.mark.parametrize("backend", ["http-client", "vllm-engine", "vllm-async-engine", "transformers", "mlx-engine"])
def test_other_backends_keep_existing_defaults(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    """其他后端的高层默认配置保持原样，不触发新增本地引擎导入。"""

    def new_client(**kwargs: Any) -> object:
        """替换底层模型加载，只检查真实高层构造器选择的采样配置。"""
        return object()

    monkeypatch.setattr("mineru_vl_utils.mineru_client.new_vlm_client", new_client)
    client = MinerUClient(backend=backend, model=object(), processor=object(), vllm_llm=object(), vllm_async_llm=object())
    assert client.sampling_params == DEFAULT_SAMPLING_PARAMS
    assert client.sampling_params is not DEFAULT_SAMPLING_PARAMS
    for params in client.sampling_params.values():
        assert (params.repeat_last_n, params.do_sample, params.random_seed) == (None, None, None)


def test_http_payload_keeps_vllm_parameters_and_excludes_engine_only_fields() -> None:
    """HTTP 继续发送 vLLM 参数；新增 engine 专属字段不会进入请求体。"""
    client = object.__new__(HttpVlmClient)
    client.sampling_params = None
    client.prompt = ""
    client.text_before_image = False
    client.model_name = "mineru"
    client.debug = False
    params = MinerUSamplingParams(
        presence_penalty=1.0, frequency_penalty=0.05, repeat_last_n=512, do_sample=True, random_seed=0
    )
    request = client.build_request_body("", [], "text", params, None, None)
    assert request == {
        "model": "mineru",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "text"}]}],
        "temperature": 0.0,
        "top_p": 0.01,
        "top_k": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.05,
        "repetition_penalty": 1.0,
        "vllm_xargs": {"no_repeat_ngram_size": 100, "debug": False},
        "skip_special_tokens": False,
    }
    assert asdict(DEFAULT_SAMPLING_PARAMS["image"])["repeat_last_n"] is None
