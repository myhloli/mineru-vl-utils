"""用真实模型验证修复后的原生 UTF-8 空前缀在普通批量中仍计作成功。"""

import asyncio
import os

import pytest

from mineru_vl_utils.vlm_client.llama_cpp_engine_client import LlamaCppEngineVlmClient

native = pytest.importorskip("mineru_llama_cpp")


class ByteTokenSampling(native.SamplingParams):
    """只在集成测试中通过采样偏置确定性制造部分中文字符。"""

    def __init__(self):
        """强制只生成中文字符的首个字节 token，并在一个 token 后停止。"""
        super().__init__(n_predict=1, temperature=0.0, top_k=1, seed=42)

    def to_json_fields(self):
        """沿正式请求路径注入测试偏置，保留默认 Unicode grammar。"""
        fields = super().to_json_fields()
        fields["logit_bias"] = {"160": 1000.0}
        return fields


@pytest.fixture(scope="module")
def native_engine():
    """仅在显式指定本地模型时启动真实原生引擎，普通测试不下载模型。"""
    model = os.getenv("MINERU_LLAMA_CPP_TEST_MODEL")
    projector = os.getenv("MINERU_LLAMA_CPP_TEST_MMPROJ")
    if not model or not projector:
        pytest.skip("set local MinerU Q8_0 model and mmproj paths to run native integration")
    with native.Engine(model, projector, n_ctx_seq=512, n_gpu_layers=99, n_parallel=2) as engine:
        yield engine


@pytest.mark.parametrize("use_async", [False, True])
def test_fixed_native_empty_prefix_is_success(native_engine, monkeypatch, use_async):
    """整批合法空前缀应正常返回，不能被误判为全部识别失败。"""
    client = LlamaCppEngineVlmClient(
        native_engine,
        allow_truncated_content=True,
        isolate_block_errors=True,
        use_tqdm=False,
    )
    monkeypatch.setattr(client, "build_llama_cpp_sampling_params", lambda params: ByteTokenSampling())
    if use_async:
        outputs = asyncio.run(client.aio_batch_predict([None, None], ["输出中文。", "输出中文。"]))
    else:
        outputs = client.batch_predict([None, None], ["输出中文。", "输出中文。"])
    assert outputs == ["", ""]
