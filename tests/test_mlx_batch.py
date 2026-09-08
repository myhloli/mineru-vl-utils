from __future__ import annotations

import json
import sys
from io import BytesIO
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from mineru_vl_utils.vlm_client.base_client import SamplingParams, ServerError, UnsupportedError
from mineru_vl_utils.vlm_client.mlx_client import MlxVlmClient


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> MlxVlmClient:
    """用公开接口替身验证分组、参数和锁，不要求 CI 安装 Metal。"""
    calls: list[dict[str, Any]] = []

    def template(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """用可逆格式检查 system 和媒体内容未丢失。"""
        return json.dumps(messages)

    def generate(**kwargs: Any) -> Any:
        """记录单样本路径，输出对应的文本。"""
        assert result._generation_lock.locked()
        calls.append({"single": kwargs})
        content = json.loads(kwargs["prompt"])[-1]["content"]
        return SimpleNamespace(text="".join(p.get("text", "") for p in content))

    class BatchGenerator:
        def __init__(self, model: Any, processor: Any, **kwargs: Any) -> None:
            """保存调度配置并记录关闭状态。"""
            self.call = kwargs
            self.closed = False
            self.has_work = True
            calls.append(self.call)

        def insert(self, tokens: Any, prompt_kwargs: Any, logits_processors: Any) -> list[int]:
            """使用非连续请求 ID，检查调用方不会把 ID 当作数组索引。"""
            self.call.update(prompts=[p["messages"] for p in prompt_kwargs], logits_processors=logits_processors)
            images = [p["image"] for p in prompt_kwargs]
            self.call["images"] = images if images[0] is not None else None
            self.uids = [11 + i * 3 for i in range(len(tokens))]
            return self.uids

        def next(self) -> tuple[list[Any], list[Any]]:
            """倒序完成请求，验证结果仍按原始顺序返回。"""
            assert result._generation_lock.locked()
            self.has_work = False
            responses = []
            for uid, messages in reversed(list(zip(self.uids, self.call["prompts"]))):
                text = "".join(p.get("text", "") for p in messages[-1]["content"])
                responses.extend(
                    [
                        SimpleNamespace(uid=uid, token=text, finish_reason=None),
                        SimpleNamespace(uid=uid, token="EOS", finish_reason="stop"),
                    ]
                )
            return [], responses

        def close(self) -> None:
            """记录异常和成功路径均关闭调度器。"""
            self.closed = True
            self.call["closed"] = True

    class Detokenizer:
        def reset(self) -> None:
            """每条样本清空解码状态。"""
            self.text = ""

        def add_token(self, token: str) -> None:
            """用文本片段模拟 token，确保停止标记未混入输出。"""
            self.text += token

        def finalize(self) -> None:
            """模拟解码完成。"""

    def prepare(image: Any, prompt: str) -> tuple[list[int], dict[str, Any]]:
        """模拟视觉特征，但检查原始模板生成处于同一把锁内。"""
        assert result._generation_lock.locked()
        return [1], {"messages": result.build_messages(prompt, image is not None), "image": image}

    def make_processors(**kwargs: Any) -> list[Any]:
        """为每条序列创建独立对象，检测状态是否被跨样本共享。"""
        return [dict(kwargs)]

    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(generate=generate))
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", SimpleNamespace(BatchGenerator=BatchGenerator))
    monkeypatch.setitem(
        sys.modules, "mlx_vlm.prompt_utils", SimpleNamespace(apply_chat_template=lambda p, c, m, **kw: template(m))
    )
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.sample_utils",
        SimpleNamespace(make_sampler=lambda **kw: kw, make_logits_processors=make_processors),
    )
    model = SimpleNamespace(
        language_model=object(), config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=2048))
    )
    result = MlxVlmClient(
        model, SimpleNamespace(apply_chat_template=template, detokenizer=Detokenizer()), use_tqdm=False, batch_size=2
    )
    result._prepare_batch_prompt = prepare
    result.calls = calls
    return result


def test_grouping_restores_input_order_and_sampling(client: MlxVlmClient) -> None:
    """非相邻的同配置样本可以合批，不同配置不能串用，并按原索引返回。"""
    image = Image.new("RGB", (100, 100))
    a = SamplingParams(temperature=0.2, top_p=0.9, top_k=4, max_new_tokens=80, presence_penalty=1.0)
    b = SamplingParams(temperature=0.0, max_new_tokens=20)
    texts = client.batch_predict([image] * 4, ["a", "b", "c", "d"], [a, b, a, b])
    assert texts == ["a", "b", "c", "d"]
    assert [call["max_tokens"] for call in client.calls] == [80, 20]
    first = client.calls[0]
    assert first["sampler"] == {"temp": 0.2, "top_p": 0.9, "top_k": 4}
    assert first["prompts"][0][0] == {"role": "system", "content": client.system_prompt}
    assert first["logits_processors"][0] is not first["logits_processors"][1]
    assert first["logits_processors"][0][0] is not first["logits_processors"][1][0]
    assert first["logits_processors"][0][0]["presence_penalty"] == 1.0


def test_chunk_limit_and_tail(client: MlxVlmClient) -> None:
    """显式小批量为二，尾部单样本复用原始生成路径。"""
    assert client.batch_size == 2
    assert client.batch_predict([None] * 5, ["0", "1", "2", "3", "4"]) == ["0", "1", "2", "3", "4"]
    assert [len(c["prompts"]) for c in client.calls if "prompts" in c] == [2, 2]
    assert "single" in client.calls[-1]


def test_mixed_image_shapes_and_text_only(client: MlxVlmClient) -> None:
    """图片与纯文本分组，图片按大小聚合，防止顺序与媒体错配。"""
    small, large = Image.new("RGB", (10, 10)), Image.new("RGB", (20, 20))
    inputs = [small, None, large, small, None, large]
    texts = [str(i) for i in range(6)]
    assert client.batch_predict(inputs, texts) == texts
    assert [None if c["images"] is None else c["images"][0].size for c in client.calls] == [(10, 10), (20, 20), None]


def test_pixel_budget_preserves_resolution(client: MlxVlmClient) -> None:
    """超过同批像素预算时拆分，绝不通过缩小原图规避预算。"""
    client.batch_size = 4
    image = Image.new("RGB", (2250, 2250))
    assert client.batch_predict([image] * 2, ["a", "b"]) == ["a", "b"]
    assert all(c["single"]["image"].size == (2250, 2250) for c in client.calls)
    assert image.size == (2250, 2250)


def test_explicit_one_keeps_original_path(client: MlxVlmClient) -> None:
    """显式 batch_size=1 继续逐张生成，可用于回退对照。"""
    client.batch_size = 1
    assert client.batch_predict([None] * 2, ["a", "b"]) == ["a", "b"]
    assert all("single" in c for c in client.calls)


@pytest.mark.parametrize("field", ["prompts", "sampling_params", "priority"])
def test_lengths_validated_before_inference(client: MlxVlmClient, field: str) -> None:
    """错误长度在推理前抛出，不产生部分输出或静默截断。"""
    with pytest.raises(ValueError, match=field):
        client.batch_predict([None, None], **{field: []})
    assert client.calls == []


def test_empty_and_multi_image_inputs(client: MlxVlmClient) -> None:
    """空批次无推理；保持不支持单样本多图的边界。"""
    assert client.batch_predict([]) == []
    with pytest.raises(UnsupportedError):
        client.batch_predict([[Image.new("RGB", (2, 2))]])
    assert not client.calls


def test_bytes_and_paths_are_materialized(client: MlxVlmClient, tmp_path: Any) -> None:
    """复用资源读取规则，解码后的图片在批调用期间保持有效。"""
    buffer = BytesIO()
    Image.new("RGB", (5, 5), "red").save(buffer, format="PNG")
    path = tmp_path / "sample.png"
    path.write_bytes(buffer.getvalue())
    assert client.batch_predict([buffer.getvalue(), str(path)], ["a", "b"]) == ["a", "b"]
    assert all(im.getpixel((0, 0)) == (255, 0, 0) for im in client.calls[0]["images"])


def test_explicit_image_position_preserved(client: MlxVlmClient) -> None:
    """显式图文顺序直接交给既有模板，不经上游格式重排。"""
    image = Image.new("RGB", (10, 10))
    assert client.batch_predict([image] * 2, "before<image>after") == ["beforeafter"] * 2
    assert [p["type"] for p in client.calls[0]["prompts"][0][-1]["content"]] == ["text", "image", "text"]


def test_native_failure_propagates_and_unlocks(client: MlxVlmClient) -> None:
    """原生失败不得伪装成成功或自动重试，且必须归还生成锁。"""
    native = client.batch_generator_type

    def fail(*args: Any, **kwargs: Any) -> Any:
        """模拟上游内存或推理错误。"""
        raise RuntimeError("native failure")

    client.batch_generator_type = fail
    with pytest.raises(RuntimeError, match="native failure"):
        client.batch_predict([None] * 2)
    assert not client._generation_lock.locked()
    client.batch_generator_type = native
    assert len(client.batch_predict([None] * 2)) == 2


def test_incomplete_response_is_rejected_and_closed(client: MlxVlmClient) -> None:
    """调度器未产生完整结果时抛错，同时释放上游资源。"""
    native = client.batch_generator_type

    class Incomplete(native):
        def next(self) -> tuple[list[Any], list[Any]]:
            """模拟上游提前结束而未发送完成标记。"""
            self.has_work = False
            return [], []

    client.batch_generator_type = Incomplete
    with pytest.raises(ServerError, match="incomplete"):
        client.batch_predict([None] * 2)
    assert client.calls[0]["closed"] is True
    assert not client._generation_lock.locked()


def test_variable_shapes_can_share_batch(client: MlxVlmClient) -> None:
    """不同尺寸允许按相近面积同批，结果仍对应原始顺序。"""
    images = [Image.new("RGB", (30, 30)), Image.new("RGB", (20, 10))]
    assert client.batch_predict(images, ["large", "small"]) == ["large", "small"]
    assert [im.size for im in client.calls[0]["images"]] == [(20, 10), (30, 30)]


def test_single_fallback_keeps_execution_order(client: MlxVlmClient) -> None:
    """batch 一不按参数重新排序，保持原有逐张执行顺序。"""
    client.batch_size = 1
    client.batch_predict(
        [None] * 3,
        ["a", "b", "c"],
        [SamplingParams(max_new_tokens=20), SamplingParams(max_new_tokens=30), SamplingParams(max_new_tokens=20)],
    )
    assert [c["single"]["max_tokens"] for c in client.calls] == [20, 30, 20]


def test_effective_sampling_inherits_client_defaults(client: MlxVlmClient) -> None:
    """先合并客户端默认值再分组，部分覆盖不丢失未指定的采样配置。"""
    client.sampling_params = SamplingParams(temperature=0.3, top_p=0.8, max_new_tokens=50, frequency_penalty=0.2)
    client.batch_predict([None] * 2, ["a", "b"], [None, SamplingParams(temperature=0.3)])
    assert len(client.calls) == 1
    assert client.calls[0]["max_tokens"] == 50
    assert client.calls[0]["sampler"] == {"temp": 0.3, "top_p": 0.8, "top_k": 0}
    assert client.calls[0]["logits_processors"][0][0]["frequency_penalty"] == 0.2


@pytest.mark.parametrize("requested,expected", [(0, 8), (1, 1), (2, 2), (4, 4), (8, 8)])
def test_default_and_explicit_batch_sizes(client: MlxVlmClient, requested: int, expected: int) -> None:
    """自动档采用八，显式较小批次仍可用于降低内存占用或回退。"""
    configured = MlxVlmClient(client.model, client.processor, batch_size=requested, use_tqdm=False)
    assert configured.batch_size == expected


@pytest.mark.parametrize("batch_size,expected", [(4, [4, 4]), (8, [8])])
def test_layout_batches_fit_pixel_budget(client: MlxVlmClient, batch_size: int, expected: list[int]) -> None:
    """真实 layout 尺寸可完整容纳四张或八张，不再被像素预算拆小。"""
    client.batch_size = batch_size
    image = Image.new("RGB", (1036, 1036))
    prompts = [str(i) for i in range(8)]
    assert client.batch_predict([image] * 8, prompts) == prompts
    assert [len(call["prompts"]) for call in client.calls] == expected


def test_pixel_budget_still_caps_large_requested_batch(client: MlxVlmClient) -> None:
    """即使用户请求更大批次，第九张 layout 仍被像素预算拆开。"""
    client.batch_size = 16
    image = Image.new("RGB", (1036, 1036))
    assert list(client._iter_batches(list(range(9)), [image] * 9)) == [list(range(8)), [8]]
