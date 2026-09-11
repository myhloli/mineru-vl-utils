"""Mock 单元测试：LlamaCppEngineVlmClient 的消息构造、参数改名映射、异常转换逻辑。

策略：object.__new__(Engine) 绕过 Engine.__init__ (会真的加载 GGUF 模型)，
用 unittest.mock 替换 generate()/agenerate()，不需要真实模型/GPU。
"""

import asyncio
import threading
from unittest.mock import MagicMock

import pytest
from mineru_llama_cpp import EngineError, GenerateResult, InvalidRequestError
from mineru_llama_cpp import Engine as LlamaCppEngine
from PIL import Image

from mineru_vl_utils.vlm_client.base_client import RequestError, SamplingParams, ServerError
from mineru_vl_utils.vlm_client.llama_cpp_engine_client import LlamaCppEngineVlmClient
from mineru_vl_utils.vlm_client import llama_cpp_engine_client


def _make_client(mock_engine: LlamaCppEngine) -> LlamaCppEngineVlmClient:
    c = LlamaCppEngineVlmClient(llama_cpp_engine=mock_engine)
    return c


@pytest.fixture
def mock_engine() -> LlamaCppEngine:
    """绕过 Engine.__init__（会真的加载 GGUF 模型），构造一个足以通过 isinstance 检查的空壳。"""
    return object.__new__(LlamaCppEngine)


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (4, 4), color="white")


def _stub_generate_result(content: str = "hello", finish_reason: str = "stop") -> GenerateResult:
    return GenerateResult(
        content=content,
        finish_reason=finish_reason,  # type: ignore[arg-type]
        tokens_evaluated=10,
        tokens_predicted=5,
        timings=None,
    )


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------


def test_build_messages_default_image_before_text(mock_engine):
    c = _make_client(mock_engine)
    messages = c.build_messages(["data:image/png;base64,AAAA"], "describe this")
    assert messages[0] == {"role": "system", "content": c.system_prompt}
    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    assert user_content[1] == {"type": "text", "text": "describe this"}


def test_build_messages_text_before_image(mock_engine):
    c = _make_client(mock_engine)
    c.text_before_image = True
    messages = c.build_messages(["data:image/png;base64,AAAA"], "describe this")
    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "text", "text": "describe this"}
    assert user_content[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_build_messages_placeholder_splits_multi_image(mock_engine):
    c = _make_client(mock_engine)
    image_urls = ["data:image/png;base64,AAAA", "data:image/png;base64,BBBB"]
    messages = c.build_messages(image_urls, "before<image>middle<image>after")
    user_content = messages[1]["content"]
    # split("<image>", maxsplit=2) on a prompt with exactly 2 occurrences
    # splits at both, yielding 3 parts ("before"/"middle"/"after") -- the
    # trailing "after" has no image to pair with and is kept as trailing
    # text, same behavior as http_client.py's build_request_body.
    assert user_content == [
        {"type": "text", "text": "before"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "text", "text": "middle"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
        {"type": "text", "text": "after"},
    ]


# ---------------------------------------------------------------------------
# build_llama_cpp_sampling_params
# ---------------------------------------------------------------------------


def test_sampling_params_field_renames(mock_engine):
    c = _make_client(mock_engine)
    sp = SamplingParams(
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        max_new_tokens=128,
        no_repeat_ngram_size=3,
    )
    llama_cpp_sp = c.build_llama_cpp_sampling_params(sp)
    assert llama_cpp_sp.temperature == 0.5
    assert llama_cpp_sp.top_p == 0.9
    assert llama_cpp_sp.top_k == 40
    assert llama_cpp_sp.repeat_penalty == 1.1  # renamed from repetition_penalty
    assert llama_cpp_sp.n_predict == 128  # renamed from max_new_tokens
    # no_repeat_ngram_size has no equivalent field and must not raise/appear
    assert not hasattr(llama_cpp_sp, "no_repeat_ngram_size") or llama_cpp_sp.no_repeat_ngram_size is None


def test_sampling_params_unset_fields_stay_unset(mock_engine):
    c = _make_client(mock_engine)
    llama_cpp_sp = c.build_llama_cpp_sampling_params(None)
    assert llama_cpp_sp.temperature is None
    assert llama_cpp_sp.n_predict is None


# ---------------------------------------------------------------------------
# get_output_content / finish_reason handling
# ---------------------------------------------------------------------------


def test_get_output_content_stop(mock_engine):
    c = _make_client(mock_engine)
    result = _stub_generate_result(content="done", finish_reason="stop")
    assert c.get_output_content(result) == "done"


def test_get_output_content_length_raises_by_default(mock_engine):
    c = _make_client(mock_engine)
    result = _stub_generate_result(finish_reason="length")
    with pytest.raises(RequestError):
        c.get_output_content(result)


def test_get_output_content_length_allowed(mock_engine):
    c = _make_client(mock_engine)
    c.allow_truncated_content = True
    result = _stub_generate_result(content="truncated", finish_reason="length")
    assert c.get_output_content(result) == "truncated"


# ---------------------------------------------------------------------------
# predict() / aio_predict() happy path + exception mapping
# ---------------------------------------------------------------------------


def test_predict_happy_path(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)
    monkeypatch.setattr(mock_engine, "generate", lambda messages, sp: _stub_generate_result("hi"))
    assert c.predict(image=image) == "hi"


def test_predict_maps_invalid_request_error(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    def _raise(messages, sp):
        raise InvalidRequestError("bad request")

    monkeypatch.setattr(mock_engine, "generate", _raise)
    with pytest.raises(RequestError):
        c.predict(image=image)


def test_predict_maps_engine_error(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    def _raise(messages, sp):
        raise EngineError("internal failure")

    monkeypatch.setattr(mock_engine, "generate", _raise)
    with pytest.raises(ServerError):
        c.predict(image=image)


def test_aio_predict_happy_path(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    async def _agenerate(messages, sp):
        return _stub_generate_result("hi async")

    monkeypatch.setattr(mock_engine, "agenerate", _agenerate)
    assert asyncio.run(c.aio_predict(image=image)) == "hi async"


def test_aio_predict_maps_context_exceeded_error(mock_engine, image, monkeypatch):
    """ContextExceededError is a subclass of InvalidRequestError -- must map the same way."""
    from mineru_llama_cpp import ContextExceededError

    c = _make_client(mock_engine)

    async def _raise(messages, sp):
        raise ContextExceededError("too long")

    monkeypatch.setattr(mock_engine, "agenerate", _raise)
    with pytest.raises(RequestError):
        asyncio.run(c.aio_predict(image=image))


# ---------------------------------------------------------------------------
# batch_predict() / aio_batch_predict()
# ---------------------------------------------------------------------------


def test_batch_predict_preserves_order(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)
    call_count = 0

    def _generate(messages, sp):
        nonlocal call_count
        call_count += 1
        # echo back which text prompt was embedded in this call
        text_part = next(p for p in messages[-1]["content"] if p["type"] == "text")
        return _stub_generate_result(content=text_part["text"])

    monkeypatch.setattr(mock_engine, "generate", _generate)
    results = c.batch_predict(images=[image, image, image], prompts=["a", "b", "c"])
    assert results == ["a", "b", "c"]
    assert call_count == 3


def test_aio_batch_predict_preserves_order(mock_engine, image, monkeypatch):
    c = _make_client(mock_engine)

    async def _agenerate(messages, sp):
        text_part = next(p for p in messages[-1]["content"] if p["type"] == "text")
        return _stub_generate_result(content=text_part["text"])

    monkeypatch.setattr(mock_engine, "agenerate", _agenerate)
    results = asyncio.run(c.aio_batch_predict(images=[image, image, image], prompts=["a", "b", "c"]))
    assert results == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_none_engine():
    with pytest.raises(ValueError):
        LlamaCppEngineVlmClient(llama_cpp_engine=None)


def test_constructor_rejects_wrong_type():
    with pytest.raises(ValueError):
        LlamaCppEngineVlmClient(llama_cpp_engine=object())


def test_batch_progress_updates_before_slow_first_request(mock_engine, image, monkeypatch):
    """首个请求等待进度更新后才完成，验证进度不受输入顺序阻塞且结果不串位。"""
    client = _make_client(mock_engine)
    progress_seen = threading.Event()
    bar = MagicMock()
    bar.__enter__.return_value = bar
    bar.update.side_effect = lambda count: progress_seen.set()
    progress_factory = MagicMock(return_value=bar)
    monkeypatch.setattr(llama_cpp_engine_client, "tqdm", progress_factory)

    def predict(image, prompt, params, priority):
        """模拟第一个请求比后续请求慢，并核验进度先于首项返回。"""
        if prompt == "first":
            assert progress_seen.wait(5), "Progress was blocked by the first request"
        return prompt

    monkeypatch.setattr(client, "predict", predict)
    assert client.batch_predict([image] * 3, ["first", "second", "third"]) == ["first", "second", "third"]
    progress_factory.assert_called_once_with(total=3, desc="VLM Predict", disable=False)
    assert [call.args for call in bar.update.call_args_list] == [(1,), (1,), (1,)]
    bar.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize("enabled", [False, True])
def test_client_factory_forwards_progress_switch(mock_engine, image, monkeypatch, capsys, enabled):
    """从公开客户端构造链传递开关，确认同步推理只在启用时输出进度。"""
    from mineru_vl_utils import MinerUClient

    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=mock_engine, use_tqdm=enabled)
    assert client.client.use_tqdm is enabled
    monkeypatch.setattr(mock_engine, "generate", lambda messages, sp: _stub_generate_result("ok"))
    assert client.client.batch_predict([image]) == ["ok"]
    assert ("VLM Predict" in capsys.readouterr().err) is enabled


def test_empty_batch_does_not_start_threads_or_progress(mock_engine, monkeypatch):
    """空输入不创建线程池或进度条。"""
    executor = MagicMock(side_effect=AssertionError("Unexpected executor"))
    progress = MagicMock(side_effect=AssertionError("Unexpected progress"))
    monkeypatch.setattr(llama_cpp_engine_client, "ThreadPoolExecutor", executor)
    monkeypatch.setattr(llama_cpp_engine_client, "tqdm", progress)
    assert _make_client(mock_engine).batch_predict([]) == []
    executor.assert_not_called()
    progress.assert_not_called()


def test_batch_failure_closes_progress_and_propagates(mock_engine, image, monkeypatch):
    """推理失败保持原异常，并退出进度条上下文。"""
    client = _make_client(mock_engine)
    bar = MagicMock()
    bar.__enter__.return_value = bar
    monkeypatch.setattr(llama_cpp_engine_client, "tqdm", MagicMock(return_value=bar))
    monkeypatch.setattr(client, "predict", MagicMock(side_effect=ServerError("inference failed")))
    with pytest.raises(ServerError, match="inference failed"):
        client.batch_predict([image])
    assert bar.__exit__.call_args.args[0] is ServerError
    bar.update.assert_not_called()


@pytest.mark.parametrize("use_async", [False, True])
def test_two_step_keeps_page_progress_and_shared_concurrency(mock_engine, monkeypatch, use_async):
    """两阶段提取共享布局和内容请求的并发上限，仅展示一个按页统计的进度条。"""
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock
    from mineru_vl_utils.vlm_client import utils

    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=mock_engine, max_concurrency=2)
    assert client.batching_mode == "concurrent"
    active = 0
    peak = 0
    calls = []
    bars = []

    async def predict(image, prompt="", sampling_params=None, priority=None):
        """记录布局和内容请求的总并发，使用不同延迟制造完成顺序差异。"""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        page = image.width
        calls.append((page, prompt))
        try:
            await asyncio.sleep(0.002 * (4 - page))
            return str(page) if prompt == client.prompts["[layout]"] else f"{page}:{prompt}"
        finally:
            active -= 1

    async def prepare_layout(executor, image):
        """保留页标识，避免测试依赖实际图像缩放。"""
        return image

    async def parse_layout(executor, text):
        """为每页构造两个内容块，核验页和块的对应关系。"""
        return [ContentBlock("text", [0, 0, 1, 0.5]), ContentBlock("text", [0, 0.5, 1, 1])]

    async def prepare_extract(executor, image, layout, not_extract_list, image_analysis):
        """将每页的两个块映射到带有页标识的模拟内容请求。"""
        return [image, image], ["a", "b"], [None, None], [0, 1]

    async def post_process(executor, layout):
        """返回已填充内容的块，隔离与本次调度无关的后处理。"""
        return layout

    def progress_factory(**kwargs):
        """记录每层进度条的启用状态和完成计数。"""
        bar = MagicMock()
        bar.__enter__.return_value = bar
        bars.append((kwargs, bar))
        return bar

    monkeypatch.setattr(client.client, "aio_predict", predict)
    monkeypatch.setattr(client.helper, "aio_prepare_for_layout", prepare_layout)
    monkeypatch.setattr(client.helper, "aio_parse_layout_output", parse_layout)
    monkeypatch.setattr(client.helper, "aio_prepare_for_extract", prepare_extract)
    monkeypatch.setattr(client.helper, "aio_post_process", post_process)
    monkeypatch.setattr(utils, "tqdm", progress_factory)
    monkeypatch.setattr(llama_cpp_engine_client, "tqdm", MagicMock(side_effect=AssertionError("Nested sync progress")))
    images = [Image.new("RGB", (page, 4)) for page in (1, 2, 3)]
    results = asyncio.run(client.aio_batch_two_step_extract(images)) if use_async else client.batch_two_step_extract(images)
    assert [[block.content for block in page] for page in results] == [[f"{p}:a", f"{p}:b"] for p in (1, 2, 3)]
    assert peak == 2
    assert len(calls) == 9
    enabled_bars = [(options, bar) for options, bar in bars if not options["disable"]]
    assert len(enabled_bars) == 1
    options, bar = enabled_bars[0]
    assert options == {"total": 3, "desc": "Two Step Extraction", "disable": False}
    assert sum(call.args[0] for call in bar.update.call_args_list) == 3
