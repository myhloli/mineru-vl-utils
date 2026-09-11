"""验证各后端进度所有权、异步开关隔离及真实 HTTP 客户端的完成计数。"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.vlm_client import utils
from mineru_vl_utils.vlm_client.base_client import VlmClient
from mineru_vl_utils.vlm_client.mlx_client import MlxVlmClient
from mineru_vl_utils.vlm_client.transformers_client import TransformersVlmClient
from mineru_vl_utils.vlm_client.vllm_async_engine_client import VllmAsyncEngineVlmClient


@pytest.fixture
def bars(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, Any], MagicMock]]:
    """记录进度参数、完成次数与上下文关闭，保留实际请求调度。"""
    records = []

    def factory(**kwargs: Any) -> MagicMock:
        """为每次调用创建独立进度对象，避免并发断言共享状态。"""
        bar = MagicMock()
        bar.__enter__.return_value = bar
        records.append((kwargs, bar))
        return bar

    monkeypatch.setattr(utils, "tqdm", factory)
    monkeypatch.setattr("mineru_vl_utils.vlm_client.mlx_client.tqdm", factory)
    monkeypatch.setattr("mineru_vl_utils.vlm_client.transformers_client.tqdm", factory)
    return records


@pytest.fixture(params=[MlxVlmClient, TransformersVlmClient])
def local_client(request: pytest.FixtureRequest) -> VlmClient:
    """仅替换实际模型生成，运行真实同步分批和异步线程包装。"""
    client = object.__new__(request.param)
    VlmClient.__init__(client)
    client.use_tqdm = True
    client.batch_size = 1
    if isinstance(client, MlxVlmClient):
        client._predict_batch = lambda images, prompts, params: prompts
    else:
        client.processor = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: "recognized")
        client._predict_one_batch = lambda **kwargs: kwargs["chat_prompts"]
    return client


@pytest.mark.parametrize("enabled", [False, True])
def test_local_sync_and_async_progress(local_client: VlmClient, bars: list, enabled: bool) -> None:
    """实例与调用级开关故意取反，确认同步和异步各自遵守正确配置。"""
    local_client.use_tqdm = not enabled
    assert len(local_client.batch_predict([None, None], ["a", "b"])) == 2
    assert bars[-1][0] == {"total": 2, "desc": "Predict", "disable": enabled}
    assert (
        len(
            asyncio.run(
                local_client.aio_batch_predict(
                    [None, None],
                    ["a", "b"],
                    use_tqdm=enabled,
                    tqdm_desc="External Layout Extraction",
                )
            )
        )
        == 2
    )
    assert bars[-1][0] == {"total": 2, "desc": "External Layout Extraction", "disable": not enabled}
    assert sum(call.args[0] for call in bars[-1][1].update.call_args_list) == 2
    assert local_client.use_tqdm is not enabled
    bars[-1][1].__exit__.assert_called_once()


def test_local_concurrent_progress_is_call_local(local_client: VlmClient, bars: list) -> None:
    """用屏障让两条线程同时生成，确认不同开关和描述不会互相污染。"""
    barrier = threading.Barrier(2)

    def generate(*args: Any, **kwargs: Any) -> list[str]:
        """屏障期间实例配置必须保持原值。"""
        assert local_client.use_tqdm is True
        barrier.wait(timeout=5)
        return ["ok"]

    if isinstance(local_client, MlxVlmClient):
        local_client._predict_batch = generate
    else:
        local_client._predict_one_batch = generate

    async def run() -> list:
        """同一实例并发执行显式开启和关闭两种调用。"""
        return await asyncio.gather(
            *[local_client.aio_batch_predict([None], use_tqdm=enabled, tqdm_desc=str(enabled)) for enabled in (True, False)]
        )

    assert asyncio.run(run()) == [["ok"], ["ok"]]
    assert {(opts["desc"], opts["disable"]) for opts, _ in bars} == {("True", False), ("False", True)}
    assert local_client.use_tqdm is True


def test_local_empty_and_failure_progress(local_client: VlmClient, bars: list) -> None:
    """空批不展示，模型异常时进度收尾且不虚增完成数。"""
    assert local_client.batch_predict([]) == []
    assert asyncio.run(local_client.aio_batch_predict([], use_tqdm=True)) == []
    assert not bars
    failure = MagicMock(side_effect=RuntimeError("generation failed"))
    if isinstance(local_client, MlxVlmClient):
        local_client._predict_batch = failure
    else:
        local_client._predict_one_batch = failure
    with pytest.raises(RuntimeError, match="generation failed"):
        local_client.batch_predict([None])
    assert bars[-1][1].__exit__.call_args.args[0] is RuntimeError
    bars[-1][1].update.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_http_factory_sync_progress_and_early_completion(monkeypatch: pytest.MonkeyPatch, bars: list, enabled: bool) -> None:
    """通过真实 HTTP 序列化和响应解析，验证慢首请求不会挡住其他请求的进度。"""
    client = MinerUClient(
        backend="http-client",
        server_url="http://progress.test",
        model_name="test",
        skip_model_name_checking=True,
        use_tqdm=enabled,
    )
    assert client.client.use_tqdm is enabled

    async def handle(request: httpx.Request) -> httpx.Response:
        """第二请求先完成，首请求等待进度发生；无需模型服务器或网络。"""
        if b"slow" in request.content:
            for _ in range(100):
                if bars and bars[-1][1].update.called:
                    break
                await asyncio.sleep(0.001)
            assert bars[-1][1].update.called, "Progress waited for the slow first request"
            text = "slow"
        else:
            text = "fast"
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})

    transport = httpx.MockTransport(handle)
    async_client = httpx.AsyncClient(transport=transport)

    async def get_client() -> httpx.AsyncClient:
        """注入传输层，保留真实 aio_predict 和 gather_tasks。"""
        return async_client

    monkeypatch.setattr(client.client, "_aio_client", get_client)
    try:
        assert client.client.batch_predict([None, None], ["slow", "fast"]) == ["slow", "fast"]
        opts, bar = bars[-1]
        assert opts == {"total": 2, "desc": "VLM Predict", "disable": not enabled}
        assert sum(call.args[0] for call in bar.update.call_args_list) == 2
        bar.__exit__.assert_called_once()
    finally:
        asyncio.run(async_client.aclose())
        client.client._client.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_async_vllm_external_layout_progress(monkeypatch: pytest.MonkeyPatch, bars: list, enabled: bool) -> None:
    """运行高层外部布局抽取，验证异步 vLLM 只统计实际内容请求。"""
    from PIL import Image
    from mineru_vl_utils.structs import ContentBlock

    backend = object.__new__(VllmAsyncEngineVlmClient)
    backend.max_concurrency = 2

    async def predict(image: Any, prompt: str = "", **kwargs: Any) -> str:
        """替代 GPU 单次生成，保留后端的批量并发实现。"""
        return "text"

    monkeypatch.setattr(backend, "aio_predict", predict)
    monkeypatch.setattr("mineru_vl_utils.mineru_client.new_vlm_client", lambda **kwargs: backend)
    client = MinerUClient(backend="vllm-async-engine", vllm_async_llm=object(), use_tqdm=enabled)
    image = Image.new("RGB", (32, 32))
    blocks = [ContentBlock("text", [0, 0, 1, 0.5]), ContentBlock("image", [0, 0.5, 1, 1])]
    try:
        result = asyncio.run(client.aio_batch_extract_with_layout([image], [blocks], image_analysis=False))
        assert result[0][0].content == "text"
        extraction_bars = [(opts, bar) for opts, bar in bars if opts["desc"] == "External Layout Extraction"]
        assert len(extraction_bars) == 1
        opts, bar = extraction_bars[0]
        assert opts == {"total": 1, "desc": "External Layout Extraction", "disable": not enabled}
        bar.update.assert_called_once_with(1)
        bars.clear()
        asyncio.run(client.aio_batch_extract_with_layout([image], [[]]))
        assert not any(opts["desc"] == "External Layout Extraction" for opts, _ in bars)
    finally:
        image.close()


def test_gather_empty_failure_and_cancellation(bars: list) -> None:
    """空任务静默；错误或取消后关闭进度并等待子任务清理。"""
    assert asyncio.run(utils.gather_tasks([], use_tqdm=True)) == []
    assert not bars

    async def run(cancel: bool) -> None:
        """用事件确保在途任务存在后再触发错误或取消。"""
        entered, cleaned = asyncio.Event(), asyncio.Event()

        async def pending() -> None:
            """记录被取消任务的 finally 清理已执行。"""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        async def fail() -> None:
            """等另一任务启动后报错，检查聚合层会取消在途任务。"""
            await entered.wait()
            raise RuntimeError("failed")

        task = asyncio.create_task(utils.gather_tasks([pending()] if cancel else [pending(), fail()], use_tqdm=True))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
            await task
        assert cleaned.is_set()

    for cancel in (False, True):
        asyncio.run(run(cancel))
        assert bars[-1][1].__exit__.call_args.args[0] is (asyncio.CancelledError if cancel else RuntimeError)
        bars[-1][1].update.assert_not_called()
