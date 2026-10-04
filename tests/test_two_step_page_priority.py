"""回归测试：页面级 priority 列表在两步提取的块展开批处理中必须按页展开。

历史缺陷：stepping_two_step_extract / aio_stepping_two_step_extract /
batch_content_extract / aio_batch_content_extract 把长度为页数的 priority
列表直接传给按 block 展开的批量预测，客户端断言
`len(priority) == len(images)` 在任一页面产出多个 block 时必然失败
（真实文档几乎总是多块）。batch_extract_with_layout 通过
_expand_block_priorities 走对了，四条 two-step 路径漏了同样的展开。
"""

import asyncio

import pytest
from mineru_llama_cpp import Engine as LlamaCppEngine
from mineru_llama_cpp import GenerateResult
from PIL import Image

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import ContentBlock, ExtractResult


@pytest.fixture
def mock_engine() -> LlamaCppEngine:
    engine = object.__new__(LlamaCppEngine)
    engine.generate = lambda messages, sp: GenerateResult(
        content="content", finish_reason="stop", tokens_evaluated=1, tokens_predicted=1, timings=None
    )
    return engine


@pytest.fixture
def image() -> Image.Image:
    return Image.new("RGB", (32, 32), color="white")


def _fake_layout(images, blocks_per_page):
    def fake_layout(pri=None, sem=None, sco=None):
        return [ExtractResult([ContentBlock("text", [0.0, 0.0, 1.0, 1.0]) for _ in range(n)], None) for n in blocks_per_page]

    async def aio_fake(self, imgs, priority=None, semaphore=None, scored=None):
        return fake_layout(imgs, blocks_per_page)

    return aio_fake


def test_stepping_two_step_expands_page_priority(mock_engine, image, monkeypatch):
    """页级 priority 列表 + 多块页面：不得触发客户端长度断言，且各块都能提取。"""
    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=mock_engine, max_concurrency=2)
    client.batching_mode = "stepping"
    monkeypatch.setattr(MinerUClient, "aio_batch_layout_detect", _fake_layout(image, [2, 1]))

    results = asyncio.run(client.aio_batch_two_step_extract(images=[image, image], priority=[7, 9]))
    assert [len(r) for r in results] == [2, 1]
    assert all(block.content == "content" for r in results for block in r)


def test_batch_content_extract_expands_page_priority(mock_engine, image, monkeypatch):
    """外部 layout 的批量内容提取同样必须展开页级 priority。"""
    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=mock_engine, max_concurrency=2)
    blocks_per_page = [
        [ContentBlock("text", [0.0, 0.0, 1.0, 1.0]) for _ in range(2)],
        [ContentBlock("text", [0.0, 0.0, 1.0, 1.0])],
    ]
    results = client.batch_extract_with_layout([image, image], blocks_per_page, priority=[7, 9])
    assert [len(r) for r in results] == [2, 1]
    assert all(block.content == "content" for r in results for block in r)


def test_two_step_scalar_priority_unchanged(mock_engine, image):
    """标量 priority（默认路径）行为保持不变。"""
    client = MinerUClient(backend="llama-cpp-engine", llama_cpp_engine=mock_engine, max_concurrency=2)
    client.batching_mode = "stepping"
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(MinerUClient, "aio_batch_layout_detect", _fake_layout(image, [2, 1]))
        results = asyncio.run(client.aio_batch_two_step_extract(images=[image, image]))
        assert [len(r) for r in results] == [2, 1]
    finally:
        monkeypatch.undo()
