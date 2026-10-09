"""验证两步提取与内容提取按实际参与识别的块映射页面优先级。"""

import asyncio
from collections.abc import Sequence

import pytest
from PIL import Image

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.structs import ContentBlock, ExtractResult


@pytest.fixture
def image():
    """生成无需外部文件的页面图像。"""
    return Image.new("RGB", (32, 32), "white")


@pytest.fixture
def recording_client(monkeypatch):
    """模拟 HTTP 批量推理，记录底层实际收到的块级优先级。"""
    client = MinerUClient(
        backend="http-client", server_url="http://test", model_name="test",
        skip_model_name_checking=True, use_tqdm=False,
    )
    calls = []

    def predict(images, prompts="", sampling_params=None, priority=None, **kwargs):
        """保留低层长度约束，并为每个输入返回可检查的文本。"""
        if isinstance(priority, Sequence):
            assert len(priority) == len(images)
            priority = list(priority)
        calls.append(priority)
        return ["content"] * len(images)

    async def aio_predict(*args, **kwargs):
        """异步入口复用同一记录逻辑。"""
        return predict(*args, **kwargs)

    monkeypatch.setattr(client.client, "batch_predict", predict)
    monkeypatch.setattr(client.client, "aio_batch_predict", aio_predict)
    yield client, calls
    asyncio.run(client.client.aclose())


def _install_layout(monkeypatch, client, types_per_page):
    """注入新建的布局块，避免后处理对后续测试数据产生影响。"""
    def layout(images, priority=None, scored=None):
        """按给定块类型生成每页布局结果。"""
        return [ExtractResult([ContentBlock(kind, [0, 0, 1, 1]) for kind in types]) for types in types_per_page]

    async def aio_layout(images, priority=None, semaphore=None, scored=None):
        """异步布局返回与同步布局一致的新对象。"""
        return layout(images, priority, scored)

    monkeypatch.setattr(client, "batch_layout_detect", layout)
    monkeypatch.setattr(client, "aio_batch_layout_detect", aio_layout)


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("priority,incremental,expected", [
    ([7, 9], False, [7, 7, 9]), (None, True, [0, 0, 1]),
    (7, False, 7), (None, False, None),
])
def test_stepping_expands_actual_block_priorities(recording_client, image, monkeypatch, use_async, priority, incremental, expected):
    """同步和异步多块页面均传递正确的列表、自动或标量优先级。"""
    client, calls = recording_client
    client.incremental_priority = incremental
    _install_layout(monkeypatch, client, [["text", "text"], ["text"]])
    if use_async:
        results = asyncio.run(client.aio_stepping_two_step_extract([image, image], priority=priority))
    else:
        results = client.stepping_two_step_extract([image, image], priority=priority)
    assert calls == [expected]
    assert [len(page) for page in results] == [2, 1]
    assert all(block.content == "content" for page in results for block in page)


@pytest.mark.parametrize("use_async", [False, True])
def test_stepping_skipped_page_keeps_remaining_priority(recording_client, image, monkeypatch, use_async):
    """不参与识别的图片页面不能导致后续文本块使用错误页优先级。"""
    client, calls = recording_client
    _install_layout(monkeypatch, client, [["image"], ["text"]])
    if use_async:
        results = asyncio.run(client.aio_stepping_two_step_extract([image, image], priority=[7, 9]))
    else:
        results = client.stepping_two_step_extract([image, image], priority=[7, 9])
    assert calls == [[9]]
    assert results[1][0].content == "content"


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("types,expected,result", [
    (["image", "text"], [9], [None, "content"]),
    (["image", "image"], [], [None, None]),
])
def test_batch_content_extract_filters_page_priorities(recording_client, image, use_async, types, expected, result):
    """直接调用实际内容提取接口，覆盖部分及全部页面被过滤的情形。"""
    client, calls = recording_client
    if use_async:
        outputs = asyncio.run(client.aio_batch_content_extract([image, image], types=types, priority=[7, 9]))
    else:
        outputs = client.batch_content_extract([image, image], types=types, priority=[7, 9])
    assert calls == [expected]
    assert outputs == result


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", ["stepping", "content"])
def test_empty_batch_priority(recording_client, monkeypatch, use_async, mode):
    """空页面列表和空优先级列表正常返回空结果。"""
    client, calls = recording_client
    _install_layout(monkeypatch, client, [])
    if mode == "stepping":
        method = client.aio_stepping_two_step_extract if use_async else client.stepping_two_step_extract
    else:
        method = client.aio_batch_content_extract if use_async else client.batch_content_extract
    outputs = method([], priority=[])
    if use_async:
        outputs = asyncio.run(outputs)
    assert outputs == []
    assert calls == [[]]


def test_external_layout_existing_priority_mapping(recording_client, image):
    """外部布局接口作为已有行为的对照，不计作本次修复的失败复现。"""
    client, calls = recording_client
    blocks = [[ContentBlock("text", [0, 0, 1, 1]) for _ in range(2)], [ContentBlock("text", [0, 0, 1, 1])]]
    results = client.batch_extract_with_layout([image, image], blocks, priority=[7, 9])
    assert calls == [[7, 7, 9]]
    assert [len(page) for page in results] == [2, 1]
