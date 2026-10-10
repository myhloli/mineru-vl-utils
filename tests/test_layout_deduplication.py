"""验证 VLM 重复版面框在裁图识别之前消除，并保留合法容器关系。"""

import asyncio
from collections.abc import Iterator, Sequence
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from mineru_vl_utils import MinerUClient
from mineru_vl_utils.mineru_client import MinerUClientHelper
from mineru_vl_utils.structs import BLOCK_TYPES, ContentBlock, ExtractResult
from mineru_vl_utils.vlm_client.base_client import ImageType, SamplingParams, ScoredOutput


RAW_LAYOUT = """<|box_start|>100 100 900 300<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>txt_contd_tgt
<|box_start|>101 100 900 300<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>100 100 900 600<|box_end|><|ref_start|>list<|ref_end|><|rotate_up|>
<|box_start|>100 400 900 600<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>
<|box_start|>100 400 900 600<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>"""


@pytest.fixture
def helper() -> MinerUClientHelper:
    """构造只解析字符串的轻量 helper，不加载推理模型。"""
    return MinerUClientHelper("llama-cpp-engine", {}, {}, (1036, 1036), 28, 50, False, True, False, False, False, False)


def test_real_page_layout_keeps_two_conclusions_and_list(helper: MinerUClientHelper) -> None:
    """保存论文第八页的原始 VLM 证据，两组重复正文只能保留一组。"""
    raw = (Path(__file__).parent / "fixtures" / "baige_page8_layout.txt").read_text()
    blocks = helper.parse_layout_output(raw)
    assert len(blocks) == 22
    assert [block.type for block in blocks[18:]] == ["text", "text", "list", "footer"]
    assert [block.bbox for block in blocks[18:20]] == [[0.512, 0.625, 0.887, 0.715], [0.511, 0.719, 0.887, 0.945]]


@pytest.mark.parametrize("block_type", sorted(BLOCK_TYPES))
def test_all_same_type_duplicates_keep_first_object(helper: MinerUClientHelper, block_type: str) -> None:
    """所有同类型块均使用首个对象，不能丢失正文、续接标记或额外属性。"""
    first = ContentBlock(block_type, [0.1, 0.2, 0.8, 0.4], angle=0, content="首次内容", merge_prev=block_type == "text")
    first["custom_metadata"] = {"source": "first"}
    duplicate = ContentBlock(block_type, [0.101, 0.2, 0.8, 0.4], angle=0, content="后续内容")
    blocks = [first, duplicate, deepcopy(duplicate)]
    snapshot = deepcopy(blocks)
    result = helper._deduplicate_layout_blocks(blocks)
    assert len(result) == 1 and result[0] is first
    assert blocks == snapshot
    assert helper._deduplicate_layout_blocks(result) == result


@pytest.mark.parametrize("height,kept_count", [(0.899, 2), (0.9, 2), (0.901, 1), (1.0, 1)])
def test_iou_threshold_is_strict(helper: MinerUClientHelper, height: float, kept_count: int) -> None:
    """IoU 等于 0.9 时保留，只有严格超过阈值才抑制后续框。"""
    blocks = [ContentBlock("text", [0, 0, 1, 1]), ContentBlock("text", [0, 0, 1, height])]
    assert len(helper._deduplicate_layout_blocks(blocks)) == kept_count


@pytest.mark.parametrize("second_type,second_angle", [("title", 0), ("text", 90), ("text", None)])
def test_other_types_and_angles_survive(helper: MinerUClientHelper, second_type: str, second_angle: int | None) -> None:
    """相同区域的不同类型或不同方向仍是独立候选，包括未指定角度。"""
    blocks = [ContentBlock("text", [0, 0, 1, 1], angle=0), ContentBlock(second_type, [0, 0, 1, 1], angle=second_angle)]
    assert helper._deduplicate_layout_blocks(blocks) == blocks


@pytest.mark.parametrize("parent_type,child_type", [("list", "text"), ("image_block", "image"), ("table", "text")])
def test_deduplication_preserves_container_members(helper: MinerUClientHelper, parent_type: str, child_type: str) -> None:
    """去重不能把容器包含关系当成重复，表内过滤继续由现有独立阶段负责。"""
    blocks = [ContentBlock(parent_type, [0, 0, 1, 1]), ContentBlock(child_type, [0, 0, 1, 1])]
    assert helper._deduplicate_layout_blocks(blocks) == blocks


@pytest.mark.parametrize("bbox", [[0.6, 0.6, 1, 1], [0.2, 0, 1, 1], [0, 0, 0.5, 0.5]])
def test_spatially_distinct_text_is_not_removed(helper: MinerUClientHelper, bbox: list[float]) -> None:
    """相同正文位于不同位置、部分重叠或被较大块包含时均保留。"""
    blocks = [ContentBlock("text", [0, 0, 1, 1], content="相同正文"), ContentBlock("text", bbox, content="相同正文")]
    assert helper._deduplicate_layout_blocks(blocks) == blocks


def test_suppressed_candidate_does_not_suppress_next(helper: MinerUClientHelper) -> None:
    """链式重叠只与已保留的框比较，禁止通过已删除候选扩大抑制范围。"""
    blocks = [ContentBlock("text", [x, 0, x + 0.8, 1]) for x in (0, 0.03, 0.06)]
    result = helper._deduplicate_layout_blocks(blocks)
    assert len(result) == 2 and result[0] is blocks[0] and result[1] is blocks[2]


@pytest.mark.parametrize("blocks", [[], [ContentBlock("text", [0, 0, 1, 1])]])
def test_empty_and_single_layout(helper: MinerUClientHelper, blocks: list[ContentBlock]) -> None:
    """空结果和单块结果无需特殊配置且保持原有内容。"""
    assert helper._deduplicate_layout_blocks(blocks) == blocks


def test_existing_table_internal_filter_still_applies(helper: MinerUClientHelper) -> None:
    """去重后的版面仍执行原有表内文字过滤，保留首个表格框。"""
    raw = "\n".join(
        f"<|box_start|>100 100 900 900<|box_end|><|ref_start|>{kind}<|ref_end|><|rotate_up|>"
        for kind in ("table", "table", "text")
    )
    assert [block.type for block in helper.parse_layout_output(raw)] == ["table"]


@pytest.fixture
def recording_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[MinerUClient, list[int], ScoredOutput]]:
    """模拟底层推理，真实执行布局解析、裁图和后处理，记录内容识别数量。"""
    client = MinerUClient(
        backend="http-client", server_url="http://test", model_name="test", skip_model_name_checking=True, use_tqdm=False
    )
    calls: list[int] = []
    layout_score = ScoredOutput(RAW_LAYOUT, [1], [-0.1], 1.1, -0.1, 0.0)

    def predict(
        image: ImageType, prompt: str, params: SamplingParams | None = None, priority: int | None = None
    ) -> ScoredOutput:
        """布局预测返回重复框，内容预测记录实际识别的一张裁图。"""
        if prompt == client.prompts["[layout]"]:
            return layout_score
        calls.append(1)
        return ScoredOutput("识别正文", [2], [-0.2], 1.2, -0.2, 0.0)

    def batch_predict(
        images: Sequence[ImageType],
        prompts: Sequence[str] | str,
        params: object = None,
        priority: object = None,
    ) -> list[ScoredOutput]:
        """批量识别按实际输入逐项记录，避免绕过高层索引映射。"""
        prompt_list = [prompts] * len(images) if isinstance(prompts, str) else prompts
        return [predict(image, prompt) for image, prompt in zip(images, prompt_list, strict=True)]

    async def aio_predict(*args: object, **kwargs: object) -> ScoredOutput:
        """异步单项使用相同确定性预测数据。"""
        return predict(*args, **kwargs)

    async def aio_batch_predict(*args: object, **kwargs: object) -> list[ScoredOutput]:
        """异步批量保留普通预测参数并接受调度层进度选项。"""
        return batch_predict(*args)

    monkeypatch.setattr(client.client, "predict_scored", predict)
    monkeypatch.setattr(client.client, "batch_predict_scored", batch_predict)
    monkeypatch.setattr(client.client, "aio_predict_scored", aio_predict)
    monkeypatch.setattr(client.client, "aio_batch_predict_scored", aio_batch_predict)
    yield client, calls, layout_score
    asyncio.run(client.aclose())


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_layout_interfaces_share_deduplication(
    recording_client: tuple[MinerUClient, list[int], ScoredOutput],
    use_async: bool,
    batch: bool,
) -> None:
    """四个布局接口逐页去重，不能跨页删除同一位置的块或改写整页评分。"""
    client, calls, score = recording_client
    with Image.new("RGB", (100, 100), "white") as image:
        if use_async:
            method = client.aio_batch_layout_detect if batch else client.aio_layout_detect
            output = asyncio.run(method([image, image] if batch else image, scored=True))
        else:
            method = client.batch_layout_detect if batch else client.layout_detect
            output = method([image, image] if batch else image, scored=True)
        pages = output if batch else [output]
        assert len(pages) == (2 if batch else 1)
        assert calls == []
        for page in pages:
            assert isinstance(page, ExtractResult) and page.layout_scored is score
            assert [block.type for block in page] == ["text", "list", "text"]
            assert page[0].merge_prev is True


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", ["single", "concurrent", "stepping"])
@pytest.mark.parametrize("txt_mode", [False, True])
def test_two_step_deduplicates_before_recognition(
    recording_client: tuple[MinerUClient, list[int], ScoredOutput],
    use_async: bool,
    mode: str,
    txt_mode: bool,
) -> None:
    """TXT 跳过正文识别，OCR 每个正文区域仅识别一次，覆盖两种批量编排。"""
    client, calls, score = recording_client
    name = "two_step_extract" if mode == "single" else f"{mode}_two_step_extract"
    if use_async:
        name = "aio_" + name
    with Image.new("RGB", (100, 100), "white") as image:
        output = getattr(client, name)(
            image if mode == "single" else [image, image], not_extract_list=["text"] if txt_mode else None, scored=True
        )
        if use_async:
            output = asyncio.run(output)
        pages = [output] if mode == "single" else output
        assert len(calls) == (0 if txt_mode else 2 * len(pages))
        for page in pages:
            assert page.layout_scored is score
            assert [block.type for block in page] == ["text", "list", "text"]
            assert [block.content for block in page if block.type == "text"] == (
                [None, None] if txt_mode else ["识别正文", "识别正文"]
            )


@pytest.mark.parametrize("use_async", [False, True])
def test_external_layout_is_unchanged(
    recording_client: tuple[MinerUClient, list[int], ScoredOutput],
    use_async: bool,
) -> None:
    """外部布局仍保留两个候选，并保持现有原地回填正文的行为。"""
    client, calls, _ = recording_client
    blocks = [ContentBlock("text", [0, 0, 1, 1]), ContentBlock("text", [0, 0, 1, 1])]
    with Image.new("RGB", (100, 100), "white") as image:
        method = client.aio_extract_with_layout if use_async else client.extract_with_layout
        output = method(image, blocks, scored=True)
        if use_async:
            output = asyncio.run(output)
    assert len(output) == 2 and len(calls) == 2
    assert all(output[index] is block for index, block in enumerate(blocks))
    assert all(block.bbox == [0, 0, 1, 1] and block.content == "识别正文" for block in blocks)
