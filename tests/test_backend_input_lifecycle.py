"""验证新版后端的视觉特征保留和线程取消生命周期。"""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from mineru_vl_utils.vlm_client.mlx_client import MlxVlmClient
from mineru_vl_utils.vlm_client.vllm_engine_client import VllmEngineVlmClient


class RenderingEngine:
    """模拟同步引擎内部渲染；重复输入 EngineInput 会丢失视觉内容。"""

    def __init__(self) -> None:
        """记录每次引擎内部处理得到的实际图像特征。"""
        self.renderer = SimpleNamespace(render_cmpl=self.render)
        self.render_count = 0
        self.features = []

    def render(self, prompts: list[dict]) -> list[dict]:
        """将 raw 图像物化为特征，刻意不把已物化输入当作 raw 图像。"""
        self.render_count += 1
        return [{"type": "multimodal", "mm_kwargs": p.get("multi_modal_data", {}).get("image", [])} for p in prompts]

    def generate(self, prompts: list[dict], **kwargs: object) -> list[object]:
        """同步 generate 自行渲染，输出后续断言要核对的图像像素。"""
        rendered = self.render(prompts)
        self.features.extend([im.getpixel((0, 0)) for p in rendered for im in p["mm_kwargs"]])
        return [SimpleNamespace(text="recognized") for _ in prompts]


@pytest.mark.parametrize("mode", ["predict", "predict_scored", "score"])
def test_vllm_sync_keeps_visual_features(mode: str) -> None:
    """普通预测和两种评分都必须让引擎恰好渲染一次，并保留两张不同图像。"""
    client = object.__new__(VllmEngineVlmClient)
    client.vllm_llm = RenderingEngine()
    client.use_tqdm = False
    client.batch_size = 0
    client.tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: "prompt")
    client.build_messages = lambda *a: []
    client.build_vllm_sampling_params = lambda *a: SimpleNamespace()
    client.get_output_content = lambda output: output.text
    client.get_output_scored = lambda output: output
    client._build_score_prompt_pair = lambda *a: ("prompt", 1)
    client._extract_prompt_logprobs = lambda output, count: output
    images = [Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")]
    if mode == "predict":
        outputs = client.batch_predict(images)
    elif mode == "predict_scored":
        outputs = client.batch_predict_scored(images)
    else:
        outputs = client.batch_score(images, ["a", "b"])
    assert len(outputs) == 2
    assert client.vllm_llm.render_count == 1
    assert client.vllm_llm.features == [(255, 0, 0), (0, 0, 255)]


@pytest.mark.parametrize("batch", [False, True])
def test_mlx_cancellation_holds_lease_until_worker_exits(batch: bool) -> None:
    """取消单图或批量请求时，不能在底层线程仍使用模型时提前结束协程。"""
    entered, release = threading.Event(), threading.Event()
    client = object.__new__(MlxVlmClient)

    def predict(*args: object) -> str:
        """用事件精确控制在途推理，不依赖模型下载或 GPU。"""
        entered.set()
        assert release.wait(5)
        return "done"

    client.predict = predict
    client.batch_predict = predict

    async def run() -> None:
        """重复取消仍须等待线程，异常完成后释放并发名额。"""
        semaphore = asyncio.Semaphore(1)
        call = client.aio_batch_predict([None], semaphore=semaphore) if batch else client.aio_predict(None)
        task = asyncio.create_task(call)
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            if batch:
                assert semaphore.locked()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not semaphore.locked()

    asyncio.run(run())
