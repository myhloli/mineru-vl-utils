import asyncio
from typing import Sequence
from PIL import Image
from tqdm import tqdm

from .base_client import (VlmClient, 
                          SamplingParams, 
                          DEFAULT_USER_PROMPT, 
                          DEFAULT_SYSTEM_PROMPT)

# 确保导入了 MLX 的相关函数
try:
    from mlx_vlm import generate
except ImportError:
    # 允许在没有安装 mlx-vlm 的环境中导入此文件
    generate = None

class MlxVlmClient(VlmClient):
    """
    一个使用 mlx-vlm 库在 Apple Silicon 上本地运行模型的 VLM 客户端。
    """
    def __init__(
        self,
        model,  # MLX model object
        processor,  # MLX processor object
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 1,
        use_tqdm: bool = True,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        if generate is None:
            raise ImportError("请运行 'pip install mlx-vlm' 来使用 MLX 后端。")
        
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,  # MLX 本地运行不支持优先级
    ) -> str:
        # MLX 的 generate 函数本身不支持批量处理，所以我们直接调用
        final_prompt = prompt or self.prompt
        
        # 构建 message
        messages = [
            {"role": "user", "content": f"<image>\n{final_prompt}"},
        ]

        # 构建采样参数
        final_sampling_params = self.build_sampling_params(sampling_params)
        
        response = generate(
            self.model,
            self.processor,
            image,
            messages,
            temperature=final_sampling_params.temperature or 0.7,
            max_tokens=final_sampling_params.max_new_tokens or 1024,
        )
        return response

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        results = []
        
        # 准备 prompts 和 sampling_params 列表
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
            
        iterable = range(len(images))
        if self.use_tqdm:
            iterable = tqdm(iterable, desc="使用 MLX 批量处理中")

        # 由于 mlx-vlm 的 generate 函数不是批处理的，我们只能循环调用
        for i in iterable:
            result = self.predict(
                image=images[i],
                prompt=prompts[i],
                sampling_params=sampling_params[i],
            )
            results.append(result)
            
        return results

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """
        异步推理接口：将同步的 predict 封装到线程池，避免阻塞事件循环。
        说明：不改变 MLX 的同步执行特性，仅提供与 MinerUClient 异步路径兼容的接口。
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.predict,
            image,
            prompt,
            sampling_params,
            priority,
        )

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        # 注意: MLX 本身是同步阻塞操作。此处使用 asyncio.to_thread 将 CPU/GPU 密集任务移至线程池，
        # 避免阻塞事件循环，但不改变其同步执行特性与行为。
        return await asyncio.to_thread(
            self.batch_predict,
            images,
            prompts,
            sampling_params,
            priority,
        )
