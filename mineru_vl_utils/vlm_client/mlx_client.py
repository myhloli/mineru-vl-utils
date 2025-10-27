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

    def build_messages(self, prompt: str) -> list[dict]:
        """根据传入的 prompt 构建消息结构。
        与其他后端保持一致：支持 system 提示与 <image> 占位处理，
        同时支持 text_before_image 的顺序切换。
        """
        prompt = prompt or self.prompt
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_messages: list[dict] = []
            if prompt_1.strip():
                user_messages.append({"type": "text", "text": prompt_1})
            user_messages.append({"type": "image"})
            if prompt_2.strip():
                user_messages.append({"type": "text", "text": prompt_2})
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ]
        else:
            user_messages = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages
    
    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
        ) -> str:
        final_prompt = prompt or self.prompt
        final_sampling_params = self.build_sampling_params(sampling_params)
        # 构建 chat prompt（与 transformers/http-client 保持一致）
        if hasattr(self.processor, "apply_chat_template"):
            messages = self.build_messages(final_prompt)
            chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        else:
            # 兜底：纯文本 + <image> 占位
            chat_prompt = f"<image>\n{final_prompt}"

        response = generate(
            model=self.model,
            processor=self.processor,
            prompt=chat_prompt,
            image=image,
            temperature=final_sampling_params.temperature or 0.3,
            max_tokens=final_sampling_params.max_new_tokens or 1024,
        )
        return response.text

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
