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
    # 专用于 MLX-VLM 的聊天模板工具，支持多图数量控制
    try:
        from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template
    except Exception:  # pragma: no cover - 兼容旧版本 mlx-vlm
        mlx_apply_chat_template = None
except ImportError:
    # 允许在没有安装 mlx-vlm 的环境中导入此文件
    generate = None
    mlx_apply_chat_template = None

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

    def build_multi_image_prompt(self, prompt: str, num_images: int) -> str:
        """
        构建多图对话的 chat prompt。
        优先使用 MLX-VLM 官方提供的 `apply_chat_template`，可显式指定图片数量，
        若不可用，则退化为通过重复 <image> 标记进行提示的兼容实现。
        """
        prompt = prompt or self.prompt

        # 优先使用 MLX 自带的模板函数（与官方示例一致）
        if mlx_apply_chat_template is not None:
            try:
                config = getattr(self.model, "config", None)
                return mlx_apply_chat_template(self.processor, config, prompt, num_images=num_images)
            except Exception:
                # 继续走到兜底逻辑
                pass

        # 兜底：根据 text_before_image 决定 <image> 与文本的顺序
        image_tokens = "\n".join(["<image>"] * max(1, int(num_images)))
        if self.text_before_image:
            return f"{prompt}\n{image_tokens}"
        else:
            return f"{image_tokens}\n{prompt}"
    
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

    def predict_multi_images(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """
        多图单轮对话推理：在一次 `generate` 调用中同时传入多张图片，
        适用于需要跨图对比/联合推理的场景（与官方 Multi-Image Chat Support 一致）。

        注意：该接口与 `predict` 的差异在于 `images` 为序列，且内部优先使用
        `mlx_vlm.prompt_utils.apply_chat_template(..., num_images=len(images))` 来构建提示。
        """
        final_prompt = prompt or self.prompt
        final_sampling_params = self.build_sampling_params(sampling_params)

        chat_prompt = self.build_multi_image_prompt(final_prompt, num_images=len(images))

        # 参考官方示例使用位置参数传递 images，以兼容不同版本签名
        response = generate(
            self.model,
            self.processor,
            chat_prompt,
            images,
            temperature=final_sampling_params.temperature or 0.3,
            max_tokens=final_sampling_params.max_new_tokens or 1024,
        )
        return getattr(response, "text", response)

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

    def batch_predict_multi_images(
        self,
        images_list: Sequence[Sequence[Image.Image | bytes | str]],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        """
        多图批量推理：`images_list` 的每个元素代表一次对话中使用的多张图片序列。
        会循环调用 `predict_multi_images` 收集结果，支持进度条显示。
        """
        results: list[str] = []

        # prompts / sampling_params 广播
        if isinstance(prompts, str):
            prompts = [prompts] * len(images_list)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images_list)

        iterable = range(len(images_list))
        if self.use_tqdm:
            iterable = tqdm(iterable, desc="使用 MLX 多图批量处理中")

        for i in iterable:
            result = self.predict_multi_images(
                images=images_list[i],
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

    async def aio_predict_multi_images(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """
        异步多图单轮推理：将同步 `predict_multi_images` 移至线程池执行，避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.predict_multi_images,
            images,
            prompt,
            sampling_params,
            priority,
        )

    async def aio_batch_predict_multi_images(
        self,
        images_list: Sequence[Sequence[Image.Image | bytes | str]],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        """
        异步多图批量推理：在线程池中执行 `batch_predict_multi_images`。
        """
        return await asyncio.to_thread(
            self.batch_predict_multi_images,
            images_list,
            prompts,
            sampling_params,
            priority,
        )
