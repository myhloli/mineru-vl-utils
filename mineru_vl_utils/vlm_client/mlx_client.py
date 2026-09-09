# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import threading
from dataclasses import astuple
from io import BytesIO
from typing import Any, Iterator, Sequence

from PIL import Image

from tqdm import tqdm

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    ImageType,
    SamplingParams,
    ServerError,
    SingleImageType,
    UnsupportedError,
    VlmClient,
)
from .utils import get_rgb_image, load_resource, run_in_thread_until_complete

# 覆盖八张 1036×1036 layout 输入（8,586,368 像素）；这是分批预算而非硬内存上限。
# 单张超预算仍允许单独推理，不改变调用方分辨率。
MLX_BATCH_PIXEL_BUDGET = 9_000_000


class MlxVlmClient(VlmClient):
    def __init__(
        self,
        model,  # MLX model object
        processor,  # MLX processor object
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,
        use_tqdm: bool = True,
    ):
        """保存 MLX 模型，并以实例锁串行化同一模型的生成与缓存访问。"""
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        self.model = model
        # mlx-vlm 0.7 的 server 用单 GPU 线程批处理；直接 generate 跨线程并发仍会报 stream 错误。
        self._generation_lock = threading.Lock()
        self.processor = processor
        self.batch_size = batch_size if batch_size > 0 else 8
        self.use_tqdm = use_tqdm
        self.model_max_length = model.config.text_config.max_position_embeddings
        try:
            from mlx_vlm import generate
            from mlx_vlm.generate import BatchGenerator

            self.generate = generate
            self.batch_generator_type = BatchGenerator
        except ImportError:
            raise ImportError("Please install mlx-vlm to use the mlx-engine backend.")

    def build_messages(self, prompt: str, has_image: bool = True) -> list[dict]:
        prompt = prompt or self.prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if not has_image:
            user_messages = [{"type": "text", "text": prompt}]
        elif "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_messages = [
                *([{"type": "text", "text": prompt_1}] if prompt_1.strip() else []),
                {"type": "image"},
                *([{"type": "text", "text": prompt_2}] if prompt_2.strip() else []),
            ]
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages

    def build_generate_kwargs(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)
        generate_kwargs = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repetition_penalty": sp.repetition_penalty,
            # max_tokens should smaller than model max length
            "max_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }
        return generate_kwargs

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """执行单图预测，共享模型的生成操作在同一时间只允许一个线程进入。"""
        image_obj = self._image_object(image)
        with self._generation_lock:
            return self._predict_one(image_obj, prompt, sampling_params)

    @staticmethod
    def _image_object(image: ImageType) -> Image.Image | None:
        """统一图片输入并物化解码结果，不关闭调用方传入的 PIL 图片。"""
        if image is None:
            return None
        if not isinstance(image, SingleImageType):
            raise UnsupportedError("MlxVlmClient does not support multiple images per sample.")
        if isinstance(image, Image.Image):
            return get_rgb_image(image)
        if isinstance(image, str):
            image = load_resource(image)
        with Image.open(BytesIO(image)) as opened:
            return get_rgb_image(opened).copy()

    def _predict_one(self, image: Image.Image | None, prompt: str, sampling_params: SamplingParams | None) -> str:
        """在调用方持锁时执行原有单样本路径，保留自定义图文顺序。"""
        chat_prompt = self.processor.apply_chat_template(
            self.build_messages(prompt, has_image=image is not None),
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.generate(
            model=self.model,
            processor=self.processor,
            prompt=chat_prompt,
            image=image,
            **self.build_generate_kwargs(sampling_params),
        ).text

    def _batch_kwargs(self, params: SamplingParams, count: int) -> dict[str, Any]:
        """把生成参数映射到 batch API，每个样本独立构造惩罚处理器。"""
        from mlx_vlm.sample_utils import make_logits_processors, make_sampler

        return {
            "max_tokens": params.max_new_tokens if params.max_new_tokens is not None else self.model_max_length,
            "sampler": make_sampler(
                temp=params.temperature if params.temperature is not None else 0.0,
                top_p=params.top_p if params.top_p is not None else 1.0,
                top_k=params.top_k if params.top_k is not None else 0,
            ),
            "logits_processors": [
                make_logits_processors(
                    repetition_penalty=params.repetition_penalty,
                    presence_penalty=params.presence_penalty,
                    frequency_penalty=params.frequency_penalty,
                )
                for _ in range(count)
            ],
        }

    def _prepare_batch_prompt(self, image: Image.Image | None, prompt: str) -> tuple[list[int], dict[str, Any]]:
        """保留既有聊天模板，逐样本构建视觉特征，避免上游 batch_generate 重排图文。"""
        import mlx.core as mx
        from mlx_vlm.utils import prepare_inputs, should_add_special_tokens

        chat_prompt = self.processor.apply_chat_template(
            self.build_messages(prompt, has_image=image is not None),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = prepare_inputs(
            self.processor,
            images=[image] if image is not None else None,
            prompts=[chat_prompt],
            image_token_index=getattr(self.model.config, "image_token_index", None),
            add_special_tokens=should_add_special_tokens(self.model.config.model_type, self.processor),
        )
        input_ids = inputs["input_ids"]
        data = {key: value for key, value in inputs.items() if key not in {"input_ids", "pixel_values", "attention_mask"}}
        features = self.model.get_input_embeddings(
            input_ids,
            inputs.get("pixel_values"),
            mask=inputs.get("attention_mask"),
            **data,
        )
        features = {key: value for key, value in features.to_dict().items() if value is not None}
        # 先物化视觉特征，避免整个批次同时持有每张图片的视觉计算图。
        mx.eval(features)
        return input_ids.squeeze(0).tolist(), {**data, **features}

    def _predict_batch(
        self,
        images: list[Image.Image | None],
        prompts: list[str],
        params: SamplingParams,
    ) -> list[str]:
        """串行持有 GPU，在同一调度器内批量预填充与解码，按请求 ID 还原结果。"""
        with self._generation_lock:
            if len(images) == 1:
                return [self._predict_one(images[0], prompts[0], params)]
            prepared = [self._prepare_batch_prompt(image, prompt) for image, prompt in zip(images, prompts)]
            kwargs = self._batch_kwargs(params, len(images))
            processors = kwargs.pop("logits_processors")
            generator = self.batch_generator_type(
                self.model.language_model,
                self.processor,
                prefill_batch_size=len(images),
                completion_batch_size=len(images),
                compute_logprobs=False,
                **kwargs,
            )
            try:
                uids = generator.insert(
                    [item[0] for item in prepared],
                    prompt_kwargs=[item[1] for item in prepared],
                    logits_processors=processors,
                )
                if len(uids) != len(images) or len(set(uids)) != len(uids):
                    raise ServerError("MLX returned unexpected batch request IDs.")
                tokens: dict[int, list[int]] = {uid: [] for uid in uids}
                finished: set[int] = set()
                while generator.has_work:
                    _, responses = generator.next()
                    for response in responses:
                        if response.uid not in tokens:
                            raise ServerError("MLX returned an unknown batch request ID.")
                        if response.finish_reason != "stop":
                            tokens[response.uid].append(response.token)
                        if response.finish_reason is not None:
                            finished.add(response.uid)
                if finished != set(uids):
                    raise ServerError("MLX returned an incomplete batch response.")
                texts = []
                for uid in uids:
                    detokenizer = self.processor.detokenizer
                    detokenizer.reset()
                    for token in tokens[uid]:
                        detokenizer.add_token(token)
                    detokenizer.finalize()
                    texts.append(detokenizer.text)
                return texts
            finally:
                generator.close()

    def _iter_batches(self, indices: list[int], images: list[Image.Image | None]) -> Iterator[list[int]]:
        """相近像素数的图片优先同批，限制样本数和总像素，不改变图片尺寸。"""
        sized = [(images[i].width * images[i].height if images[i] is not None else 0, i) for i in indices]
        batch: list[int] = []
        pixels = 0
        for size, idx in sorted(sized):
            if batch and (len(batch) >= self.batch_size or pixels + size > MLX_BATCH_PIXEL_BUDGET):
                yield batch
                batch, pixels = [], 0
            batch.append(idx)
            pixels += size
        if batch:
            yield batch

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        """按模态与有效采样参数分组，受 batch/像素预算约束并还原输入顺序。"""
        count = len(images)
        for name, values in (("prompts", prompts), ("sampling_params", sampling_params), ("priority", priority)):
            if isinstance(values, Sequence) and not isinstance(values, str) and len(values) != count:
                raise ValueError(f"Length of {name} and images must match.")
        if not count:
            return []
        prompts = [prompts] * count if isinstance(prompts, str) else list(prompts)
        params = list(sampling_params) if isinstance(sampling_params, Sequence) else [sampling_params] * count
        image_objs = [self._image_object(image) for image in images]
        resolved = [self.build_sampling_params(param) for param in params]
        groups: dict[tuple[Any, ...], list[int]] = {}
        for idx, (image, param) in enumerate(zip(image_objs, resolved)):
            key = (image is not None, *astuple(param))
            groups.setdefault(key, []).append(idx)
        if self.batch_size == 1:
            # 回退档保持原有调用顺序，避免分组改变随机采样的消费顺序。
            groups = {(idx,): [idx] for idx in range(count)}
        outputs = [""] * count
        with tqdm(total=count, desc="Predict", disable=not self.use_tqdm) as pbar:
            for indices in groups.values():
                for batch in self._iter_batches(indices, image_objs):
                    texts = self._predict_batch(
                        [image_objs[i] for i in batch],
                        [prompts[i] for i in batch],
                        resolved[batch[0]],
                    )
                    for idx, text in zip(batch, texts):
                        outputs[idx] = text
                    pbar.update(len(batch))
        return outputs

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """在线程中生成，取消后等在途工作结束再归还模型租约。"""
        return await run_in_thread_until_complete(
            self.predict,
            image,
            prompt,
            sampling_params,
            priority,
        )

    async def aio_batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        """批量调用同样等待线程完成；调用方提供的并发名额覆盖完整生命周期。"""
        if semaphore is not None:
            async with semaphore:
                return await run_in_thread_until_complete(self.batch_predict, images, prompts, sampling_params, priority)
        return await run_in_thread_until_complete(self.batch_predict, images, prompts, sampling_params, priority)


__all__ = ["MlxVlmClient"]
