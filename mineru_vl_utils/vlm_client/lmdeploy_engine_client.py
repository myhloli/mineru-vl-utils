import asyncio
from io import BytesIO
from itertools import groupby
from typing import Any, Sequence

from PIL import Image

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
from .utils import gather_tasks, get_rgb_image, load_resource


class LmdeployEngineVlmClient(VlmClient):
    def __init__(
        self,
        lmdeploy_engine,  # LMDeploy 0.17 的公开 Pipeline 实例
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,  # batch size for sync predict
        max_concurrency: int = 100,  # max concurrency for async predict
        use_tqdm: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        try:
            from lmdeploy import GenerationConfig
            from lmdeploy.pipeline import Pipeline
        except ImportError:
            raise ImportError("Please install lmdeploy to use LmdeployEngineVlmClient.")

        if not lmdeploy_engine:
            raise ValueError("lmdeploy_engine is None.")
        if not isinstance(lmdeploy_engine, Pipeline):
            raise ValueError("lmdeploy_engine must be an instance of lmdeploy.pipeline.Pipeline.")

        self.lmdeploy_engine = lmdeploy_engine
        self.model_max_length = lmdeploy_engine.backend_config.session_len
        self.LmdeployGenerationConfig = GenerationConfig
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.use_tqdm = use_tqdm
        self.debug = debug

    def build_lmdeploy_generation_config(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        do_sample = ((sp.temperature or 0.0) > 0.0) and ((sp.top_k or 1) > 1)

        lmdeploy_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "repetition_penalty": sp.repetition_penalty,
            # WARNING - engine.py:606: num tokens is larger than max session len xxx. Update max_new_tokens=xxx.
            "max_new_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }

        return self.LmdeployGenerationConfig(
            **{k: v for k, v in lmdeploy_sp_dict.items() if v is not None},
            do_sample=do_sample,
            skip_special_tokens=False,
        )

    def predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            [sampling_params],
            [priority],
        )[0]

    def batch_predict(
        self,
        images: Sequence[ImageType],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(priority, Sequence):
            assert len(priority) == len(images), "Length of priority and images must match."

        image_objs: list[Image.Image | None] = []
        for image in images:
            if image is None:
                image_objs.append(None)
                continue
            if not isinstance(image, SingleImageType):
                raise UnsupportedError("LmdeployEngineVlmClient haven't support non-single image yet.")
            if isinstance(image, str):
                image = load_resource(image)
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
            image = get_rgb_image(image)
            image_objs.append(image)

        if isinstance(prompts, str):
            chat_prompts: list[str] = [prompts] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = list(prompts)

        if not isinstance(sampling_params, Sequence):
            gen_configs = [self.build_lmdeploy_generation_config(sampling_params)] * len(images)
        else:  # isinstance(sampling_params, Sequence)
            gen_configs = [self.build_lmdeploy_generation_config(sp) for sp in sampling_params]

        outputs = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        priorities = priority if isinstance(priority, Sequence) else [priority] * len(images)
        # Pipeline 的 priority 作用于整次调用；相邻同优先级请求仍按原顺序批处理。
        for current_priority, group in groupby(
            zip(image_objs, chat_prompts, gen_configs, priorities), key=lambda item: item[3]
        ):
            items = list(group)
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                outputs.extend(
                    self._predict_one_batch(
                        [item[0] for item in batch],
                        [item[1] for item in batch],
                        [item[2] for item in batch],
                        priority=current_priority,
                    )
                )

        return outputs

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image | None],
        chat_prompts: list[str],
        gen_configs: list[Any],
        priority: int | None = None,
    ) -> list[str]:
        """通过公开 Pipeline 接口推理，并将后端错误传播给同步与异步调用方。"""
        lmdeploy_prompts = [(prompt, image) if image is not None else prompt for prompt, image in zip(chat_prompts, image_objs)]
        generate_kwargs = {} if priority is None else {"priority": priority}
        outputs = self.lmdeploy_engine.infer(
            lmdeploy_prompts,  # type: ignore
            gen_config=gen_configs,
            **generate_kwargs,
        )
        if len(outputs) != len(lmdeploy_prompts):
            raise ServerError("LMDeploy returned an unexpected number of responses.")
        if any(getattr(output, "finish_reason", None) == "error" for output in outputs):
            raise ServerError("LMDeploy inference failed.")
        return [output.text for output in outputs]

    async def aio_predict(
        self,
        image: ImageType,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        """在线程中复用 Pipeline；取消时等待在途调用结束，再释放并发名额和共享引擎租约。"""
        work = asyncio.create_task(asyncio.to_thread(self.predict, image, prompt, sampling_params, priority))
        try:
            return await asyncio.shield(work)
        except asyncio.CancelledError:
            # 同步 infer 无法通过取消 Future 中断；提前返回会遗留仍在使用引擎的线程。
            while not work.done():
                try:
                    await asyncio.shield(work)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not work.cancelled():
                work.exception()
            raise

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
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            image: ImageType,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                )

        return await gather_tasks(
            tasks=[
                predict_with_semaphore(*args)
                for args in zip(
                    images,
                    prompts,
                    sampling_params,
                    priority,
                )
            ],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )
