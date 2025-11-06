import asyncio
from io import BytesIO
from typing import Sequence

from PIL import Image

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    RequestError,
    SamplingParams,
    ServerError,
    UnsupportedError,
    VlmClient,
)
from .utils import aio_load_resource, gather_tasks, get_rgb_image


class LmdeployAsyncEngineVlmClient(VlmClient):
    def __init__(
        self,
        lmdeploy_engine,  # lmdeploy.serve.vl_async_engine.VLAsyncEngine instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 100,
        debug: bool = False,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )
        self.lmdeploy_engine = lmdeploy_engine
        self.tokenizer = lmdeploy_engine.tokenizer
        # TODO: handle sampling params conversion
        try:
            from lmdeploy import GenerationConfig
            self.gen_config = GenerationConfig(skip_special_tokens=False, max_new_tokens=16384, top_k=1, top_p=0.01, do_sample=True)
        except ImportError:
            raise ImportError("Please install lmdeploy to use LmdeployAsyncEngineVlmClient.")
        self.max_concurrency = max_concurrency
        self.session_id = 0
        self.debug = debug

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise UnsupportedError(
            "Synchronous predict() is not supported in lmdeploy-async-engine VlmClient(backend). "
            "Please use aio_predict() instead. If you intend to use synchronous client, "
            "please use lmdeploy-engine VlmClient(backend)."
        )

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        raise UnsupportedError(
            "Synchronous batch_predict() is not supported in lmdeploy-async-engine VlmClient(backend). "
            "Please use aio_batch_predict() instead. If you intend to use synchronous client, "
            "please use lmdeploy-engine VlmClient(backend)."
        )

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        if isinstance(image, str):
            image = await aio_load_resource(image)
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image))
        image = get_rgb_image(image)

        lmdeploy_prompts = [(prompt, image),]
        generate_kwargs = {}
        if priority is not None:
            generate_kwargs["priority"] = priority

        lmdeploy_prompts = self.lmdeploy_engine._convert_prompts(lmdeploy_prompts)[0]
        final_output = ''
        self.session_id += 1
        async for output in self.lmdeploy_engine.generate(
            messages=lmdeploy_prompts,
            gen_config = self.gen_config,
            session_id=self.session_id,
            **generate_kwargs,
        ):
            final_output += output.response

        if not final_output:  # this should not happen
            raise ServerError("No output from the server.")

        return final_output

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
            image: Image.Image | bytes | str,
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
