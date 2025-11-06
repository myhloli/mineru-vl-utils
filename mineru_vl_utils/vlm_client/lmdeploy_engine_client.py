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
from .utils import get_rgb_image, load_resource


class LmdeployEngineVlmClient(VlmClient):
    def __init__(
        self,
        lmdeploy_engine,  # lmdeploy.serve.vl_async_engine.VLAsyncEngine instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,
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

        self.lmdeploy_engine = lmdeploy_engine
        self.tokenizer = lmdeploy_engine.tokenizer
        # TODO: handle sampling params conversion
        try:
            from lmdeploy import GenerationConfig
            self.gen_config = GenerationConfig(skip_special_tokens=False, max_new_tokens=16384, top_k=1, top_p=0.01, do_sample=True)
        except ImportError:
            raise ImportError("Please install lmdeploy to use LmdeployEngineVlmClient.")
        self.batch_size = batch_size
        self.use_tqdm = use_tqdm
        self.debug = debug

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            [sampling_params],
        )[0]

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
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

        image_objs: list[Image.Image] = []
        for image in images:
            if isinstance(image, str):
                image = load_resource(image)
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
            image = get_rgb_image(image)
            image_objs.append(image)

        if isinstance(prompts, str):
            chat_prompts: list[str] = [
                prompts
            ] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = [
                prompt for prompt in prompts]

        outputs = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        for i in range(0, len(images), batch_size):
            batch_image_objs = image_objs[i : i + batch_size]
            batch_chat_prompts = chat_prompts[i : i + batch_size]
            batch_outputs = self._predict_one_batch(
                batch_image_objs,
                batch_chat_prompts,
            )
            outputs.extend(batch_outputs)

        return outputs

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image],
        chat_prompts: list[str],
    ):
        lmdeploy_prompts = list(zip(chat_prompts, image_objs))
        outputs = self.lmdeploy_engine.batch_infer(lmdeploy_prompts, gen_config=self.gen_config)
        return [output.text for output in outputs]

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        raise UnsupportedError(
            "Asynchronous aio_predict() is not supported in lmdeploy-engine VlmClient(backend). "
            "Please use predict() instead. If you intend to use asynchronous client, "
            "please use lmdeploy-async-engine VlmClient(backend)."
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
        raise UnsupportedError(
            "Asynchronous aio_batch_predict() is not supported in lmdeploy-engine VlmClient(backend). "
            "Please use batch_predict() instead. If you intend to use asynchronous client, "
            "please use lmdeploy-async-engine VlmClient(backend)."
        )
