import asyncio
import uuid
from io import BytesIO
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

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
from .utils import aio_load_resource, get_rgb_image


class VllmAsyncEngineVlmClient(VlmClient):
    def __init__(
        self,
        vllm_async_llm,  # vllm.v1.engine.async_llm.AsyncLLM instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        max_concurrency: int = 1024,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        try:
            from vllm import SamplingParams
            from vllm.sampling_params import RequestOutputKind
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:
            raise ImportError("Please install vllm to use VllmEngineVlmClient.")

        if not vllm_async_llm:
            raise ValueError("vllm_async_llm is None.")
        if not isinstance(vllm_async_llm, AsyncLLM):
            raise ValueError(f"vllm_async_llm must be an instance of {AsyncLLM}")

        self.vllm_async_llm = vllm_async_llm
        self.tokenizer = vllm_async_llm.tokenizer.get_lora_tokenizer()
        self.model_max_length = vllm_async_llm.model_config.max_model_len
        self.VllmSamplingParams = SamplingParams
        self.VllmRequestOutputKind = RequestOutputKind
        self.max_concurrency = max_concurrency

    def build_messages(self, prompt: str) -> list[dict]:
        prompt = prompt or self.prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
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

    def build_vllm_sampling_params(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        vllm_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "presence_penalty": sp.presence_penalty,
            "frequency_penalty": sp.frequency_penalty,
            "repetition_penalty": sp.repetition_penalty,
            # max_tokens should smaller than model max length
            "max_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }

        return self.VllmSamplingParams(
            **{k: v for k, v in vllm_sp_dict.items() if v is not None},
            skip_special_tokens=False,
            output_kind=self.VllmRequestOutputKind.FINAL_ONLY,
        )

    def get_output_content(self, output: "RequestOutput") -> str:
        if not output.finished:
            raise ServerError("The output generation was not finished.")

        choices = output.outputs
        if not (isinstance(choices, list) and choices):
            raise ServerError("No choices found in the output.")

        finish_reason = choices[0].finish_reason
        if finish_reason == "length":
            if not self.allow_truncated_content:
                raise RequestError("The output was truncated due to length limit.")
            else:
                print("Warning: The output was truncated due to length limit.")
        elif finish_reason != "stop":
            raise RequestError(f"Unexpected finish reason: {finish_reason}")

        return choices[0].text

    def predict(
        self,
        image: str | bytes | Image.Image,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
    ) -> str:
        raise UnsupportedError(
            "Synchronous predict() is not supported in VllmAsyncEngineVlmClient. Please use aio_predict() instead."
        )

    def batch_predict(
        self,
        images: list[str] | list[bytes] | list[Image.Image],
        prompts: list[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
    ) -> list[str]:
        raise UnsupportedError(
            "Synchronous batch_predict() is not supported in VllmAsyncEngineVlmClient. "
            "Please use aio_batch_predict() instead."
        )

    async def aio_predict(
        self,
        image: str | bytes | Image.Image,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
    ) -> str:
        if isinstance(image, str):
            image = await aio_load_resource(image)
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image))
        image = get_rgb_image(image)

        chat_prompt: str = self.tokenizer.apply_chat_template(
            self.build_messages(prompt),  # type: ignore
            tokenize=False,
            add_generation_prompt=True,
        )

        vllm_sp = self.build_vllm_sampling_params(sampling_params)

        last_output = None
        async for output in self.vllm_async_llm.generate(
            prompt={"prompt": chat_prompt, "multi_modal_data": {"image": image}},
            sampling_params=vllm_sp,
            request_id=str(uuid.uuid4()),
        ):
            last_output = output

        if last_output is None:  # this should not happen
            raise ServerError("No output from the server.")

        return self.get_output_content(last_output)

    async def aio_batch_predict(
        self,
        images: list[str] | list[bytes] | list[Image.Image],
        prompts: list[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        semaphore: asyncio.Semaphore | None = None,
    ) -> list[str]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(image: str | bytes | Image.Image, prompt: str, sampling_params: SamplingParams | None):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                )

        tasks = [predict_with_semaphore(*args) for args in zip(images, prompts, sampling_params)]
        return await asyncio.gather(*tasks)
