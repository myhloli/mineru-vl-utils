import asyncio
import uuid
from io import BytesIO
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput

from PIL import Image

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    RequestError,
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
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,  # not supported by vllm
        max_new_tokens: int | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
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
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,  # not supported by vllm
        max_new_tokens: Optional[int] = None,
    ) -> str:
        raise UnsupportedError(
            "Synchronous predict() is not supported in VllmAsyncEngineVlmClient. Please use aio_predict() instead."
        )

    def batch_predict(
        self,
        images: List[str] | List[bytes] | List[Image.Image],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,  # not supported by vllm
        max_new_tokens: Optional[int] = None,
        max_concurrency: int = 100,
    ) -> List[str]:
        raise UnsupportedError(
            "Synchronous batch_predict() is not supported in VllmAsyncEngineVlmClient. "
            "Please use aio_batch_predict() instead."
        )

    async def aio_predict(
        self,
        image: str | bytes | Image.Image,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,  # not supported by vllm
        max_new_tokens: Optional[int] = None,
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

        sp = self.build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        vllm_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "repetition_penalty": sp.repetition_penalty,
            "presence_penalty": sp.presence_penalty,
            "max_tokens": sp.max_new_tokens,
        }

        if sp.temperature is not None:
            vllm_sp_dict["temperature"] = sp.temperature
        if sp.top_p is not None:
            vllm_sp_dict["top_p"] = sp.top_p
        if sp.top_k is not None:
            vllm_sp_dict["top_k"] = sp.top_k
        if sp.repetition_penalty is not None:
            vllm_sp_dict["repetition_penalty"] = sp.repetition_penalty
        if sp.presence_penalty is not None:
            vllm_sp_dict["presence_penalty"] = sp.presence_penalty
        if sp.max_new_tokens is not None:
            vllm_sp_dict["max_tokens"] = sp.max_new_tokens
        else:
            # max_tokens should smaller than model max length
            vllm_sp_dict["max_tokens"] = self.model_max_length

        vllm_sp = self.VllmSamplingParams(
            **{k: v for k, v in vllm_sp_dict.items() if v is not None},
            skip_special_tokens=False,
            output_kind=self.VllmRequestOutputKind.FINAL_ONLY,
        )

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
        images: List[str] | List[bytes] | List[Image.Image],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,  # not supported by vllm
        max_new_tokens: Optional[int] = None,
        max_concurrency: int = 100,
    ) -> List[str]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."

        semaphore = asyncio.Semaphore(max_concurrency)

        async def predict_with_semaphore(
            image: str | bytes | Image.Image,
            prompt: str,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_new_tokens=max_new_tokens,
                )

        tasks = [predict_with_semaphore(*args) for args in zip(images, prompts)]
        return await asyncio.gather(*tasks)
