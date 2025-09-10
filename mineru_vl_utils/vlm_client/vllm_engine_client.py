import asyncio
from typing import List, Optional, Union

from PIL import Image

from .base_client import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, VlmClient
from .utils import aio_load_resource, get_image_data_url, get_png_bytes, load_resource


class VllmEngineVlmClient(VlmClient):
    def __init__(
        self,
        vllm_llm,  # vllm.LLM instance
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
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("Please install vllm to use VllmEngineVlmClient.")

        if not vllm_llm:
            raise ValueError("vllm_llm is None.")
        if not isinstance(vllm_llm, LLM):
            raise ValueError("vllm_llm must be an instance of vllm.LLM.")

        self.vllm_llm = vllm_llm
        self.model_max_length = vllm_llm.llm_engine.model_config.max_model_len
        self.VllmSamplingParams = SamplingParams

    def build_messages(
        self,
        image: bytes,
        prompt: str,
        image_format: str | None,
    ) -> list[dict]:
        image_url = get_image_data_url(image, image_format)
        prompt = prompt or self.prompt
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if "<image>" in prompt:
            prompt_1, prompt_2 = prompt.split("<image>", 1)
            user_messages = [
                *([{"type": "text", "text": prompt_1}] if prompt_1.strip() else []),
                # {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                *([{"type": "text", "text": prompt_2}] if prompt_2.strip() else []),
            ]
        elif self.text_before_image:
            user_messages = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:  # image before text, which is the default behavior.
            user_messages = [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": prompt},
            ]
        messages.append({"role": "user", "content": user_messages})
        return messages

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
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )[0]

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
    ) -> List[str]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."

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
        )

        messages_list = []
        for image, prompt in zip(images, prompts):
            image_format = None
            if isinstance(image, str):
                image = load_resource(image)
            if isinstance(image, Image.Image):
                image = get_png_bytes(image)
                image_format = "png"
            messages = self.build_messages(image, prompt, image_format)
            messages_list.append(messages)

        outputs = self.vllm_llm.chat(messages_list, sampling_params=vllm_sp)
        return [output.outputs[0].text for output in outputs]

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
        return await asyncio.to_thread(
            self.predict,
            image,
            prompt,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )

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
    ) -> List[str]:
        return await asyncio.to_thread(
            self.batch_predict,
            images,
            prompts,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            presence_penalty,
            no_repeat_ngram_size,
            max_new_tokens,
        )
