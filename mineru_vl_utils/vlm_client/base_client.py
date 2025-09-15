from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from PIL import Image

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_USER_PROMPT = "What is the text in the illustrate?"


class UnsupportedError(NotImplementedError):
    pass


class RequestError(ValueError):
    pass


class ServerError(RuntimeError):
    pass


@dataclass
class SamplingParams:
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    presence_penalty: float | None
    no_repeat_ngram_size: int | None
    max_new_tokens: int | None


class VlmClient:
    def __init__(
        self,
        *,
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        no_repeat_ngram_size: int | None = None,
        max_new_tokens: int | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
    ) -> None:
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.text_before_image = text_before_image
        self.allow_truncated_content = allow_truncated_content

    def build_sampling_params(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        presence_penalty: Optional[float],
        no_repeat_ngram_size: Optional[int],
        max_new_tokens: Optional[int],
    ) -> SamplingParams:
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if top_k is None:
            top_k = self.top_k
        if repetition_penalty is None:
            repetition_penalty = self.repetition_penalty
        if presence_penalty is None:
            presence_penalty = self.presence_penalty
        if no_repeat_ngram_size is None:
            no_repeat_ngram_size = self.no_repeat_ngram_size
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

    def predict(
        self,
        image: str | bytes | Image.Image,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError()

    def batch_predict(
        self,
        images: List[str] | List[bytes] | List[Image.Image],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        raise NotImplementedError()

    async def aio_predict(
        self,
        image: str | bytes | Image.Image,
        prompt: str = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError()

    async def aio_batch_predict(
        self,
        images: List[str] | List[bytes] | List[Image.Image],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
    ) -> List[str]:
        raise NotImplementedError()


def new_vlm_client(
    backend: Literal["http-client", "transformers", "vllm-engine", "vllm-async-engine"],
    model_name: str | None = None,
    server_url: str | None = None,
    model=None,  # transformers model
    processor=None,  # transformers processor
    vllm_llm=None,  # vllm.LLM model
    vllm_async_llm=None,  # vllm.v1.engine.async_llm.AsyncLLM instance
    prompt: str = DEFAULT_USER_PROMPT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    no_repeat_ngram_size: int | None = None,
    max_new_tokens: int | None = None,
    text_before_image: bool = False,
    allow_truncated_content: bool = False,
    batch_size: int = 1,
    http_timeout: int = 600,
    debug: bool = False,
) -> VlmClient:

    if backend == "http-client":
        from .http_client import HttpVlmClient

        return HttpVlmClient(
            model_name=model_name,
            server_url=server_url,
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
            http_timeout=http_timeout,
            debug=debug,
        )

    elif backend == "transformers":
        from .transformers_client import TransformersVlmClient

        return TransformersVlmClient(
            model=model,
            processor=processor,
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
            batch_size=batch_size,
        )

    elif backend == "vllm-engine":
        from .vllm_engine_client import VllmEngineVlmClient

        return VllmEngineVlmClient(
            vllm_llm=vllm_llm,
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

    elif backend == "vllm-async-engine":
        from .vllm_async_engine_client import VllmAsyncEngineVlmClient

        return VllmAsyncEngineVlmClient(
            vllm_async_llm=vllm_async_llm,
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

    else:
        raise ValueError(f"Unsupported backend: {backend}")
