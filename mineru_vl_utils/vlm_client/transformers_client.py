import asyncio
from io import BytesIO
from typing import List, Optional, Union

from PIL import Image
from tqdm import tqdm

from .base_client import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, VlmClient
from .utils import aio_load_resource, get_rgb_image, load_resource


class TransformersVlmClient(VlmClient):
    def __init__(
        self,
        model,  # transformers model
        processor,  # transformers processor
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
        if not model:
            raise ValueError("Model is None.")
        if not hasattr(model, "generate"):
            raise ValueError("Model does not have generate method.")
        if not processor:
            raise ValueError("Processor is None.")
        if not hasattr(processor, "apply_chat_template"):
            raise ValueError("Processor does not have apply_chat_template method.")
        self.model = model
        self.processor = processor
        self.eos_token_id = model.config.eos_token_id
        self.model_max_length = model.config.max_position_embeddings

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
        **kwargs,
    ) -> str:
        sp = self.build_sampling_params(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
        )

        do_sample = ((sp.temperature or 0.0) > 0.0) and ((sp.top_k or 1) > 1)

        # write these three params anyway.
        generate_kwargs = {
            "temperature": sp.temperature if do_sample and sp.temperature is not None else None,
            "top_p": sp.top_p if do_sample and sp.top_p is not None else None,
            "top_k": sp.top_k if do_sample and sp.top_k is not None else None,
        }
        if sp.repetition_penalty is not None:
            generate_kwargs["repetition_penalty"] = sp.repetition_penalty
        if sp.no_repeat_ngram_size is not None:
            generate_kwargs["no_repeat_ngram_size"] = sp.no_repeat_ngram_size
        if sp.max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = sp.max_new_tokens
        else:  # set max_length when max_new_tokens is not set
            generate_kwargs["max_length"] = self.model_max_length
        generate_kwargs["do_sample"] = do_sample

        if isinstance(image, str):
            image = load_resource(image)
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image))
        image = get_rgb_image(image)

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

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(device=self.model.device, dtype=self.model.dtype)

        output_ids = self.model.generate(
            **inputs,
            use_cache=True,
            **generate_kwargs,
            **kwargs,
        )

        output_ids = [ids[len(in_ids) :] for in_ids, ids in zip(inputs.input_ids, output_ids)]
        output_ids = [ids[:-1] if ids[-1:] == self.eos_token_id else ids for ids in output_ids]

        output_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def batch_predict(
        self,
        images: List[str] | List[bytes] | List[Image.Image],
        prompts: Union[List[str], str] = "",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,  # not supported by hf
        no_repeat_ngram_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        batch_size: int = 1,
        **kwargs,
    ) -> List[str]:
        if not isinstance(prompts, list):
            prompts = [prompts] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."

        # TODO: use batch processing

        outputs = []
        for prompt, image in tqdm(zip(prompts, images), total=len(images), desc="Predict"):
            output = self.predict(
                image,
                prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
            outputs.append(output)
        return outputs

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
        no_repeat_ngram_size: Optional[int] = None,
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
