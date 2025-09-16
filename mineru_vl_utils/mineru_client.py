import asyncio
import math
import re
from concurrent.futures import Executor
from typing import Literal, Sequence

from PIL import Image

from .post_process import post_process
from .structs import BLOCK_TYPES, ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, new_vlm_client
from .vlm_client.utils import get_rgb_image

_coord_re = r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$"
_layout_re = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$"
_parsing_re = r"^\s*<\|box_start\|>(.+?)<\|box_end\|><\|ref_start\|>(.+?)<\|ref_end\|><\|md_start\|>(.*?)<\|md_end\|>\s*"

DEFAULT_PROMPTS = {
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "[layout]": "\nLayout Detection:",
    "[parsing]": "\nDocument Parsing:",
    "[default]": "\nDocument Parsing:",
}
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.01
DEFAULT_TOP_K = 1
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_PRESENCE_PENALTY = 0.0
DEFAULT_NO_REPEAT_NGRAM_SIZE = 100

ANGLE_MAPPING = {
    "<|rotate_up|>": 0,
    "<|rotate_right|>": 90,
    "<|rotate_down|>": 180,
    "<|rotate_left|>": 270,
}


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float] | None:
    bbox = tuple(map(int, bbox))
    if any(coord < 0 or coord > 1000 for coord in bbox):
        return None
    x1, y1, x2, y2 = bbox
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return list(map(lambda num: num / 1000.0, (x1, y1, x2, y2)))


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle  # type: ignore
    return None


class MinerUClientHelper:
    def __init__(
        self,
        prompts: dict[str, str],
        layout_image_size: tuple[int, int],
        min_image_edge: int,
        max_image_edge_ratio: float,
        handle_equation_block: bool,
        abandon_list: bool,
        abandon_paratext: bool,
        debug: bool,
    ) -> None:
        self.prompts = prompts
        self.layout_image_size = layout_image_size
        self.min_image_edge = min_image_edge
        self.max_image_edge_ratio = max_image_edge_ratio
        self.handle_equation_block = handle_equation_block
        self.abandon_list = abandon_list
        self.abandon_paratext = abandon_paratext
        self.debug = debug

    def resize_by_need(self, image: Image.Image) -> Image.Image:
        edge_ratio = max(image.size) / min(image.size)
        if edge_ratio > self.max_image_edge_ratio:
            width, height = image.size
            if width > height:
                new_w, new_h = width, math.ceil(width / self.max_image_edge_ratio)
            else:  # width < height
                new_w, new_h = math.ceil(height / self.max_image_edge_ratio), height
            new_image = Image.new(image.mode, (new_w, new_h), (255, 255, 255))
            new_image.paste(image, (int((new_w - width) / 2), int((new_h - height) / 2)))
            image = new_image
        if min(image.size) < self.min_image_edge:
            scale = self.min_image_edge / min(image.size)
            new_w, new_h = round(image.width * scale), round(image.height * scale)
            image = image.resize((new_w, new_h), Image.Resampling.BICUBIC)
        return image

    def prepare_for_layout(self, image: Image.Image) -> Image.Image:
        image = get_rgb_image(image)
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        return image

    def parse_layout_output(self, output: str) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        for line in output.split("\n"):
            match = re.match(_layout_re, line)
            if not match:
                print(f"Warning: line does not match layout format: {line}")
                continue  # Skip invalid lines
            x1, y1, x2, y2, ref_type, tail = match.groups()
            bbox = _convert_bbox((x1, y1, x2, y2))
            if bbox is None:
                print(f"Warning: invalid bbox in line: {line}")
                continue  # Skip invalid bbox
            ref_type = ref_type.lower()
            if ref_type not in BLOCK_TYPES:
                print(f"Warning: unknown block type in line: {line}")
                continue  # Skip unknown block types
            angle = _parse_angle(tail)
            if angle is None:
                print(f"Warning: no angle found in line: {line}")
            blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        return blocks

    def prepare_for_extract(
        self,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image], list[str], list[int]]:
        image = get_rgb_image(image)
        width, height = image.size
        block_images: list[Image.Image] = []
        prompts: list[str] = []
        indices: list[int] = []
        for idx, block in enumerate(blocks):
            if block.type in ("image", "list", "equation_block"):
                continue  # Skip image blocks.
            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)
            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)
            block_images.append(self.resize_by_need(block_image))
            prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            prompts.append(prompt)
            indices.append(idx)
        return block_images, prompts, indices

    def post_process(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        return post_process(
            blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_list=self.abandon_list,
            abandon_paratext=self.abandon_paratext,
            debug=self.debug,
        )

    def batch_prepare_for_layout(
        self,
        executor: Executor | None,
        images: list[Image.Image],
    ) -> list[Image.Image]:
        if executor is None:
            return [self.prepare_for_layout(im) for im in images]
        return list(executor.map(self.prepare_for_layout, images))

    def batch_parse_layout_output(
        self,
        executor: Executor | None,
        outputs: list[str],
    ) -> list[list[ContentBlock]]:
        if executor is None:
            return [self.parse_layout_output(output) for output in outputs]
        return list(executor.map(self.parse_layout_output, outputs))

    def batch_prepare_for_extract(
        self,
        executor: Executor | None,
        images: list[Image.Image],
        blocks_list: list[list[ContentBlock]],
    ) -> list[tuple[list[Image.Image], list[str], list[int]]]:
        if executor is None:
            return [self.prepare_for_extract(im, bls) for im, bls in zip(images, blocks_list)]
        return list(executor.map(self.prepare_for_extract, images, blocks_list))

    def batch_post_process(
        self,
        executor: Executor | None,
        blocks_list: list[list[ContentBlock]],
    ) -> list[list[ContentBlock]]:
        if executor is None:
            return [self.post_process(blocks) for blocks in blocks_list]
        return list(executor.map(self.post_process, blocks_list))

    async def aio_prepare_for_layout(
        self,
        executor: Executor | None,
        image: Image.Image,
    ) -> Image.Image:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_layout, image)

    async def aio_parse_layout_output(
        self,
        executor: Executor | None,
        output: str,
    ) -> list[ContentBlock]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.parse_layout_output, output)

    async def aio_prepare_for_extract(
        self,
        executor: Executor | None,
        image: Image.Image,
        blocks: list[ContentBlock],
    ) -> tuple[list[Image.Image], list[str], list[int]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.prepare_for_extract, image, blocks)

    async def aio_post_process(
        self,
        executor: Executor | None,
        blocks: list[ContentBlock],
    ) -> list[ContentBlock]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(executor, self.post_process, blocks)


class MinerUClient:
    def __init__(
        self,
        backend: Literal["http-client", "transformers", "vllm-engine", "vllm-async-engine"],
        model_name: str | None = None,
        server_url: str | None = None,
        model=None,  # transformers model
        processor=None,  # transformers processor
        vllm_llm=None,  # vllm.LLM model
        vllm_async_llm=None,  # vllm.v1.engine.async_llm.AsyncLLM instance
        model_path: str | None = None,
        prompts: dict[str, str] = DEFAULT_PROMPTS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float | None = DEFAULT_TEMPERATURE,
        top_p: float | None = DEFAULT_TOP_P,
        top_k: int | None = DEFAULT_TOP_K,
        repetition_penalty: float | None = DEFAULT_REPETITION_PENALTY,
        presence_penalty: float | None = DEFAULT_PRESENCE_PENALTY,
        no_repeat_ngram_size: int | None = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        max_new_tokens: int | None = None,
        max_concurrency: int = 100,
        executor: Executor | None = None,
        batch_size: int = 1,  # for transformers backend only
        http_timeout: int = 600,  # for http-client backend only
        debug: bool = False,
    ) -> None:
        if backend == "transformers":
            if model is None or processor is None:
                if not model_path:
                    raise ValueError("model_path must be provided when model or processor is None.")

                try:
                    from transformers import (
                        AutoProcessor,
                        Qwen2VLForConditionalGeneration,
                    )
                    from transformers import __version__ as transformers_version
                except ImportError:
                    raise ImportError("Please install transformers to use the transformers backend.")

                if model is None:
                    dtype_key = "torch_dtype"
                    ver_parts = transformers_version.split(".")
                    if len(ver_parts) >= 2 and int(ver_parts[0]) >= 4 and int(ver_parts[1]) >= 56:
                        dtype_key = "dtype"
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        device_map="auto",
                        **{dtype_key: "auto"},  # type: ignore
                    )
                if processor is None:
                    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

        elif backend == "vllm-engine":
            if vllm_llm is None:
                if not model_path:
                    raise ValueError("model_path must be provided when vllm_llm is None.")

                try:
                    import vllm
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-engine backend.")

                vllm_llm = vllm.LLM(model_path)

        elif backend == "vllm-async-engine":
            if vllm_async_llm is None:
                if not model_path:
                    raise ValueError("model_path must be provided when vllm_async_llm is None.")

                try:
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.v1.engine.async_llm import AsyncLLM
                except ImportError:
                    raise ImportError("Please install vllm to use the vllm-async-engine backend.")

                vllm_async_llm = AsyncLLM.from_engine_args(AsyncEngineArgs(model_path))

        self.client = new_vlm_client(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            model=model,
            processor=processor,
            vllm_llm=vllm_llm,
            vllm_async_llm=vllm_async_llm,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            allow_truncated_content=True,  # Allow truncated content for MinerU
            batch_size=batch_size,
            http_timeout=http_timeout,
            debug=debug,
        )
        self.helper = MinerUClientHelper(
            prompts=prompts,
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            handle_equation_block=handle_equation_block,
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            debug=debug,
        )
        self.prompts = prompts
        self.max_concurrency = max_concurrency
        self.executor = executor

        if backend in ("http-client", "vllm-async-engine"):
            self.batching_mode = "concurrent"
        else:  # backend in ("transformers", "vllm-engine")
            self.batching_mode = "stepping"

    def layout_detect(self, image: Image.Image) -> list[ContentBlock]:
        image = self.helper.prepare_for_layout(image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        output = self.client.predict(image, prompt)
        return self.helper.parse_layout_output(output)

    def batch_layout_detect(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        images = self.helper.batch_prepare_for_layout(self.executor, images)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        outputs = self.client.batch_predict(images, prompt)
        return self.helper.batch_parse_layout_output(self.executor, outputs)

    async def aio_layout_detect(self, image: Image.Image) -> list[ContentBlock]:
        image = await self.helper.aio_prepare_for_layout(self.executor, image)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        output = await self.client.aio_predict(image, prompt)
        return await self.helper.aio_parse_layout_output(self.executor, output)

    async def aio_batch_layout_detect(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        images = await asyncio.gather(*[self.helper.aio_prepare_for_layout(self.executor, im) for im in images])
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        outputs = await self.client.aio_batch_predict(images, prompt)
        return await asyncio.gather(*[self.helper.aio_parse_layout_output(self.executor, out) for out in outputs])

    def one_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        prompt = self.prompts.get("[parsing]") or self.prompts["[default]"]
        output = self.client.predict(image, prompt)
        blocks: list[ContentBlock] = []
        while len(output) > 0:
            match = re.search(_parsing_re, output, re.DOTALL)
            if match is None:
                break  # No more blocks found in the output
            box_string = match.group(1).replace("<|txt_contd|>", "")
            box_match = re.match(_coord_re, box_string.strip())
            if box_match is None:
                continue  # Invalid box format
            bbox = _convert_bbox(box_match.groups())
            if bbox is None:
                continue  # Invalid bbox format
            ref_type = match.group(2).replace("<|txt_contd|>", "").strip()
            ref_type = ref_type.lower()
            content = match.group(3)
            blocks.append(ContentBlock(ref_type, bbox, content=content))
            output = output[len(match.group(0)) :]
        return self.helper.post_process(blocks)

    def two_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        blocks = self.layout_detect(image)
        block_images, prompts, indices = self.helper.prepare_for_extract(image, blocks)
        outputs = self.client.batch_predict(block_images, prompts)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        return self.helper.post_process(blocks)

    async def aio_two_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        blocks = await self.aio_layout_detect(image)
        block_images, prompts, indices = await self.helper.aio_prepare_for_extract(self.executor, image, blocks)
        outputs = await self.client.aio_batch_predict(block_images, prompts)
        for idx, output in zip(indices, outputs):
            blocks[idx].content = output
        return await self.helper.aio_post_process(self.executor, blocks)

    def concurrent_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_concurrent_two_step_extract(images)

        if loop is not None:
            return loop.run_until_complete(task)
        else:
            return asyncio.run(task)

    async def aio_concurrent_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def extract_with_semaphore(image: Image.Image):
            async with semaphore:
                return await self.aio_two_step_extract(image=image)

        return await asyncio.gather(*[extract_with_semaphore(im) for im in images])

    def stepping_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        blocks_list = self.batch_layout_detect(images)
        all_images: list[Image.Image] = []
        all_prompts: list[str] = []
        all_indices: list[tuple[int, int]] = []
        prepared_inputs = self.helper.batch_prepare_for_extract(self.executor, images, blocks_list)
        for img_idx, (block_images, prompts, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = self.client.batch_predict(all_images, all_prompts)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        return self.helper.batch_post_process(self.executor, blocks_list)

    async def aio_stepping_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        blocks_list = await self.aio_batch_layout_detect(images)
        all_images: list[Image.Image] = []
        all_prompts: list[str] = []
        all_indices: list[tuple[int, int]] = []
        tasks = [self.helper.aio_prepare_for_extract(self.executor, im, bls) for im, bls in zip(images, blocks_list)]
        prepared_inputs = await asyncio.gather(*tasks)
        for img_idx, (block_images, prompts, indices) in enumerate(prepared_inputs):
            all_images.extend(block_images)
            all_prompts.extend(prompts)
            all_indices.extend([(img_idx, idx) for idx in indices])
        outputs = await self.client.aio_batch_predict(all_images, all_prompts)
        for (img_idx, idx), output in zip(all_indices, outputs):
            blocks_list[img_idx][idx].content = output
        return await asyncio.gather(*[self.helper.aio_post_process(self.executor, blocks) for blocks in blocks_list])

    def batch_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        if self.batching_mode == "concurrent":
            return self.concurrent_two_step_extract(images)
        else:  # self.batching_mode == "stepping"
            return self.stepping_two_step_extract(images)

    async def aio_batch_two_step_extract(self, images: list[Image.Image]) -> list[list[ContentBlock]]:
        if self.batching_mode == "concurrent":
            return await self.aio_concurrent_two_step_extract(images)
        else:  # self.batching_mode == "stepping"
            return await self.aio_stepping_two_step_extract(images)
