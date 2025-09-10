import asyncio
import math
import re
from typing import Literal, Sequence

from PIL import Image

from .otsl2html import convert_otsl_to_html
from .structs import ContentBlock
from .vlm_client import DEFAULT_SYSTEM_PROMPT, new_vlm_client

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

PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}


def _convert_bbox(bbox: Sequence[int] | Sequence[str]) -> list[float]:
    x1, y1, x2, y2 = tuple(map(int, bbox))
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    return list(map(lambda num: num / 1000.0, (x1, y1, x2, y2)))


def _parse_angle(tail: str) -> Literal[None, 0, 90, 180, 270]:
    for token, angle in ANGLE_MAPPING.items():
        if token in tail:
            return angle  # type: ignore
    return None


def _bbox_cover_ratio(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if areaB == 0:
        return 0.0
    ratio = interArea / areaB
    return ratio


def _combined_equations(equation_contents):
    combined_content = "\\begin{array}{l} "
    for equation_content in equation_contents:
        combined_content += equation_content + " \\\\ "
    combined_content += "\\end{array}"
    return combined_content


def _post_process(
    blocks: list[ContentBlock],
    handle_equation_block: bool,
    abandon_paratext: bool,
) -> list[ContentBlock]:
    sem_equation_spans: dict[int, list[int]] = {}

    if handle_equation_block:
        sem_equation_indices: list[int] = []
        span_equation_indices: list[int] = []
        for idx, block in enumerate(blocks):
            if block.type == "equation_block":
                sem_equation_indices.append(idx)
            elif block.type == "equation":
                span_equation_indices.append(idx)
        for sem_idx in sem_equation_indices:
            covered_span_indices = [
                span_idx
                for span_idx in span_equation_indices
                if _bbox_cover_ratio(
                    blocks[sem_idx].bbox,
                    blocks[span_idx].bbox,
                )
                > 0.9
            ]
            if len(covered_span_indices) > 1:
                sem_equation_spans[sem_idx] = covered_span_indices

    out_blocks: list[ContentBlock] = []
    for idx in range(len(blocks)):
        block = blocks[idx]
        if any(idx in span_indices for span_indices in sem_equation_spans.values()):
            continue
        if idx in sem_equation_spans:
            span_indices = sem_equation_spans[idx]
            span_equation_contents = [blocks[span_idx].content for span_idx in span_indices]
            sem_equation_content = _combined_equations(span_equation_contents)
            out_blocks.append(
                ContentBlock(
                    type="equation",
                    bbox=block.bbox,
                    angle=block.angle,
                    content=sem_equation_content,
                )
            )
            continue
        if block.type in ["list", "equation_block"]:
            continue
        if abandon_paratext and block.type in PARATEXT_TYPES:
            continue
        out_blocks.append(block)
    return out_blocks


class MinerUClient:
    def __init__(
        self,
        backend: Literal["http-client", "transformers", "vllm-engine"],
        model_name: str | None = None,
        server_url: str | None = None,
        model=None,  # transformers model
        processor=None,  # transformers processor
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
        abandon_paratext: bool = False,
        max_new_tokens: int | None = None,
        http_timeout: int = 600,
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
                except ImportError:
                    raise ImportError("Please install transformers to use the transformers backend.")

                if model is None:
                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_path,
                        torch_dtype="auto",
                        device_map="auto",
                    )
                if processor is None:
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        use_fast=True,
                    )

        self.client = new_vlm_client(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            model=model,
            processor=processor,
            system_prompt=system_prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_new_tokens=max_new_tokens,
            allow_truncated_content=True,  # Allow truncated content for MinerU
            http_timeout=http_timeout,
            debug=debug,
        )
        self.prompts = prompts
        self.layout_image_size = layout_image_size
        self.min_image_edge = min_image_edge
        self.max_image_edge_ratio = max_image_edge_ratio
        self.handle_equation_block = handle_equation_block
        self.abandon_paratext = abandon_paratext

    def _resize_by_need(self, image: Image.Image) -> Image.Image:
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

    def layout_detect(self, image: Image.Image) -> list[ContentBlock]:
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        output = self.client.predict(image, prompt)

        blocks: list[ContentBlock] = []
        for line in output.split("\n"):
            match = re.match(_layout_re, line)
            if not match:
                continue
            x1, y1, x2, y2, ref_type, tail = match.groups()
            ref_type = ref_type.lower()
            bbox = _convert_bbox((x1, y1, x2, y2))
            angle = _parse_angle(tail)
            blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        return blocks

    async def aio_layout_detect(self, image: Image.Image) -> list[ContentBlock]:
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = image.resize(self.layout_image_size, Image.Resampling.BICUBIC)
        prompt = self.prompts.get("[layout]") or self.prompts["[default]"]
        output = await self.client.aio_predict(image, prompt)

        blocks: list[ContentBlock] = []
        for line in output.split("\n"):
            match = re.match(_layout_re, line)
            if not match:
                continue
            x1, y1, x2, y2, ref_type, tail = match.groups()
            ref_type = ref_type.lower()
            bbox = _convert_bbox((x1, y1, x2, y2))
            angle = _parse_angle(tail)
            blocks.append(ContentBlock(ref_type, bbox, angle=angle))
        return blocks

    def one_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        image = image.convert("RGB") if image.mode != "RGB" else image
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
            ref_type = match.group(2).replace("<|txt_contd|>", "").strip()
            ref_type = ref_type.lower()
            content = match.group(3)
            if ref_type == "table":
                content = convert_otsl_to_html(content)
            blocks.append(ContentBlock(ref_type, bbox, content=content))
            output = output[len(match.group(0)) :]
        blocks = _post_process(
            blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_paratext=self.abandon_paratext,
        )
        return blocks

    def two_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        image = image.convert("RGB") if image.mode != "RGB" else image
        width, height = image.size

        blocks = self.layout_detect(image)
        block_images: list[Image.Image] = []
        block_prompts: list[str] = []
        block_indices: list[int] = []
        for idx, block in enumerate(blocks):
            if block.type in ("image", "list", "equation_block"):
                continue  # Skip image blocks.

            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)
            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)

            block_images.append(self._resize_by_need(block_image))
            block_prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            block_prompts.append(block_prompt)
            block_indices.append(idx)

        outputs = self.client.batch_predict(block_images, block_prompts)
        for idx, output in zip(block_indices, outputs):
            block = blocks[idx]
            if block.type == "table":
                output = convert_otsl_to_html(output)
            block.content = output

        blocks = _post_process(
            blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_paratext=self.abandon_paratext,
        )
        return blocks

    async def aio_two_step_extract(self, image: Image.Image) -> list[ContentBlock]:
        image = image.convert("RGB") if image.mode != "RGB" else image
        width, height = image.size

        blocks = await self.aio_layout_detect(image)
        block_images: list[Image.Image] = []
        block_prompts: list[str] = []
        block_indices: list[int] = []
        for idx, block in enumerate(blocks):
            if block.type in ("image", "list", "equation_block"):
                continue  # Skip image blocks.

            x1, y1, x2, y2 = block.bbox
            scaled_bbox = (x1 * width, y1 * height, x2 * width, y2 * height)
            block_image = image.crop(scaled_bbox)
            if block.angle in [90, 180, 270]:
                block_image = block_image.rotate(block.angle, expand=True)

            block_images.append(self._resize_by_need(block_image))
            block_prompt = self.prompts.get(block.type) or self.prompts["[default]"]
            block_prompts.append(block_prompt)
            block_indices.append(idx)

        outputs = await self.client.aio_batch_predict(block_images, block_prompts)
        for idx, output in zip(block_indices, outputs):
            block = blocks[idx]
            if block.type == "table":
                output = convert_otsl_to_html(output)
            block.content = output

        blocks = _post_process(
            blocks,
            handle_equation_block=self.handle_equation_block,
            abandon_paratext=self.abandon_paratext,
        )
        return blocks

    def batch_two_step_extract(
        self,
        images: list[Image.Image],
        max_concurrency: int = 100,
    ) -> list[list[ContentBlock]]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        task = self.aio_batch_two_step_extract(
            images=images,
            max_concurrency=max_concurrency,
        )

        if loop is not None:
            return loop.run_until_complete(task)
        else:
            return asyncio.run(task)

    async def aio_batch_two_step_extract(
        self,
        images: list[Image.Image],
        max_concurrency: int = 100,
    ) -> list[list[ContentBlock]]:
        semaphore = asyncio.Semaphore(max_concurrency)
        outputs = [[]] * len(images)

        async def predict_with_semaphore(idx: int, image: Image.Image):
            async with semaphore:
                output = await self.aio_two_step_extract(image=image)
                outputs[idx] = output

        tasks = []
        for idx, image in enumerate(images):
            tasks.append(predict_with_semaphore(idx, image))
        await asyncio.gather(*tasks)

        return outputs
