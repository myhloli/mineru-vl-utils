import os
import re
from base64 import b64decode, b64encode
from io import BytesIO

import aiofiles
import httpx
from PIL import Image

from .base_client import RequestError

_timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
_file_exts = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf")
_data_uri_regex = re.compile(r"^data:[^;,]+;base64,")


def load_resource(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        response = httpx.get(uri, timeout=_timeout)
        return response.content
    if uri.startswith("file://"):
        with open(uri[len("file://") :], "rb") as file:
            return file.read()
    if uri.lower().endswith(_file_exts):
        with open(uri, "rb") as file:
            return file.read()
    if re.match(_data_uri_regex, uri):
        return b64decode(uri.split(",")[1])
    return b64decode(uri)


async def aio_load_resource(uri: str) -> bytes:
    if uri.startswith("http://") or uri.startswith("https://"):
        async with httpx.AsyncClient(timeout=_timeout) as client:
            response = await client.get(uri)
            return response.content
    if uri.startswith("file://"):
        async with aiofiles.open(uri[len("file://") :], "rb") as file:
            return await file.read()
    if uri.lower().endswith(_file_exts):
        async with aiofiles.open(uri, "rb") as file:
            return await file.read()
    if re.match(_data_uri_regex, uri):
        return b64decode(uri.split(",")[1])
    return b64decode(uri)


def get_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def get_image_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"\x89PNG"):
        return "png"
    if image_bytes.startswith(b"GIF8"):
        return "gif"
    if image_bytes.startswith(b"BM"):
        return "bmp"
    if image_bytes[0:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes.startswith(b"II\x2a\x00") or image_bytes.startswith(b"MM\x00\x2a"):
        return "tiff"
    raise RequestError("Unsupported image format.")


def get_image_data_url(image_bytes: bytes, image_format: str | None) -> str:
    image_base64 = b64encode(image_bytes).decode("utf-8")
    if not image_format:
        image_format = get_image_format(image_bytes)
    return f"data:image/{image_format};base64,{image_base64}"


def get_rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "P":
        image = image.convert("RGBA")
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
