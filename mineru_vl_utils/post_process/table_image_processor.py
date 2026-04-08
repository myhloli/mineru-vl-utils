import base64
import random
import re
from io import BytesIO
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from ..structs import ContentBlock

FONT_PATH_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",                         # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",      # macOS
    "/Library/Fonts/Arial.ttf",                           # macOS 备选
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",            # Linux 备选
]


TABLE_IMAGE_TOKEN_TEMPLATE = "[{idx}]"
TABLE_IMAGE_TOKEN_LETTERS = "ACDGHKTWXYZ"
TABLE_IMAGE_TOKEN_NUMBERS = "2345678"
TABLE_IMAGE_TOKEN_LENGTH = 4
TABLE_IMAGE_TOKEN_CHARS = TABLE_IMAGE_TOKEN_LETTERS + TABLE_IMAGE_TOKEN_NUMBERS
TABLE_IMAGE_TOKEN_MAP_KEY = "_table_image_token_map"
TABLE_IMAGE_ABSORBED_KEY = "_absorbed_by_table"


def _normalize_rotation_angle(angle: int | None) -> int:
    return angle if angle in {90, 180, 270} else 0


def _rotate_image_by_angle(image: Image.Image, angle: int | None) -> Image.Image:
    normalized_angle = _normalize_rotation_angle(angle)
    if normalized_angle == 0:
        return image
    return image.rotate(normalized_angle, expand=True)


def _rotate_box_in_image(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    angle: int | None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width, height = image_size
    normalized_angle = _normalize_rotation_angle(angle)
    if normalized_angle == 0:
        return box
    if normalized_angle == 90:
        return (y1, width - x2, y2, width - x1)
    if normalized_angle == 180:
        return (width - x2, height - y2, width - x1, height - y1)
    return (height - y2, x1, height - y1, x2)


def _load_font(size: int):
    for path in FONT_PATH_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _get_optimal_pil_font(
    text: str,
    box_w: int,
    box_h: int,
    fill_ratio: float = 0.7,
    min_size: int = 4,
    max_size: int = 256,
):
    left, right = min_size, max_size
    best_font = _load_font(left)
    best_w = 0
    best_h = 0

    for _ in range(30):
        if left > right:
            break
        mid = (left + right) // 2
        font = _load_font(mid)
        try:
            bbox = font.getbbox(text)
        except AttributeError:
            w, h = font.getsize(text)
            bbox = (0, 0, w, h)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if w <= box_w * fill_ratio and h <= box_h * fill_ratio:
            best_font = font
            best_w = w
            best_h = h
            left = mid + 1
        else:
            right = mid - 1

    return best_font, best_w, best_h


def _get_average_color(image: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    try:
        left, upper, right, lower = box
        width, height = image.size
        pad = 2
        mid_x = (left + right) // 2
        mid_y = (upper + lower) // 2
        points = [
            (left - pad, upper - pad),
            (mid_x, upper - pad),
            (right + pad, upper - pad),
            (right + pad, mid_y),
            (right + pad, lower + pad),
            (mid_x, lower + pad),
            (left - pad, lower + pad),
            (left - pad, mid_y),
        ]
        pixels: list[tuple[int, int, int]] = []
        for px, py in points:
            px = max(0, min(int(px), width - 1))
            py = max(0, min(int(py), height - 1))
            pixel = image.getpixel((px, py))
            if isinstance(pixel, int):
                pixels.append((pixel, pixel, pixel))
            elif len(pixel) >= 3:
                pixels.append(tuple(pixel[:3]))
        if not pixels:
            return (255, 255, 255)
        r = sum(pixel[0] for pixel in pixels) // len(pixels)
        g = sum(pixel[1] for pixel in pixels) // len(pixels)
        b = sum(pixel[2] for pixel in pixels) // len(pixels)
        return (r, g, b)
    except Exception:
        return (255, 255, 255)


def _get_contrast_text_color(bg_color: tuple[int, int, int]) -> tuple[int, int, int]:
    r, g, b = bg_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def _bbox_intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _bbox_area(a: Sequence[float]) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def _overlap_ratio(inner: Sequence[float], outer: Sequence[float]) -> float:
    inner_area = _bbox_area(inner)
    if inner_area == 0:
        return 0.0
    return _bbox_intersection_area(inner, outer) / inner_area


def _table_area(block: ContentBlock) -> float:
    return _bbox_area(block.bbox)


def _generate_uid(length: int = TABLE_IMAGE_TOKEN_LENGTH) -> str:
    return "".join(random.choices(TABLE_IMAGE_TOKEN_CHARS, k=length))


def _pil_image_to_jpg_data_uri(image: Image.Image) -> str:
    with BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def build_table_image_map(
    blocks: Sequence[ContentBlock],
    threshold: float = 0.9,
    table_indices: Sequence[int] | None = None,
) -> dict[int, list[int]]:
    if table_indices is None:
        table_indices = [idx for idx, block in enumerate(blocks) if block.type == "table"]
    table_indices = list(table_indices)
    table_to_images = {table_idx: [] for table_idx in table_indices}
    if not table_indices:
        return table_to_images

    for image_idx, block in enumerate(blocks):
        if block.type != "image":
            continue
        best_table_idx: int | None = None
        best_ratio = threshold
        best_area: float | None = None
        for table_idx in table_indices:
            table_block = blocks[table_idx]
            ratio = _overlap_ratio(block.bbox, table_block.bbox)
            if ratio < threshold:
                continue
            area = _table_area(table_block)
            if (
                best_table_idx is None
                or ratio > best_ratio
                or (ratio == best_ratio and best_area is not None and area < best_area)
            ):
                best_table_idx = table_idx
                best_ratio = ratio
                best_area = area
        if best_table_idx is not None:
            table_to_images[best_table_idx].append(image_idx)

    for table_idx, image_indices in table_to_images.items():
        image_indices.sort(key=lambda image_idx: (blocks[image_idx].bbox[1], blocks[image_idx].bbox[0]))
    return table_to_images


def mark_absorbed_table_images(blocks: Sequence[ContentBlock], image_indices: Sequence[int]) -> None:
    for image_idx in image_indices:
        blocks[image_idx][TABLE_IMAGE_ABSORBED_KEY] = True


def is_absorbed_table_image(block: ContentBlock) -> bool:
    return bool(block.get(TABLE_IMAGE_ABSORBED_KEY))


def replace_table_image_tokens(content: str | None, token_map: dict[str, str] | None) -> str | None:
    if not content or not token_map:
        return content

    for token, data_uri in token_map.items():
        token_inner = token[1:-1]
        pattern = r"\[\s*" + re.escape(token_inner) + r"\s*\]"
        content = re.sub(pattern, f'<img src="{data_uri}"/>', content)
    return content


def replace_table_formula_delimiters(
    content: str | None,
    enabled: bool = False,
) -> str | None:
    if not enabled or not content:
        return content

    inline_pattern = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
    block_pattern = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
    eq_tag_pattern = re.compile(r"(<eq>.*?</eq>)", re.DOTALL)

    def _wrap_formula(pattern: re.Pattern[str], text: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            inner_content = match.group(1).strip()
            return f"<eq>{inner_content}</eq>"

        return pattern.sub(_replace, text)

    parts = eq_tag_pattern.split(content)
    for idx, part in enumerate(parts):
        if not part or eq_tag_pattern.fullmatch(part):
            continue
        part = _wrap_formula(inline_pattern, part)
        part = _wrap_formula(block_pattern, part)
        parts[idx] = part
    return "".join(parts)


def cleanup_table_image_metadata(blocks: Sequence[ContentBlock]) -> list[ContentBlock]:
    for block in blocks:
        block.pop(TABLE_IMAGE_TOKEN_MAP_KEY, None)
        block.pop(TABLE_IMAGE_ABSORBED_KEY, None)
    return list(blocks)


def mask_and_encode_table_image(
    page_image: Image.Image,
    table_block: ContentBlock,
    image_entries: Sequence[tuple[int, ContentBlock]],
    table_image: Image.Image,
) -> tuple[Image.Image, dict[str, str]]:
    width, height = page_image.size
    x1_t, y1_t, _, _ = table_block.bbox
    abs_x1_t = int(x1_t * width)
    abs_y1_t = int(y1_t * height)
    original_table_size = table_image.size
    masked_table_image = _rotate_image_by_angle(table_image.copy(), table_block.angle)
    draw = ImageDraw.Draw(masked_table_image)
    token_map: dict[str, str] = {}
    used_token_codes: set[str] = set()
    max_token_count = len(TABLE_IMAGE_TOKEN_CHARS) ** TABLE_IMAGE_TOKEN_LENGTH
    font_cache: dict[tuple[int, int], tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, int, int]] = {}

    def get_font_for_box(box_w: int, box_h: int, token_text: str):
        bucket_h = int(box_h // 16)
        key = (bucket_h, len(token_text))
        if key in font_cache:
            font, text_w, text_h = font_cache[key]
            if text_w <= box_w and text_h <= box_h:
                return font, text_w, text_h
        font, text_w, text_h = _get_optimal_pil_font(
            token_text,
            box_w,
            box_h,
            fill_ratio=0.7,
            min_size=4,
            max_size=max(100, int(box_h * 0.7)),
        )
        font_cache[key] = (font, text_w, text_h)
        return font, text_w, text_h

    for _, image_block in image_entries:
        ix1, iy1, ix2, iy2 = image_block.bbox
        abs_ix1 = ix1 * width
        abs_iy1 = iy1 * height
        abs_ix2 = ix2 * width
        abs_iy2 = iy2 * height

        rel_x1 = int(max(0, abs_ix1 - abs_x1_t))
        rel_y1 = int(max(0, abs_iy1 - abs_y1_t))
        rel_x2 = int(min(original_table_size[0], abs_ix2 - abs_x1_t))
        rel_y2 = int(min(original_table_size[1], abs_iy2 - abs_y1_t))
        if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
            continue

        crop_box = (int(abs_ix1), int(abs_iy1), int(abs_ix2), int(abs_iy2))
        crop_image = page_image.crop(crop_box)
        if crop_image.width < 1 or crop_image.height < 1:
            continue

        if len(used_token_codes) >= max_token_count:
            raise RuntimeError("Exhausted random table image token space for this table.")

        while True:
            token_code = _generate_uid()
            if token_code not in used_token_codes:
                used_token_codes.add(token_code)
                break

        token_text = TABLE_IMAGE_TOKEN_TEMPLATE.format(idx=token_code)
        rotated_crop_image = _rotate_image_by_angle(crop_image, table_block.angle)
        token_map[token_text] = _pil_image_to_jpg_data_uri(rotated_crop_image)

        image_mask_bbox = _rotate_box_in_image((rel_x1, rel_y1, rel_x2, rel_y2), original_table_size, table_block.angle)
        avg_color = _get_average_color(masked_table_image, image_mask_bbox)
        draw.rectangle(image_mask_bbox, fill=avg_color, outline=None)

        box_w = image_mask_bbox[2] - image_mask_bbox[0]
        box_h = image_mask_bbox[3] - image_mask_bbox[1]
        font, text_w, text_h = get_font_for_box(box_w, box_h, token_text)
        if text_w <= box_w and text_h <= box_h:
            center_x = image_mask_bbox[0] + box_w / 2
            center_y = image_mask_bbox[1] + box_h / 2
            text_pos = (center_x - text_w / 2, center_y - text_h / 2)
            text_color = _get_contrast_text_color(avg_color)
            draw.text(text_pos, token_text, fill=text_color, font=font)

    return masked_table_image, token_map
