from typing import Literal

from .vlm_client.base_client import ScoredOutput


class BlockType:
    TEXT = "text"  # 文本
    TITLE = "title"  # 段落标题
    TABLE = "table"  # 表格
    EQUATION = "equation"  # 公式(独立公式)
    CODE = "code"  # 代码
    ALGORITHM = "algorithm"  # 算法/伪代码
    ASIDE_TEXT = "aside_text"  # 侧栏文本(装订线等)
    REF_TEXT = "ref_text"  # 参考文献(一条)
    PHONETIC = "phonetic"  # 注音符号
    LIST_ITEM = "list_item"  # 列表项(无序/有序列表)
    # captions
    TABLE_CAPTION = "table_caption"  # 表格标题
    IMAGE_CAPTION = "image_caption"  # 图像标题
    CODE_CAPTION = "code_caption"  # 代码标题
    TABLE_FOOTNOTE = "table_footnote"  # 表格脚注
    IMAGE_FOOTNOTE = "image_footnote"  # 图像脚注
    # paratexts
    HEADER = "header"  # 页眉
    FOOTER = "footer"  # 页脚
    PAGE_NUMBER = "page_number"  # 页码
    PAGE_FOOTNOTE = "page_footnote"  # 脚注
    # images
    IMAGE = "image"  # 图像
    CHART = "chart"
    # containers
    LIST = "list"  # 列表块(无序/有序列表)
    IMAGE_BLOCK = "image_block"  # 图像块(多图)
    EQUATION_BLOCK = "equation_block"  # 公式块(多行公式)
    # unknown
    UNKNOWN = "unknown"  # 未知块


BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.TABLE,
    BlockType.EQUATION,
    BlockType.CODE,
    BlockType.ALGORITHM,
    BlockType.ASIDE_TEXT,
    BlockType.REF_TEXT,
    BlockType.PHONETIC,
    BlockType.LIST_ITEM,
    # captions
    BlockType.TABLE_CAPTION,
    BlockType.IMAGE_CAPTION,
    BlockType.CODE_CAPTION,
    BlockType.TABLE_FOOTNOTE,
    BlockType.IMAGE_FOOTNOTE,
    # paratexts
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    # images
    BlockType.IMAGE,
    BlockType.CHART,
    # containers
    BlockType.LIST,
    BlockType.IMAGE_BLOCK,
    BlockType.EQUATION_BLOCK,
    # unknown
    BlockType.UNKNOWN,
}

ANGLE_OPTIONS = {
    None,
    0,
    90,
    180,
    270,
}


class ExtractResult(list["ContentBlock"]):
    """
    list[ContentBlock] subclass returned by two_step_extract() and related methods.
    Backward-compatible: all existing list[ContentBlock] usage works unchanged.

    When scored=True is passed to the extraction method:
    - layout_scored: ScoredOutput for the layout detection step (whole-page score)
    - blocks[i].scored: ScoredOutput for each content block's extraction step
    """

    layout_scored: ScoredOutput | None

    def __init__(self, blocks=()):
        super().__init__(blocks)
        self.layout_scored = None


class ContentBlock(dict):
    def __init__(
        self,
        type: str,
        bbox: list[float],
        angle: Literal[None, 0, 90, 180, 270] = None,
        content: str | None = None,
        merge_type: Literal[None, 'src', 'tgt'] = None,
    ):
        """
        Initialize a layout block.
        Args:
            type (str): Type of the block (e.g., 'text', 'image', 'table').
            bbox (list[float]): Bounding box coordinates [xmin, ymin, xmax, ymax].
            angle (int or None): Rotation angle of the block. Must be one of {None, 0, 90, 180, 270}.
            content (str or None): The content of the block (if exists).
        """
        super().__init__()

        assert type in BLOCK_TYPES, f"Unknown type: {type}"
        assert isinstance(bbox, list) and len(bbox) == 4, "Bounding box must be a list of four coordinates"
        assert all(isinstance(coord, (int, float)) for coord in bbox), "Bounding box coordinates must be numbers"
        assert all(0 <= coord <= 1 for coord in bbox), "Bounding box coordinates must be in the range [0, 1]"
        assert bbox[0] < bbox[2], "Bounding box x1 must be less than x2"
        assert bbox[1] < bbox[3], "Bounding box y1 must be less than y2"
        assert angle in ANGLE_OPTIONS, f"Invalid angle: {angle}. Must be one of {ANGLE_OPTIONS}"
        assert content is None or isinstance(content, str), "Content must be a string or None"
        assert merge_type is None or merge_type in {'src', 'tgt'}, f"Invalid merge type: {merge_type}. Must be one of {'src', 'tgt'} or None"
        self["type"] = type
        self["bbox"] = bbox
        self["angle"] = angle
        self["content"] = content
        self["merge_type"] = merge_type
        self["scored"] = None
        
    @property
    def type(self) -> str:
        return self["type"]

    @type.setter
    def type(self, value: str):
        assert value in BLOCK_TYPES, f"Unknown type: {value}"
        self["type"] = value

    @property
    def bbox(self) -> list[float]:
        return self["bbox"]

    @bbox.setter
    def bbox(self, value: list[float]):
        assert isinstance(value, list) and len(value) == 4, "Bounding box must be a list of four coordinates"
        assert all(isinstance(coord, (int, float)) for coord in value), "Bounding box coordinates must be numbers"
        assert all(0 <= coord <= 1 for coord in value), "Bounding box coordinates must be in the range [0, 1]"
        assert value[0] < value[2], "Bounding box x1 must be less than x2"
        assert value[1] < value[3], "Bounding box y1 must be less than y2"
        self["bbox"] = value

    @property
    def angle(self) -> Literal[None, 0, 90, 180, 270]:
        return self["angle"]

    @angle.setter
    def angle(self, value: Literal[None, 0, 90, 180, 270]):
        assert value in ANGLE_OPTIONS, f"Invalid angle: {value}. Must be one of {ANGLE_OPTIONS}"
        self["angle"] = value

    @property
    def content(self) -> str | None:
        return self["content"]

    @content.setter
    def content(self, value: str | None):
        assert value is None or isinstance(value, str), "Content must be a string or None"
        self["content"] = value

    @property
    def merge_type(self) -> Literal[None, 'src', 'tgt']:
        return self["merge_type"]

    @merge_type.setter
    def merge_type(self, value: Literal[None, 'src', 'tgt']):
        assert value is None or value in {'src', 'tgt'}, f"Invalid merge type: {value}. Must be one of {'src', 'tgt'} or None"
        self["merge_type"] = value

    @property
    def scored(self) -> ScoredOutput | None:
        return self["scored"]

    @scored.setter
    def scored(self, value: ScoredOutput | None):
        self["scored"] = value
