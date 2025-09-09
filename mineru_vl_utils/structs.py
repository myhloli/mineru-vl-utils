from typing import Literal


class BlockType:
    TEXT = "text"  # 文本
    TITLE = "title"  # 段落标题
    TABLE = "table"  # 表格
    IMAGE = "image"  # 图像
    CODE = "code"  # 代码
    ALGORITHM = "algorithm"  # 算法/伪代码
    HEADER = "header"  # 页眉
    FOOTER = "footer"  # 页脚
    PAGE_NUMBER = "page_number"  # 页码
    PAGE_FOOTNOTE = "page_footnote"  # 脚注
    ASIDE_TEXT = "aside_text"  # 侧栏文本(装订线等)
    EQUATION = "equation"  # 公式(独立或行内)
    EQUATION_BLOCK = "equation_block"  # 公式块(多行公式)
    REF_TEXT = "ref_text"  # 参考文献(一条)
    LIST = "list"  # 列表块(无序/有序列表)
    PHONETIC = "phonetic"  # 注音符号

    # captions
    TABLE_CAPTION = "table_caption"  # 表格标题
    IMAGE_CAPTION = "image_caption"  # 图像标题
    CODE_CAPTION = "code_caption"  # 代码标题
    TABLE_FOOTNOTE = "table_footnote"  # 表格脚注
    IMAGE_FOOTNOTE = "image_footnote"  # 图像脚注

    UNKNOWN = "unknown"  # 未知块


BLOCK_TYPES = set(
    [
        BlockType.TEXT,
        BlockType.TITLE,
        BlockType.TABLE,
        BlockType.IMAGE,
        BlockType.CODE,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.PAGE_FOOTNOTE,
        BlockType.ASIDE_TEXT,
        BlockType.EQUATION,
        BlockType.EQUATION_BLOCK,
        BlockType.REF_TEXT,
        BlockType.TABLE_CAPTION,
        BlockType.IMAGE_CAPTION,
        BlockType.TABLE_FOOTNOTE,
        BlockType.IMAGE_FOOTNOTE,
        BlockType.ALGORITHM,
        BlockType.CODE_CAPTION,
        BlockType.LIST,
        BlockType.PHONETIC,
        BlockType.UNKNOWN,
    ]
)


CONTENT_FORMATS = set(
    [
        "text",
        "markdown",
        "html",  # for table
        "latex",
        "csv",
        "otsl",  # for table
    ]
)

ANGLE_OPTIONS = set(
    [
        None,
        0,
        90,
        180,
        270,
    ]
)


class LayoutBlock(dict):
    def __init__(
        self,
        type: str,
        bbox: list[float],
        angle: Literal[None, 0, 90, 180, 270] = None,
        score: float | None = None,
        block_tags: list[str] | None = None,
    ):
        """
        Initialize a layout block.
        Args:
            type (str): Type of the block (e.g., 'text', 'image', 'table').
            bbox (list[float]): Bounding box coordinates [xmin, ymin, xmax, ymax].
            score (float | None): Confidence score for the detection, optional.
        """
        super().__init__()

        assert type in BLOCK_TYPES, f"Unknown type: {type}"
        assert isinstance(bbox, list) and len(bbox) == 4, "Bounding box must be a list of four coordinates"
        assert all(isinstance(coord, (int, float)) for coord in bbox), "Bounding box coordinates must be numbers"
        assert all(0 <= coord <= 1 for coord in bbox), "Bounding box coordinates must be in the range [0, 1]"
        assert bbox[0] < bbox[2], "Bounding box x1 must be less than x2"
        assert bbox[1] < bbox[3], "Bounding box y1 must be less than y2"
        assert angle in ANGLE_OPTIONS, f"Invalid angle: {angle}. Must be one of {ANGLE_OPTIONS}"
        assert score is None or (0 < score <= 1), "Score must be None or in the range (0, 1]"
        assert block_tags is None or isinstance(block_tags, list), "Block tags must be a list or None"
        assert all(isinstance(tag, str) for tag in block_tags or []), "All block tags must be strings"

        self["type"] = type
        self["bbox"] = bbox
        self["angle"] = angle
        self["score"] = score
        self["block_tags"] = block_tags

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
    def score(self) -> float | None:
        return self["score"]

    @score.setter
    def score(self, value: float | None):
        assert value is None or (0 < value <= 1), "Score must be None or in the range (0, 1]"
        self["score"] = value

    @property
    def block_tags(self) -> list[str] | None:
        return self["block_tags"]

    @block_tags.setter
    def block_tags(self, value: list[str] | None):
        assert value is None or isinstance(value, list), "Block tags must be a list or None"
        assert all(isinstance(tag, str) for tag in value or []), "All block tags must be strings"
        self["block_tags"] = value


class ContentBlock(LayoutBlock):
    def __init__(
        self,
        type: str,
        bbox: list[float],
        angle: Literal[None, 0, 90, 180, 270] = None,
        score: float | None = None,
        content: str | None = None,
        format: str | None = None,
        block_tags: list[str] | None = None,
        content_tags: list[str] | None = None,
    ):
        """
        Initialize a content block.
        Args:
            type (str): Type of the block (e.g., 'text', 'image', 'table').
            bbox (list[float]): Bounding box coordinates [xmin, ymin, xmax, ymax].
            content (str): The content of the block.
            score (float | None): Confidence score for the detection, optional.
            format (str): Format of the content, default is 'markdown'.
        """
        super().__init__(type, bbox, angle, score, block_tags)

        assert content is None or isinstance(content, str), "Content must be a string or None"
        assert format is None or format in CONTENT_FORMATS, f"Unknown content format: {format}"
        assert content_tags is None or isinstance(content_tags, list), "Content tags must be a list or None"
        assert all(isinstance(tag, str) for tag in content_tags or []), "All content tags must be strings"

        self["content"] = content
        self["format"] = format
        self["content_tags"] = content_tags

    @property
    def content(self) -> str | None:
        return self["content"]

    @content.setter
    def content(self, value: str | None):
        assert value is None or isinstance(value, str), "Content must be a string or None"
        self["content"] = value

    @property
    def format(self) -> str | None:
        return self["format"]

    @format.setter
    def format(self, value: str | None):
        assert value is None or value in CONTENT_FORMATS, f"Unknown content format: {value}"
        self["format"] = value

    @property
    def content_tags(self) -> list[str] | None:
        return self["content_tags"]

    @content_tags.setter
    def content_tags(self, value: list[str] | None):
        assert value is None or isinstance(value, list), "Content tags must be a list or None"
        assert all(isinstance(tag, str) for tag in value or []), "All content tags must be strings"
        self["content_tags"] = value
