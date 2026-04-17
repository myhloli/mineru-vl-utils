import html
import re

from loguru import logger

IMAGE_CHART_FIELD_TAGS = {
    "class": ("<|class_start|>", "<|class_end|>"),
    "sub_class": ("<|sub_class_start|>", "<|sub_class_end|>"),
    "caption": ("<|caption_start|>", "<|caption_end|>"),
    "content": ("<|content_start|>", "<|content_end|>"),
}

CHART_SUB_CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    # 键名为 chart-types.md 中定义的短形式；变体均小写，由 _canonicalize_chart_sub_class 统一处理
    "line": ("line chart", "line graph", "line plot", "line"),  # 折线图
    "bar": (  # 柱状图
        "bar chart",
        "column chart",
        "column bar chart",
        "vertical bar chart",
        "column",
        "bar",
        "bar graph",
        "vertical bar",
    ),
    "pie": ("pie chart", "pie graph", "pie"),  # 饼图
    "scatter": ("scatterplot", "scatter plot", "scatter chart", "scatter graph", "scatter"),  # 散点图
    "histogram": ("histogram", "bar histogram", "frequency histogram"),  # 直方图
    "heatmap": ("heatmap", "heat map", "heat chart", "correlation heatmap"),  # 热力图
    "bar_stacked": (  # 堆叠柱状图（含百分比堆叠、水平堆叠）
        "stacked bar chart",
        "stacked bar",
        "stacked column chart",
        "percent stacked bar chart",
        "100 percent stacked bar chart",
        "percentage stacked bar",
        "horizontal stacked bar chart",
        "stacked bar graph",
        "stacked horizontal bar",
        "stacked column",
        "100 percent stacked bar",
    ),
    "donut": ("donut chart", "doughnut chart", "doughnut graph", "donut", "doughnut"),  # 环形图
    "area": ("area chart", "area graph", "area plot", "area"),  # 面积图
    "area_stacked": (  # 堆叠面积图
        "stacked area chart",
        "stacked area",
        "stacked area graph",
        "3d area chart",
        "3d area",
    ),
    "bubble": ("bubble chart", "bubble plot", "bubble scatter", "bubble"),  # 气泡图
    "boxplot": (  # 箱线图
        "box plot",
        "boxplot",
        "box chart",
        "box and whisker plot",
        "box and whisker",
        "box whisker",
        "whisker plot",
    ),
    "violin": ("violin plot", "violin chart", "violin graph", "violin"),  # 小提琴图
    "radar": ("radar chart", "spider chart", "spider web chart", "radar graph", "radar"),  # 雷达图
    "bar_line": (  # 柱线混合图
        "bar-line hybrid",
        "bar line hybrid",
        "bar line",
        "bar-line hybrid chart",
        "bar and line chart",
        "combined bar line chart",
        "combo chart",
        "dual axis chart",
        "mixed bar line chart",
        "bar line combination chart",
        "combination chart",
    ),
    "funnel": ("funnel chart", "funnel graph", "funnel"),  # 漏斗图
    "geo": (  # 地理图表
        "geospatial charts",
        "geospatial chart",
        "geospatial",
        "map chart",
        "choropleth map",
        "choropleth chart",
        "choropleth",
        "world map",
        "regional map",
        "country map",
        "geographic map",
        "geo map",
        "heat map geographic",
        "world map with table",
        "world map with bubble markers",
    ),
    "gauge": ("gauge chart", "gauge graph", "gauge", "speedometer chart", "dial chart"),  # 仪表盘
    "waterfall": ("waterfall chart", "waterfall graph", "waterfall"),  # 瀑布图
    "treemap": ("treemap", "tree map", "tree map chart", "treemap chart"),  # 矩形树图
    "sankey": ("sankey chart", "sankey diagram", "sankey graph", "sankey"),  # 桑基图
    "network": ("network chart", "network graph", "network diagram", "network"),  # 网络关系图
    "tree": (  # 树形结构图
        "tree chart",
        "tree diagram",
        "tree graph",
        "tree structure",
        "hierarchy tree",
        "hierarchical tree",
        "tree",
    ),
    "word_cloud": ("word cloud", "tag cloud", "word cloud chart"),  # 词云图
    "sunburst": ("sunburst chart", "sunburst graph", "sunburst"),  # 旭日图
    "candlestick": (  # 蜡烛图/K线图
        "candlestick chart",
        "candle chart",
        "ohlc chart",
        "k-line chart",
        "k line chart",
        "candlestick",
    ),
    "polar": ("polar chart", "polar plot", "polar graph"),  # 极坐标图
    "polar_bar": ("rose chart", "polar bar chart", "radial bar chart", "wind rose chart", "wind rose"),  # 玫瑰图
    "contour": ("contour plot", "contour chart", "contour map", "contour graph", "contour"),  # 等高线图
    "surface_3d": ("surface plot", "3d surface plot", "3d surface chart", "3d surface"),  # 三维曲面图
    "scatter_3d": ("3d scatter plot", "3d scatter chart", "3d scatter"),  # 三维散点图
    "dendrogram": ("dendrogram", "dendrogram chart", "cluster dendrogram", "cluster tree"),  # 层次聚类树
    "pairplot": ("scatter matrix", "pair plot", "pairplot"),  # 散点矩阵
    "hexbin": ("hexbin plot", "hexbin chart", "hex bin plot", "hexbin"),  # 六边形密度图
    "qq": ("q-q plot", "qq plot", "quantile-quantile plot", "quantile plot"),  # Q-Q 图
    "roc": ("roc curve", "roc chart", "roc curve chart", "receiver operating characteristic"),  # ROC 曲线
    "confusion": ("confusion matrix", "confusion matrix chart", "confusion chart"),  # 混淆矩阵
    "forest": ("forest plot", "forest diagram", "meta analysis plot"),  # 森林图
    "manhattan": ("manhattan plot", "manhattan chart", "manhattan graph"),  # 曼哈顿图
    "volcano": ("volcano plot", "volcano chart", "volcano graph"),  # 火山图
    "survival": ("survival curve", "kaplan meier curve", "kaplan-meier curve", "survival plot"),  # 生存曲线
    "spectrogram": ("spectrogram", "spectrum chart", "spectrogram chart"),  # 频谱图
    "other": (  # 兜底类型
        "complex & scientific",
        "complex and scientific",
        "complex scientific",
        "scientific",
        "other",
    ),
}


def _canonicalize_chart_sub_class(sub_class: str) -> str:
    normalized_sub_class = re.sub(r"\s+", " ", sub_class).strip().lower()
    normalized_sub_class = normalized_sub_class.replace("&", " and ")
    normalized_sub_class = re.sub(r"[\W_]+", " ", normalized_sub_class)
    return re.sub(r"\s+", " ", normalized_sub_class).strip()


def _build_chart_sub_class_lookup(groups: dict[str, tuple[str, ...]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for mapped_sub_class, variants in groups.items():
        for variant in variants:
            canonical_variant = _canonicalize_chart_sub_class(variant)
            existing_sub_class = lookup.get(canonical_variant)
            if existing_sub_class is not None and existing_sub_class != mapped_sub_class:
                raise ValueError(
                    f"Duplicate canonical chart sub_class variant '{canonical_variant}' "
                    f"for '{existing_sub_class}' and '{mapped_sub_class}'"
                )
            lookup[canonical_variant] = mapped_sub_class
    return lookup


CANONICAL_CHART_SUB_CLASS_LOOKUP = _build_chart_sub_class_lookup(CHART_SUB_CLASS_GROUPS)


def _keyword_match_chart_sub_class(normalized: str) -> str | None:
    """关键词匹配兜底：处理 CHART_SUB_CLASS_GROUPS 未能精确匹配的情况。

    参考 mineru_vl_2_5._parse_chart_type 的实现思路，按优先级从高到低匹配。
    normalized 为经过 _canonicalize_chart_sub_class 处理后的小写字符串。
    """
    # 组合/复合图优先（避免被单关键词规则误匹配）
    if "waterfall" in normalized:
        return "waterfall"
    if "candlestick" in normalized or "candle" in normalized or "k line" in normalized or "ohlc" in normalized:
        return "candlestick"
    if "donut" in normalized or "doughnut" in normalized:
        return "donut"
    if "sankey" in normalized:
        return "sankey"
    if "sunburst" in normalized:
        return "sunburst"
    if "treemap" in normalized or "tree map" in normalized:
        return "treemap"
    if "funnel" in normalized:
        return "funnel"
    if "word cloud" in normalized or "tag cloud" in normalized:
        return "word_cloud"
    if "gauge" in normalized or "speedometer" in normalized:
        return "gauge"
    if "violin" in normalized:
        return "violin"
    if "radar" in normalized or "spider" in normalized:
        return "radar"
    if "heatmap" in normalized or "heat map" in normalized:
        return "heatmap"
    if "bubble" in normalized:
        return "bubble"
    # 柱线混合（bar+line 必须在单独的 bar/line 之前）
    if "line" in normalized and "bar" in normalized:
        return "bar_line"
    if "line" in normalized:
        return "line"
    # 堆叠柱（stacked+bar/column 必须在单独的 bar 之前）
    if ("stacked" in normalized or "stack" in normalized) and ("bar" in normalized or "column" in normalized):
        return "bar_stacked"
    if ("stacked" in normalized or "stack" in normalized) and "area" in normalized:
        return "area_stacked"
    if "bar" in normalized or "column" in normalized:
        return "bar"
    if "area" in normalized:
        return "area"
    if "scatter" in normalized:
        return "scatter"
    if "histogram" in normalized:
        return "histogram"
    if "pie" in normalized:
        return "pie"
    if ("box" in normalized and ("plot" in normalized or "whisker" in normalized)) or "boxplot" in normalized:
        return "boxplot"
    if "map" in normalized or "geo" in normalized or "choropleth" in normalized or "world map" in normalized:
        return "geo"
    if "tree" in normalized or "hierarchy" in normalized or "dendrogram" in normalized:
        return "tree"
    if "network" in normalized:
        return "network"
    if "3d surface" in normalized or "surface plot" in normalized:
        return "surface_3d"
    if "3d scatter" in normalized:
        return "scatter_3d"
    if "polar bar" in normalized or "radial bar" in normalized or "rose" in normalized or "wind rose" in normalized:
        return "polar_bar"
    if "polar" in normalized:
        return "polar"
    if "contour" in normalized:
        return "contour"
    if "confusion" in normalized:
        return "confusion"
    if "roc" in normalized or "receiver operating" in normalized:
        return "roc"
    if "forest" in normalized and "plot" in normalized:
        return "forest"
    if "manhattan" in normalized:
        return "manhattan"
    if "volcano" in normalized:
        return "volcano"
    if "survival" in normalized or "kaplan" in normalized:
        return "survival"
    if "spectrogram" in normalized or "spectrum" in normalized:
        return "spectrogram"
    if "scatter matrix" in normalized or "pair plot" in normalized or "pairplot" in normalized:
        return "pairplot"
    if "hexbin" in normalized or "hex bin" in normalized:
        return "hexbin"
    if "qq" in normalized or "q q" in normalized or "quantile" in normalized:
        return "qq"
    return None


def _extract_tagged_field(text: str, start_tag: str, end_tag: str) -> str:
    start_idx = text.find(start_tag)
    if start_idx < 0:
        return ""
    start_idx += len(start_tag)
    end_idx = text.find(end_tag, start_idx)
    if end_idx < 0:
        return ""
    return text[start_idx:end_idx].strip()


def _count_markdown_table_columns(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return len(stripped.split("|"))


def _is_markdown_table_separator_line(line: str) -> bool:
    # Example: | --- | :---: | ---: |
    return re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", line) is not None


def _is_markdown_table_row_candidate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return "|" in stripped and _count_markdown_table_columns(stripped) >= 2


def _split_markdown_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False

    for char in stripped:
        if escaped:
            if char == "|":
                current.append("|")
            else:
                current.append("\\")
                current.append(char)
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def convert_markdown_table_to_html(content: str) -> str | None:
    if not content or not content.strip():
        return None

    lines = [line.strip() for line in content.strip().splitlines()]
    if len(lines) < 2 or any(not line for line in lines):
        return None

    header_line = lines[0]
    separator_line = lines[1]
    body_lines = lines[2:]

    if not _is_markdown_table_row_candidate(header_line) or not _is_markdown_table_separator_line(separator_line):
        return None

    header_cells = _split_markdown_table_row(header_line)
    separator_cells = _split_markdown_table_row(separator_line)
    if len(header_cells) < 2 or len(separator_cells) != len(header_cells):
        return None

    body_cells_list: list[list[str]] = []
    for body_line in body_lines:
        if not _is_markdown_table_row_candidate(body_line):
            return None
        row_cells = _split_markdown_table_row(body_line)
        if len(row_cells) != len(header_cells):
            return None
        body_cells_list.append(row_cells)

    header_html = "".join(f"<th>{html.escape(cell)}</th>" for cell in header_cells)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row_cells) + "</tr>" for row_cells in body_cells_list
    )
    return f"<table><tr>{header_html}</tr>{body_html}</table>"


def has_malformed_markdown_table(content: str) -> bool:
    lines = content.splitlines()
    separator_indices = [i for i, line in enumerate(lines) if _is_markdown_table_separator_line(line)]
    row_candidate_indices = [i for i, line in enumerate(lines) if _is_markdown_table_row_candidate(line)]
    found_valid_table = False

    for sep_idx in separator_indices:
        # 找 separator 前最近一行非空作为表头
        header_idx = sep_idx - 1
        while header_idx >= 0 and not lines[header_idx].strip():
            header_idx -= 1
        if header_idx < 0 or not _is_markdown_table_row_candidate(lines[header_idx]):
            return True

        header_cols = _count_markdown_table_columns(lines[header_idx])
        sep_cols = _count_markdown_table_columns(lines[sep_idx])
        if header_cols < 2 or sep_cols != header_cols:
            return True

        # 检查数据行列数是否一致（直到遇到空行或非表格行）
        row_idx = sep_idx + 1
        while row_idx < len(lines):
            row_line = lines[row_idx]
            if not row_line.strip():
                break
            if not _is_markdown_table_row_candidate(row_line):
                break
            if _count_markdown_table_columns(row_line) != header_cols:
                return True
            row_idx += 1
        found_valid_table = True

    # 有明显表格行但没有合法表格结构（常见于缺失/损坏 separator）
    if len(row_candidate_indices) >= 2 and not found_valid_table:
        return True

    return False


def extract_and_validate_mermaid_strict(content: str) -> str:
    """
    严格提取并校验 Mermaid flowchart 代码，修复常见的格式和语法问题。
    支持修复：
    1. 不规范的 Markdown 代码块
    2. 声明头部拼写错误 (如 grap -> graph)
    3. 错误的连线箭头 (如 ->, - -> 修正为 -->)
    4. 节点 ID 中的空格 (替换为下划线)
    5. 节点文本中的嵌套双引号 (转义为 &quot;) 和换行符问题
    """
    if not content or not content.strip():
        return ""

    content = content.strip()

    # ==========================================
    # 步骤 1: 提取核心的 mermaid 代码
    # ==========================================
    mermaid_match = re.search(r"```mermaid\s*(.*?)\s*```", content, re.DOTALL)
    if mermaid_match:
        code = mermaid_match.group(1).strip()
    else:
        code_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = content.strip()

    if code.lower().startswith("mermaid"):
        code = code[7:].strip()

    # ==========================================
    # 步骤 2: 修复图表声明拼写错误
    # ==========================================
    # 修复常见的大小写或拼写错误
    code = re.sub(r"^(grap|grapg|graphh)\b", "graph", code, flags=re.IGNORECASE)
    code = re.sub(r"^(flowchar|flowchartt)\b", "flowchart", code, flags=re.IGNORECASE)

    # ==========================================
    # 步骤 3: 修复不规范的连线箭头
    # ==========================================
    # 修复带有意外空格的箭头: - ->, -- >, -  -> 等，统一替换为 -->
    code = re.sub(r"-\s+->", "-->", code)
    code = re.sub(r"--\s+>", "-->", code)
    code = re.sub(r"-\s+-\s+>", "-->", code)
    # 修复漏写减号的箭头 -> (使用负向后瞻 (?<![-=]) 确保前面不是 - 或 =，防止误伤正常箭头)
    code = re.sub(r"(?<![-=])\s+->\s+", " --> ", code)

    # ==========================================
    # 步骤 4: 修复节点 ID 和 嵌套双引号
    # ==========================================

    def node_fixer(match):
        raw_node_id = match.group(1).strip()
        # 1. 修复节点 ID 里的空格：仅将水平空格/制表符替换为下划线，不影响换行符
        node_id = re.sub(r"[ \t]+", "_", raw_node_id)

        raw_text = match.group(2)

        # 2. 如果文本为空，直接返回
        if not raw_text:
            return f"{node_id}[]"

        # 3. 剥离原有的外层双引号
        if raw_text.startswith('"') and raw_text.endswith('"'):
            raw_text = raw_text[1:-1]

        # 4. 转义文本内部残留的双引号！
        safe_text = raw_text.replace('"', "&quot;")

        # 5. [新增] 将物理换行符替换为 Mermaid 友好的 <br> 标签，防止渲染崩溃
        safe_text = safe_text.replace("\n", "<br>")

        # 6. 重新用标准的双引号包裹
        return f'{node_id}["{safe_text}"]'

    # 正则解析修改：
    # 将原来的 \s 替换为 [ \t]，严格限制节点 ID 不能跨行！
    processed_code = re.sub(
        r"([a-zA-Z0-9_\u4e00-\u9fa5\-]+(?:[ \t]+[a-zA-Z0-9_\u4e00-\u9fa5\-]+)*)[ \t]*\[(.*?)\]",
        node_fixer,
        code,
        flags=re.DOTALL,
    )

    # ==========================================
    # 步骤 5: 返回标准格式
    # ==========================================
    return f"```mermaid\n{processed_code}\n```"


def _normalize_chart_sub_class(sub_class: str) -> str:
    normalized_sub_class = _canonicalize_chart_sub_class(sub_class)
    mapped_sub_class = CANONICAL_CHART_SUB_CLASS_LOOKUP.get(normalized_sub_class)
    if mapped_sub_class is not None:
        return mapped_sub_class

    # 精确匹配失败，尝试关键词匹配兜底
    keyword_matched = _keyword_match_chart_sub_class(normalized_sub_class)
    if keyword_matched is not None:
        logger.debug("Unknown chart sub_class: {}; keyword-matched to {}", sub_class, keyword_matched)
        return keyword_matched

    logger.warning("Unknown chart sub_class: {}; mapped to other", sub_class)
    return "other"


def process_image_or_chart(content: str) -> dict[str, str]:
    values = {field: _extract_tagged_field(content, tags[0], tags[1]) for field, tags in IMAGE_CHART_FIELD_TAGS.items()}

    class_name = values["class"].strip().lower()
    values["class"] = class_name
    normalized_content = values["content"]

    # 1) chemical 类别：content 置空
    if class_name == "chemical":
        normalized_content = ""
    # 2) flowchart 类别：严格校验并提取 mermaid
    elif class_name == "flowchart":
        normalized_content = extract_and_validate_mermaid_strict(normalized_content)
    # 3) markdown 表格语法有误或不闭合：content 置空
    elif class_name == "chart":
        values["sub_class"] = _normalize_chart_sub_class(values["sub_class"])
        if normalized_content and has_malformed_markdown_table(normalized_content):
            normalized_content = ""

    values["content"] = normalized_content

    return values


if __name__ == "__main__":
    content = """
<|class_start|>flowchart<|class_end|>
<|sub_class_start|>flowchart<|sub_class_end|>
<|caption_start|>Formula处理流程图，展示formula输入经formula valid与Formula Refiner（自带Unify效果）处理并输出公式后处理的过程<|caption_end|>
<|content_start|>

graph TD
    A[formula] --> B[formula valid]
    A --> C[formula valid]
    B --> D[Formula Unifier]
    C --> E[Formula Refiner (自带Unify效果)]
    D --> F[公式后处理]
    E --> F

<|content_end|>

    """
    logger.info(process_image_or_chart(content))
