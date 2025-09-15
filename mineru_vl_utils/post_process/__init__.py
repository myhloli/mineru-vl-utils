from ..structs import ContentBlock
from .equation_block import do_handle_equation_block
from .equation_left_right import try_match_equation_left_right
from .otsl2html import convert_otsl_to_html

PARATEXT_TYPES = {
    "header",
    "footer",
    "page_number",
    "aside_text",
    "page_footnote",
    "unknown",
}


def post_process(
    blocks: list[ContentBlock],
    handle_equation_block: bool,
    abandon_list: bool,
    abandon_paratext: bool,
    debug: bool = False,
) -> list[ContentBlock]:
    for block in blocks:
        if block.type == "table" and block.content:
            block.content = convert_otsl_to_html(block.content)
        if block.type == "equation" and block.content:
            block.content = try_match_equation_left_right(block.content, debug=debug)

    if handle_equation_block:
        blocks = do_handle_equation_block(blocks, debug=debug)

    out_blocks: list[ContentBlock] = []
    for block in blocks:
        if block.type == "equation_block":  # drop equation_block anyway
            continue
        if abandon_list and block.type == "list":
            continue
        if abandon_paratext and block.type in PARATEXT_TYPES:
            continue
        out_blocks.append(block)

    return out_blocks
