import re

def try_fix_equation_double_subscript(latex: str, debug: bool = False) -> str:
    pattern = r'_\s*\{([^{}]|\{[^{}]*\})*\}\s*_\s*\{([^{}]|\{[^{}]*\})*\}'
    print(re.search(pattern, latex))
    if re.search(pattern, latex):
        latex = re.sub(pattern, "", latex)
    return latex
    