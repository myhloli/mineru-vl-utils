import re

def try_fix_equation_eqqcolon(latex: str, debug: bool = False) -> str:
    latex = re.sub(r"\\eqqcolon", "=:", latex)
    latex = re.sub(r"\\coloneqq", ":=", latex)
    return latex
    
if __name__ == "__main__":
    latex = r"a \coloneqq b"
    print(try_fix_equation_eqqcolon(latex))