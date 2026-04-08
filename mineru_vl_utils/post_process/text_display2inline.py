import re

def try_convert_display_to_inline(text: str, debug: bool = False) -> str:
    
    new_text = re.sub(r'\\\[(.+?)\\\]', r'\\(\1\\)', text, flags=re.DOTALL)
    
    if debug and new_text != text:
        print(f"Fixed equation delimeters from: {text} to: {new_text}")
    
    return new_text

if __name__ == "__main__":
    text = r"(B) \[ \begin{pmatrix}2&-1\\-1&2\end{pmatrix}. \]"
    print(try_convert_display_to_inline(text))