import re


def try_fix_equation_big(latex: str, debug: bool = False) -> str:
    
    # ------------------ \big{)} -> \big) ------------------ #
    
    # \big
    latex = re.sub(r"\\big{\)}", r"\\big)", latex)
    latex = re.sub(r"\\big{\(}", r"\\big(", latex)
    
    # \bigr
    latex = re.sub(r"\\bigr{\)}", r"\\bigr)", latex)
    latex = re.sub(r"\\bigr{\(}", r"\\bigr(", latex)
    
    # \bigm
    latex = re.sub(r"\\bigm{\)}", r"\\bigm)", latex)
    latex = re.sub(r"\\bigm{\(}", r"\\bigm(", latex)
    
    # \bigg
    latex = re.sub(r"\\bigg{\)}", r"\\bigg)", latex)
    latex = re.sub(r"\\bigg{\(}", r"\\bigg(", latex)
    
    # \biggr
    latex = re.sub(r"\\biggr{\)}", r"\\biggr)", latex)
    latex = re.sub(r"\\biggr{\(}", r"\\biggr(", latex)
    
    # \biggm
    latex = re.sub(r"\\biggm{\)}", r"\\biggm)", latex)
    latex = re.sub(r"\\biggm{\(}", r"\\biggm(", latex)
    
    # \Big
    latex = re.sub(r"\\Big{\)}", r"\\Big)", latex)
    latex = re.sub(r"\\Big{\(}", r"\\Big(", latex)
    
    # \Bigr
    latex = re.sub(r"\\Bigr{\)}", r"\\Bigr)", latex)
    latex = re.sub(r"\\Bigr{\(}", r"\\Bigr(", latex)
    
    # \Bigm
    latex = re.sub(r"\\Bigm{\)}", r"\\Bigr)", latex)
    latex = re.sub(r"\\Bigm{\(}", r"\\Bigr(", latex)
    
    # \Bigg
    latex = re.sub(r"\\Bigg{\)}", r"\\Bigg)", latex)
    latex = re.sub(r"\\Bigg{\(}", r"\\Bigg(", latex)
    
    # \Biggr
    latex = re.sub(r"\\Biggr{\)}", r"\\Biggr)", latex)
    latex = re.sub(r"\\Biggr{\(}", r"\\Biggr(", latex)
    
    # \Biggm
    latex = re.sub(r"\\Biggm{\)}", r"\\Biggm)", latex)
    latex = re.sub(r"\\Biggm{\(}", r"\\Biggm(", latex)
    
    # ------------------ \big{\}} -> \big\} ------------------ #
    
    # \big
    latex = re.sub(r"\\big\{\\\}\}", r"\\big\\}", latex)
    latex = re.sub(r"\\big\{\\\{\}", r"\\big\\{", latex)
    
    # \bigr
    latex = re.sub(r"\\bigr\{\\\}\}", r"\\bigr\\}", latex)
    latex = re.sub(r"\\bigr\{\\\{\}", r"\\bigr\\{", latex)
    
    # \bigm
    latex = re.sub(r"\\bigm\{\\\}\}", r"\\bigm\\}", latex)
    latex = re.sub(r"\\bigm\{\\\{\}", r"\\bigm\\{", latex)
    
    # \bigg
    latex = re.sub(r"\\bigg\{\\\}\}", r"\\bigg\\}", latex)
    latex = re.sub(r"\\bigg\{\\\{\}", r"\\bigg\\{", latex)
    
    # \biggr
    latex = re.sub(r"\\biggr\{\\\}\}", r"\\biggr\\}", latex)
    latex = re.sub(r"\\biggr\{\\\{\}", r"\\biggr\\{", latex)
    
    # \biggm
    latex = re.sub(r"\\biggm\{\\\}\}", r"\\biggm\\}", latex)
    latex = re.sub(r"\\biggm\{\\\{\}", r"\\biggm\\{", latex)
    
    # \Big
    latex = re.sub(r"\\Big\{\\\}\}", r"\\Big\\}", latex)
    latex = re.sub(r"\\Big\{\\\{\}", r"\\Big\\{", latex)
    
    # \Bigr
    latex = re.sub(r"\\Bigr\{\\\}\}", r"\\Bigr\\}", latex)
    latex = re.sub(r"\\Bigr\{\\\{\}", r"\\Bigr\\{", latex)
    
    # \Bigm
    latex = re.sub(r"\\Bigm\{\\\}\}", r"\\Bigm\\}", latex)
    latex = re.sub(r"\\Bigm\{\\\{\}", r"\\Bigm\\{", latex)
    
    # \Bigg
    latex = re.sub(r"\\Bigg\{\\\}\}", r"\\Bigg\\}", latex)
    latex = re.sub(r"\\Bigg\{\\\{\}", r"\\Bigg\\{", latex)
    
    # \Biggr
    latex = re.sub(r"\\Biggr\{\\\}\}", r"\\Biggr\\}", latex)
    latex = re.sub(r"\\Biggr\{\\\{\}", r"\\Biggr\\{", latex)
    
    # \Biggm
    latex = re.sub(r"\\Biggm\{\\\}\}", r"\\Biggm\\}", latex)
    latex = re.sub(r"\\Biggm\{\\\{\}", r"\\Biggm\\{", latex)
    
    # ------------------ \big{\|} -> \big\| ------------------ #
    
    # \big
    latex = re.sub(r"\\big{\|}", r"\\big|", latex)
    latex = re.sub(r"\\Big{\|}", r"\\Big|", latex)
    
    # \bigm
    latex = re.sub(r"\\bigm{\|}", r"\\bigm|", latex)
    latex = re.sub(r"\\Bigm{\|}", r"\\Bigm|", latex)
    
    # \bigr
    latex = re.sub(r"\\bigr{\|}", r"\\bigr|", latex)
    latex = re.sub(r"\\Bigr{\|}", r"\\Bigr|", latex)
    
    # \bigg
    latex = re.sub(r"\\bigg{\|}", r"\\bigg|", latex)
    latex = re.sub(r"\\Bigg{\|}", r"\\Bigg|", latex)
    
    # \biggr
    latex = re.sub(r"\\biggr{\|}", r"\\biggr|", latex)
    latex = re.sub(r"\\Biggr{\|}", r"\\Biggr|", latex)
    
    # \biggm
    latex = re.sub(r"\\biggm\{\\\|\}", r"\\biggm\|", latex)
    latex = re.sub(r"\\Biggm\{\\\|\}", r"\\Biggm\|", latex)
    
    return latex


if __name__ == "__main__":
    latex = r"a \bigg{|} b"
    print(try_fix_equation_big(latex))
