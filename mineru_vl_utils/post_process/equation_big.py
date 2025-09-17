import re


def try_fix_equation_big(latex: str, debug: bool = False) -> str:
    
    # ------------------ \big{)} -> \big) ------------------ #
    
    # \big
    latex = re.sub(r"\\big{\)}", r"\\big)", latex)
    latex = re.sub(r"\\big{\(}", r"\\big(", latex)
    latex = re.sub(r"\\big {\)}", r"\\big)", latex)
    latex = re.sub(r"\\big {\(}", r"\\big(", latex)
    
    # \bigr
    latex = re.sub(r"\\bigr{\)}", r"\\bigr)", latex)
    latex = re.sub(r"\\bigr{\(}", r"\\bigr(", latex)
    latex = re.sub(r"\\bigr {\)}", r"\\bigr)", latex)
    latex = re.sub(r"\\bigr {\(}", r"\\bigr(", latex)
    
    # \bigm
    latex = re.sub(r"\\bigm{\)}", r"\\bigm)", latex)
    latex = re.sub(r"\\bigm{\(}", r"\\bigm(", latex)
    latex = re.sub(r"\\bigm {\)}", r"\\bigm)", latex)
    latex = re.sub(r"\\bigm {\(}", r"\\bigm(", latex)
    
    # \bigl
    latex = re.sub(r"\\bigl{\)}", r"\\bigl)", latex)
    latex = re.sub(r"\\bigl{\(}", r"\\bigl(", latex)
    latex = re.sub(r"\\bigl {\)}", r"\\bigl)", latex)
    latex = re.sub(r"\\bigl {\(}", r"\\bigl(", latex)
    
    # \bigg
    latex = re.sub(r"\\bigg{\)}", r"\\bigg)", latex)
    latex = re.sub(r"\\bigg{\(}", r"\\bigg(", latex)
    latex = re.sub(r"\\bigg {\)}", r"\\bigg)", latex)
    latex = re.sub(r"\\bigg {\(}", r"\\bigg(", latex)
    
    # \biggr
    latex = re.sub(r"\\biggr{\)}", r"\\biggr)", latex)
    latex = re.sub(r"\\biggr{\(}", r"\\biggr(", latex)
    latex = re.sub(r"\\biggr {\)}", r"\\biggr)", latex)
    latex = re.sub(r"\\biggr {\(}", r"\\biggr(", latex)
    
    # \biggm
    latex = re.sub(r"\\biggm{\)}", r"\\biggm)", latex)
    latex = re.sub(r"\\biggm{\(}", r"\\biggm(", latex)
    latex = re.sub(r"\\biggm {\)}", r"\\biggm)", latex)
    latex = re.sub(r"\\biggm {\(}", r"\\biggm(", latex)
    
    # \biggl
    latex = re.sub(r"\\biggl{\)}", r"\\biggl)", latex)
    latex = re.sub(r"\\biggl{\(}", r"\\biggl(", latex)
    latex = re.sub(r"\\biggl {\)}", r"\\biggl)", latex)
    latex = re.sub(r"\\biggl {\(}", r"\\biggl(", latex)
    
    # \Big
    latex = re.sub(r"\\Big{\)}", r"\\Big)", latex)
    latex = re.sub(r"\\Big{\(}", r"\\Big(", latex)
    latex = re.sub(r"\\Big {\)}", r"\\Big)", latex)
    latex = re.sub(r"\\Big {\(}", r"\\Big(", latex)
    
    # \Bigr
    latex = re.sub(r"\\Bigr{\)}", r"\\Bigr)", latex)
    latex = re.sub(r"\\Bigr{\(}", r"\\Bigr(", latex)
    latex = re.sub(r"\\Bigr {\)}", r"\\Bigr)", latex)
    latex = re.sub(r"\\Bigr {\(}", r"\\Bigr(", latex)
    
    # \Bigm
    latex = re.sub(r"\\Bigm{\)}", r"\\Bigm)", latex)
    latex = re.sub(r"\\Bigm{\(}", r"\\Bigm(", latex)
    latex = re.sub(r"\\Bigm {\)}", r"\\Bigm)", latex)
    latex = re.sub(r"\\Bigm {\(}", r"\\Bigm(", latex)
    
    # \Bigl
    latex = re.sub(r"\\Bigl{\)}", r"\\Bigl)", latex)
    latex = re.sub(r"\\Bigl{\(}", r"\\Bigl(", latex)
    latex = re.sub(r"\\Bigl {\)}", r"\\Bigl)", latex)
    latex = re.sub(r"\\Bigl {\(}", r"\\Bigl(", latex)
    
    # \Bigg
    latex = re.sub(r"\\Bigg{\)}", r"\\Bigg)", latex)
    latex = re.sub(r"\\Bigg{\(}", r"\\Bigg(", latex)
    latex = re.sub(r"\\Bigg {\)}", r"\\Bigg)", latex)
    latex = re.sub(r"\\Bigg {\(}", r"\\Bigg(", latex)
    
    # \Biggr
    latex = re.sub(r"\\Biggr{\)}", r"\\Biggr)", latex)
    latex = re.sub(r"\\Biggr{\(}", r"\\Biggr(", latex)
    latex = re.sub(r"\\Biggr {\)}", r"\\Biggr)", latex)
    latex = re.sub(r"\\Biggr {\(}", r"\\Biggr(", latex)
    
    # \Biggm
    latex = re.sub(r"\\Biggm{\)}", r"\\Biggm)", latex)
    latex = re.sub(r"\\Biggm{\(}", r"\\Biggm(", latex)
    latex = re.sub(r"\\Biggm {\)}", r"\\Biggm)", latex)
    latex = re.sub(r"\\Biggm {\(}", r"\\Biggm(", latex)
    
    # \Biggl
    latex = re.sub(r"\\Biggl{\)}", r"\\Biggl)", latex)
    latex = re.sub(r"\\Biggl{\(}", r"\\Biggl(", latex)
    latex = re.sub(r"\\Biggl {\)}", r"\\Biggl)", latex)
    latex = re.sub(r"\\Biggl {\(}", r"\\Biggl(", latex)
    
    # ------------------ \big{\}} -> \big\} ------------------ #
    
    # \big
    latex = re.sub(r"\\big\{\\\}\}", r"\\big\\}", latex)
    latex = re.sub(r"\\big\{\\\{\}", r"\\big\\{", latex)
    latex = re.sub(r"\\big \{\\\}\}", r"\\big\\}", latex)
    latex = re.sub(r"\\big \{\\\{\}", r"\\big\\{", latex)
    
    # \bigr
    latex = re.sub(r"\\bigr\{\\\}\}", r"\\bigr\\}", latex)
    latex = re.sub(r"\\bigr\{\\\{\}", r"\\bigr\\{", latex)
    latex = re.sub(r"\\bigr \{\\\}\}", r"\\bigr\\}", latex)
    latex = re.sub(r"\\bigr \{\\\{\}", r"\\bigr\\{", latex)
    
    # \bigm
    latex = re.sub(r"\\bigm\{\\\}\}", r"\\bigm\\}", latex)
    latex = re.sub(r"\\bigm\{\\\{\}", r"\\bigm\\{", latex)
    latex = re.sub(r"\\bigm \{\\\}\}", r"\\bigm\\}", latex)
    latex = re.sub(r"\\bigm \{\\\{\}", r"\\bigm\\{", latex)
    
    # \bigl
    latex = re.sub(r"\\bigl\{\\\}\}", r"\\bigl\\}", latex)
    latex = re.sub(r"\\bigl\{\\\{\}", r"\\bigl\\{", latex)
    latex = re.sub(r"\\bigl \{\\\}\}", r"\\bigl\\}", latex)
    latex = re.sub(r"\\bigl \{\\\{\}", r"\\bigl\\{", latex)
    
    # \bigg
    latex = re.sub(r"\\bigg\{\\\}\}", r"\\bigg\\}", latex)
    latex = re.sub(r"\\bigg\{\\\{\}", r"\\bigg\\{", latex)
    latex = re.sub(r"\\bigg \{\\\}\}", r"\\bigg\\}", latex)
    latex = re.sub(r"\\bigg \{\\\{\}", r"\\bigg\\{", latex)
    
    # \biggr
    latex = re.sub(r"\\biggr\{\\\}\}", r"\\biggr\\}", latex)
    latex = re.sub(r"\\biggr\{\\\{\}", r"\\biggr\\{", latex)
    latex = re.sub(r"\\biggr \{\\\}\}", r"\\biggr\\}", latex)
    latex = re.sub(r"\\biggr \{\\\{\}", r"\\biggr\\{", latex)
    
    # \biggm
    latex = re.sub(r"\\biggm\{\\\}\}", r"\\biggm\\}", latex)
    latex = re.sub(r"\\biggm\{\\\{\}", r"\\biggm\\{", latex)
    latex = re.sub(r"\\biggm \{\\\}\}", r"\\biggm\\}", latex)
    latex = re.sub(r"\\biggm \{\\\{\}", r"\\biggm\\{", latex)
    
    # \biggl
    latex = re.sub(r"\\biggl\{\\\}\}", r"\\biggl\\}", latex)
    latex = re.sub(r"\\biggl\{\\\{\}", r"\\biggl\\{", latex)
    latex = re.sub(r"\\biggl \{\\\}\}", r"\\biggl\\}", latex)
    latex = re.sub(r"\\biggl \{\\\{\}", r"\\biggl\\{", latex)
    
    # \Big
    latex = re.sub(r"\\Big\{\\\}\}", r"\\Big\\}", latex)
    latex = re.sub(r"\\Big\{\\\{\}", r"\\Big\\{", latex)
    latex = re.sub(r"\\Big \{\\\}\}", r"\\Big\\}", latex)
    latex = re.sub(r"\\Big \{\\\{\}", r"\\Big\\{", latex)
    
    # \Bigr
    latex = re.sub(r"\\Bigr\{\\\}\}", r"\\Bigr\\}", latex)
    latex = re.sub(r"\\Bigr\{\\\{\}", r"\\Bigr\\{", latex)
    latex = re.sub(r"\\Bigr \{\\\}\}", r"\\Bigr\\}", latex)
    latex = re.sub(r"\\Bigr \{\\\{\}", r"\\Bigr\\{", latex)
    
    # \Bigm
    latex = re.sub(r"\\Bigm\{\\\}\}", r"\\Bigm\\}", latex)
    latex = re.sub(r"\\Bigm\{\\\{\}", r"\\Bigm\\{", latex)
    latex = re.sub(r"\\Bigm \{\\\}\}", r"\\Bigm\\}", latex)
    latex = re.sub(r"\\Bigm \{\\\{\}", r"\\Bigm\\{", latex)
    
    # \Bigl
    latex = re.sub(r"\\Bigl\{\\\}\}", r"\\Bigl\\}", latex)
    latex = re.sub(r"\\Bigl\{\\\{\}", r"\\Bigl\\{", latex)
    latex = re.sub(r"\\Bigl \{\\\}\}", r"\\Bigl\\}", latex)
    latex = re.sub(r"\\Bigl \{\\\{\}", r"\\Bigl\\{", latex)
    
    # \Bigg
    latex = re.sub(r"\\Bigg\{\\\}\}", r"\\Bigg\\}", latex)
    latex = re.sub(r"\\Bigg\{\\\{\}", r"\\Bigg\\{", latex)
    latex = re.sub(r"\\Bigg \{\\\}\}", r"\\Bigg\\}", latex)
    latex = re.sub(r"\\Bigg \{\\\{\}", r"\\Bigg\\{", latex)
    
    # \Biggr
    latex = re.sub(r"\\Biggr\{\\\}\}", r"\\Biggr\\}", latex)
    latex = re.sub(r"\\Biggr\{\\\{\}", r"\\Biggr\\{", latex)
    latex = re.sub(r"\\Biggr \{\\\}\}", r"\\Biggr\\}", latex)
    latex = re.sub(r"\\Biggr \{\\\{\}", r"\\Biggr\\{", latex)
    
    # \Biggl
    latex = re.sub(r"\\Biggl\{\\\}\}", r"\\Biggl\\}", latex)
    latex = re.sub(r"\\Biggl\{\\\{\}", r"\\Biggl\\{", latex)
    latex = re.sub(r"\\Biggl \{\\\}\}", r"\\Biggl\\}", latex)
    latex = re.sub(r"\\Biggl \{\\\{\}", r"\\Biggl\\{", latex)
    
    # ------------------ \big{\|} -> \big\| ------------------ #
    
    # \big
    latex = re.sub(r"\\big{\|}", r"\\big|", latex)
    latex = re.sub(r"\\Big{\|}", r"\\Big|", latex)
    latex = re.sub(r"\\big {\|}", r"\\big|", latex)
    latex = re.sub(r"\\Big {\|}", r"\\Big|", latex)
    
    # \bigm
    latex = re.sub(r"\\bigm{\|}", r"\\bigm|", latex)
    latex = re.sub(r"\\Bigm{\|}", r"\\Bigm|", latex)
    latex = re.sub(r"\\bigm {\|}", r"\\bigm|", latex)
    latex = re.sub(r"\\Bigm {\|}", r"\\Bigm|", latex)
    
    # \bigr
    latex = re.sub(r"\\bigr{\|}", r"\\bigr|", latex)
    latex = re.sub(r"\\Bigr{\|}", r"\\Bigr|", latex)
    latex = re.sub(r"\\bigr {\|}", r"\\bigr|", latex)
    latex = re.sub(r"\\Bigr {\|}", r"\\Bigr|", latex)
    
    # \bigl
    latex = re.sub(r"\\bigl{\|}", r"\\bigl|", latex)
    latex = re.sub(r"\\Bigl{\|}", r"\\Bigl|", latex)
    latex = re.sub(r"\\bigl {\|}", r"\\bigl|", latex)
    latex = re.sub(r"\\Bigl {\|}", r"\\Bigl|", latex)
    
    # \bigg
    latex = re.sub(r"\\bigg{\|}", r"\\bigg|", latex)
    latex = re.sub(r"\\Bigg{\|}", r"\\Bigg|", latex)
    latex = re.sub(r"\\bigg {\|}", r"\\bigg|", latex)
    latex = re.sub(r"\\Bigg {\|}", r"\\Bigg|", latex)
    
    # \biggr
    latex = re.sub(r"\\biggr{\|}", r"\\biggr|", latex)
    latex = re.sub(r"\\Biggr{\|}", r"\\Biggr|", latex)
    latex = re.sub(r"\\biggr {\|}", r"\\biggr|", latex)
    latex = re.sub(r"\\Biggr {\|}", r"\\Biggr|", latex)
    
    # \biggm
    latex = re.sub(r"\\biggm\{\\\|\}", r"\\biggm\|", latex)
    latex = re.sub(r"\\Biggm\{\\\|\}", r"\\Biggm\|", latex)
    latex = re.sub(r"\\biggm \{\\\|\}", r"\\biggm\|", latex)
    latex = re.sub(r"\\Biggm \{\\\|\}", r"\\Biggm\|", latex)
    
    # \biggl
    latex = re.sub(r"\\biggl\{\\\|\}", r"\\biggl\|", latex)
    latex = re.sub(r"\\Biggl\{\\\|\}", r"\\Biggl\|", latex)
    latex = re.sub(r"\\biggl \{\\\|\}", r"\\biggl\|", latex)
    latex = re.sub(r"\\Biggl \{\\\|\}", r"\\Biggl\|", latex)
    
    return latex


if __name__ == "__main__":
    latex = r"\begin{array}{l} \mathcal {P} _ {z _ {1}; \iota u} \omega_ {| I | + 3} ^ {(0)} \left(z _ {2}, z _ {1}, u, I\right) \tag {32} \\ = \operatorname *{Res}_{\tilde{q} = \iota u}\frac{dz_{1}}{z_{1} - \tilde{q}}\Bigl {\{}2\sum_{i}\operatorname *{Res}_{q = \beta_{i}}K_{i}(z_{2},q)\sum_{\substack{I^{\prime}\sqcup I^{\prime \prime} = I\\ I^{\prime \prime}\neq \emptyset}}\omega_{|I^{\prime}| + 3}^{(0)}(q,\tilde{q},u,I^{\prime})\omega_{|I^{\prime \prime}| + 1}^{(0)}(\sigma_{i}(q),I^{\prime \prime})\quad (\dagger) \\ \end{array}"
    print(try_fix_equation_big(latex))
