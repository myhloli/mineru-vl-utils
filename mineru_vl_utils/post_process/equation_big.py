import re


def try_fix_equation_big(latex: str, debug: bool = False) -> str:

    # ------------------ \big{)} -> \big) ------------------ #
    original_latex = latex

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

    # ------------------ \big{\rangle} -> \big\rangle ------------------ #

    # \big
    latex = re.sub(r"\\big\{\\rangle\}", r"\\big\\rangle ", latex)
    latex = re.sub(r"\\big\{\\langle\}", r"\\big\\langle ", latex)
    latex = re.sub(r"\\big \{\\rangle\}", r"\\big\\rangle ", latex)
    latex = re.sub(r"\\big \{\\langle\}", r"\\big\\langle ", latex)

    # \bigr
    latex = re.sub(r"\\bigr\{\\rangle\}", r"\\bigr\\rangle ", latex)
    latex = re.sub(r"\\bigr\{\\langle\}", r"\\bigr\\langle ", latex)
    latex = re.sub(r"\\bigr \{\\rangle\}", r"\\bigr\\rangle ", latex)
    latex = re.sub(r"\\bigr \{\\langle\}", r"\\bigr\\langle ", latex)

    # \bigm
    latex = re.sub(r"\\bigm\{\\rangle\}", r"\\bigm\\rangle ", latex)
    latex = re.sub(r"\\bigm\{\\langle\}", r"\\bigm\\langle ", latex)
    latex = re.sub(r"\\bigm \{\\rangle\}", r"\\bigm\\rangle ", latex)
    latex = re.sub(r"\\bigm \{\\langle\}", r"\\bigm\\langle ", latex)

    # \bigl
    latex = re.sub(r"\\bigl\{\\rangle\}", r"\\bigl\\rangle ", latex)
    latex = re.sub(r"\\bigl\{\\langle\}", r"\\bigl\\langle ", latex)
    latex = re.sub(r"\\bigl \{\\rangle\}", r"\\bigl\\rangle ", latex)
    latex = re.sub(r"\\bigl \{\\langle\}", r"\\bigl\\langle ", latex)

    # \bigg
    latex = re.sub(r"\\bigg\{\\rangle\}", r"\\bigg\\rangle ", latex)
    latex = re.sub(r"\\bigg\{\\langle\}", r"\\bigg\\langle ", latex)
    latex = re.sub(r"\\bigg \{\\rangle\}", r"\\bigg\\rangle ", latex)
    latex = re.sub(r"\\bigg \{\\langle\}", r"\\bigg\\langle ", latex)

    # \biggr
    latex = re.sub(r"\\biggr\{\\rangle\}", r"\\biggr\\rangle ", latex)
    latex = re.sub(r"\\biggr\{\\langle\}", r"\\biggr\\langle ", latex)
    latex = re.sub(r"\\biggr \{\\rangle\}", r"\\biggr\\rangle ", latex)
    latex = re.sub(r"\\biggr \{\\langle\}", r"\\biggr\\langle ", latex)

    # \biggm
    latex = re.sub(r"\\biggm\{\\rangle\}", r"\\biggm\\rangle ", latex)
    latex = re.sub(r"\\biggm\{\\langle\}", r"\\biggm\\langle ", latex)
    latex = re.sub(r"\\biggm \{\\rangle\}", r"\\biggm\\rangle ", latex)
    latex = re.sub(r"\\biggm \{\\langle\}", r"\\biggm\\langle ", latex)

    # \biggl
    latex = re.sub(r"\\biggl\{\\rangle\}", r"\\biggl\\rangle ", latex)
    latex = re.sub(r"\\biggl\{\\langle\}", r"\\biggl\\langle ", latex)
    latex = re.sub(r"\\biggl \{\\rangle\}", r"\\biggl\\rangle ", latex)
    latex = re.sub(r"\\biggl \{\\langle\}", r"\\biggl\\langle ", latex)

    # \Big
    latex = re.sub(r"\\Big\{\\rangle\}", r"\\Big\\rangle ", latex)
    latex = re.sub(r"\\Big\{\\langle\}", r"\\Big\\langle ", latex)
    latex = re.sub(r"\\Big \{\\rangle\}", r"\\Big\\rangle ", latex)
    latex = re.sub(r"\\Big \{\\langle\}", r"\\Big\\langle ", latex)

    # \Bigr
    latex = re.sub(r"\\Bigr\{\\rangle\}", r"\\Bigr\\rangle ", latex)
    latex = re.sub(r"\\Bigr\{\\langle\}", r"\\Bigr\\langle ", latex)
    latex = re.sub(r"\\Bigr \{\\rangle\}", r"\\Bigr\\rangle ", latex)
    latex = re.sub(r"\\Bigr \{\\langle\}", r"\\Bigr\\langle ", latex)

    # \Bigm
    latex = re.sub(r"\\Bigm\{\\rangle\}", r"\\Bigm\\rangle ", latex)
    latex = re.sub(r"\\Bigm\{\\langle\}", r"\\Bigm\\langle ", latex)
    latex = re.sub(r"\\Bigm \{\\rangle\}", r"\\Bigm\\rangle ", latex)
    latex = re.sub(r"\\Bigm \{\\langle\}", r"\\Bigm\\langle ", latex)

    # \Bigl
    latex = re.sub(r"\\Bigl\{\\rangle\}", r"\\Bigl\\rangle ", latex)
    latex = re.sub(r"\\Bigl\{\\langle\}", r"\\Bigl\\langle ", latex)
    latex = re.sub(r"\\Bigl \{\\rangle\}", r"\\Bigl\\rangle ", latex)
    latex = re.sub(r"\\Bigl \{\\langle\}", r"\\Bigl\\langle ", latex)

    # \Bigg
    latex = re.sub(r"\\Bigg\{\\rangle\}", r"\\Bigg\\rangle ", latex)
    latex = re.sub(r"\\Bigg\{\\langle\}", r"\\Bigg\\langle ", latex)
    latex = re.sub(r"\\Bigg \{\\rangle\}", r"\\Bigg\\rangle ", latex)
    latex = re.sub(r"\\Bigg \{\\langle\}", r"\\Bigg\\langle ", latex)

    # \Biggr
    latex = re.sub(r"\\Biggr\{\\rangle\}", r"\\Biggr\\rangle ", latex)
    latex = re.sub(r"\\Biggr\{\\langle\}", r"\\Biggr\\langle ", latex)
    latex = re.sub(r"\\Biggr \{\\rangle\}", r"\\Biggr\\rangle ", latex)
    latex = re.sub(r"\\Biggr \{\\langle\}", r"\\Biggr\\langle ", latex)

    # \Biggl
    latex = re.sub(r"\\Biggl\{\\rangle\}", r"\\Biggl\\rangle ", latex)
    latex = re.sub(r"\\Biggl\{\\langle\}", r"\\Biggl\\langle ", latex)
    latex = re.sub(r"\\Biggl \{\\rangle\}", r"\\Biggl\\rangle ", latex)
    latex = re.sub(r"\\Biggl \{\\langle\}", r"\\Biggl\\langle ", latex)

    if debug and original_latex != latex:
        print(f"Fixed equation big from: {original_latex} to: {latex}")

    return latex


if __name__ == "__main__":
    latex = r"\begin{array}{l} \widehat {J} (\zeta ; u) \\ = \frac{1}{2}\mathbb{E}\left[\int_{0}^{T}\Bigg{\langle}\left( \begin{array}{ccc}0 & S_{1} + \Phi C & S_{2} + \Phi B\\ (S_{1} + \Phi C)^{\top} & R_{11} + \Phi & R_{12}\\ (S_{2} + \Phi B)^{\top} & R_{12}^{\top} & R_{22} \end{array} \right)\left( \begin{array}{c}Y - \mathbb{E}[Y]\\ Z - \mathbb{E}[Z]\\ u - \mathbb{E}[u] \end{array} \right),\left( \begin{array}{c}Y - \mathbb{E}[Y]\\ Z - \mathbb{E}[Z]\\ u - \mathbb{E}[u] \end{array} \right)\Bigg{\rangle}dt\right. \\ + \int_ {0} ^ {T} \left\langle \left( \begin{array}{c c c} 0 & \widetilde {S} _ {1} + \widetilde {\Phi} \widetilde {C} & \widetilde {S} _ {2} + \widetilde {\Phi} \widetilde {B} \\ (\widetilde {S} _ {1} + \widetilde {\Phi} \widetilde {C}) ^ {\top} & \widetilde {R} _ {1 1} + \Phi & \widetilde {R} _ {1 2} \\ (\widetilde {S} _ {2} + \widetilde {\Phi} \widetilde {B}) ^ {\top} & \widetilde {R} _ {1 2} ^ {\top} & \widetilde {R} _ {2 2} \end{array} \right) \left( \begin{array}{c} \mathbb {E} [ Y ] \\ \mathbb {E} [ Z ] \\ \mathbb {E} [ u ] \end{array} \right), \left( \begin{array}{c} \mathbb {E} [ Y ] \\ \mathbb {E} [ Z ] \\ \mathbb {E} [ u ] \end{array} \right) \right\rangle d t \\ \left. + 2 \int_ {0} ^ {T} \left\langle \left( \begin{array}{c} \widetilde {q} \\ \rho_ {1} \\ \rho_ {2} \end{array} \right), \left( \begin{array}{c} Y \\ Z \\ u \end{array} \right) \right\rangle d t + 2 \langle g, Y (0) \rangle \right]. \\ \end{array}"
    print(try_fix_equation_big(latex))
