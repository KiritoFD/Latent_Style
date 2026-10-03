"""Render the deck's equations and the TikZ architecture figure to PNG.

    python make_equations.py

Writes assets/eq_*.png (transparent), assets/architecture.png, and assets/equations.json with
each image's natural size in points (10 pt math), which make_deck.py uses to scale them.
Requires pdflatex, pdftocairo, and pdfinfo.
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
ARCH = HERE.parent / "icme2027" / "fig_arch.tex"

EQUATIONS = {
    "eq_sandwich": r"""\begin{aligned}
&S(y) > S(y_{\mathrm{IDT}})\quad \forall S\in\{\text{DINO-S},\ \text{CLIP-S}\}\\
&C(y) \succ C(y_{\mathrm{TGT}})\quad \forall C\in\{\text{LPIPS},\ \text{DINO-C}\}
\end{aligned}""",
    "eq_target": r"""z_1^\star=\mathcal W^{-1}\big((1-\alpha)\,\ell_c+\alpha\,\mathrm{AdaIN}(\ell_c,\ell_s),\ h_{1,s},\,h_{2,s},\,h_{3,s}\big)""",
    "eq_bound": r"""\|u_\ell\|^2=\alpha^2N\big[(\sigma_s-\sigma_c)^2+(\mu_s-\mu_c)^2\big]\ \le\ \alpha^2\,\|\ell_s-\ell_c\|^2""",
    "eq_loss": r"""\mathcal L=\lambda_{LL}\|v_\ell-u_\ell\|^2+\|v_{h_1}-u_{h_1}\|^2+\|v_{h_2}-u_{h_2}\|^2""",
    "eq_proof": (r"""\begin{aligned}
&\mathrm{AdaIN}(\ell_c,\ell_s)-\ell_c=\big(\tfrac{\sigma_s}{\sigma_c}-1\big)(\ell_c-\mu_c)+(\mu_s-\mu_c)\\
&\alpha^2\|\ell_s-\ell_c\|^2-\|u_\ell\|^2=2\alpha^2N\sigma_s\sigma_c\,(1-\rho)\ \ge\ 0
\end{aligned}""", "4B5563"),
    "eq_step": r"""\begin{aligned}
\bar z^{k+1}&=z^k+\tfrac1K\,\mathcal W^{-1}\big(v_\ell^k,\,v_{h_1}^k,\,v_{h_2}^k,\,0\big)\\
z^{k+1}&=P_L\bar z^{k+1}+(1-\beta)\,\bar q+\beta\,\mathcal A\big(\bar q;\,(1+\eta)\,q_s\big)
\end{aligned}""",
}

EQ_TEX = r"""\documentclass[border=1.5pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage[HTML]{xcolor}
\begin{document}
\color[HTML]{%s}$\displaystyle %s$
\end{document}
"""

# Sans-serif rendering of the paper's TikZ figure so it matches the slide typography.
ARCH_TEX = r"""\documentclass[border=4pt]{standalone}
\usepackage{amsmath,amssymb}
\usepackage[scaled=0.95]{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{sansmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,calc,fit,backgrounds}
\begin{document}
\sansmath
\input{fig_arch}
\end{document}
"""


def run(cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def page_size_pt(pdf):
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    w, h = re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out).groups()
    return float(w), float(h)


def main():
    ASSETS.mkdir(exist_ok=True)
    sizes = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, spec in EQUATIONS.items():
            body, color = spec if isinstance(spec, tuple) else (spec, "0E2841")
            (tmp / f"{name}.tex").write_text(EQ_TEX % (color, body), encoding="utf-8")
            run(["pdflatex", "-interaction=nonstopmode", f"{name}.tex"], tmp)
            sizes[name] = page_size_pt(tmp / f"{name}.pdf")
            run(["pdftocairo", "-png", "-transp", "-r", "600", "-singlefile", f"{name}.pdf", name], tmp)
            shutil.copy(tmp / f"{name}.png", ASSETS / f"{name}.png")
        shutil.copy(ARCH, tmp / "fig_arch.tex")
        (tmp / "architecture.tex").write_text(ARCH_TEX, encoding="utf-8")
        run(["pdflatex", "-interaction=nonstopmode", "architecture.tex"], tmp)
        sizes["architecture"] = page_size_pt(tmp / "architecture.pdf")
        run(["pdftocairo", "-png", "-r", "400", "-singlefile", "architecture.pdf", "architecture"], tmp)
        shutil.copy(tmp / "architecture.png", ASSETS / "architecture.png")
    (ASSETS / "equations.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")
    for k, (w, h) in sizes.items():
        print(f"{k}: {w:.1f} x {h:.1f} pt")


if __name__ == "__main__":
    main()
