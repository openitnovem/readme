import base64
from io import BytesIO
from typing import List

import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.tools as tls

from fbd_interpreter.utils import read_sections_from_txt


def plotly_figures_to_html(
    dic_figs, html_sections, plot_type, path: str, title: str = ""
) -> None:
    """Convert a dict of plotly figures to html format.

    Args:
        dic_figs: Dict of plotly figures to be saved.
        path: Path to write html file
        title: Title
    Returns:
        string in HTML format
    """
    figs = list(dic_figs.values())
    titles = list(dic_figs.keys())
    dico_sections = read_sections_from_txt(html_sections)
    add_js = True

    html = """<html><head><meta charset="utf-8"/><style>
            img {
              display: block;
              margin-left: auto;
              margin-right: auto;
            }
            </style></head><body>\n"""
    html += f'<h1 style="color:MediumBlue;text-align:center;font-size:300%">{title}</h1>\n\n'
    html += f'<h1 style="color:Navy;font-size:160%">Description générale : </h1>\n\n'
    commun_section = dico_sections["COMMUN"]
    for el in commun_section:
        html += f'<p style="font-size:120%"> {el} </p>'
    html += f"<hr>\n\n"

    type_plot_section = dico_sections[plot_type]

    html += (
        f'<h1 style="color:Navy;font-size:160%">Description de {plot_type} : </h1>\n\n'
    )
    for el in type_plot_section:
        html += f'<p style="font-size:120%"> {el} </p>'
    html += f"<hr>\n\n"

    html += (
        f'<p style="color:Navy;font-size:160%"> <strong>Features list </strong> : </p>'
    )
    html += "<ul>"
    for feat in titles:
        html += f"<li style=color:Blue><strong><a href=#{feat}> <strong>{feat}</strong></a></li>"
    html += "</ul>"
    html += f"<hr>\n\n"

    for idx, fig in enumerate(figs):
        html += f"<section id ={titles[idx]}>"
        html += f'<p style="text-align:center;font-size:160%">{title+" for : <strong>"+ titles[idx]+"</strong>"}</p>'
        if plot_type != "SHAP_GLOBAL":
            inner_html = pyo.plot(fig, include_plotlyjs=add_js, output_type="div",)
        else:
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format="png", bbox_inches="tight")
            encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
            inner_html = f"<img src='data:image/png;base64,{encoded}'>"
        html += inner_html
        html += "</section>"
        html += f"<hr>"
        add_js = False

    html += "</body></html>\n"

    with open(path, "w") as f:
        f.write(html)

    return None
