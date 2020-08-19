from typing import List

import plotly.graph_objs as go
import plotly.offline as pyo


def plotly_figures_to_html(dic_figs, path: str, title: str = "") -> None:
    """Convert a dict of plotly figures to html format.

    Args:
        dic_figs: Dict of plotly figures to be saved.
        path: Path to write html file
        title: Title
    Returns:
        string in HTML format
    """
    html = "<html><head></head><body>\n"
    html += (
        f'<h1 style="color:SlateBlue;text-align:center;font-size:300%">{title}</h1>\n\n'
    )
    figs = list(dic_figs.values())
    titles = list(dic_figs.keys())

    add_js = True
    # html += f'<h1 style="text-align:center;font-size:160%">STOP </h1>'
    html += (
        f'<p style="font-size:160%">Rappel\n l\'interpretation \n points '
        f"attention </p>"
    )
    html += f'<p style="font-size:160%"> <strong>Features list </strong> : </p>'
    html += "<ul>"
    for feat in titles:
        html += f"<li style=color:blue><strong><a href=#{feat}> <strong>{feat}</strong></a></li>"
    html += "</ul>"
    html += f"<hr>\n\n"

    for idx, fig in enumerate(figs):
        html += f"<section id ={titles[idx]}>"
        html += f'<p style="text-align:center;font-size:160%">{title+" for : <strong>"+ titles[idx]+"</strong>"}</p>'
        inner_html = pyo.plot(fig, include_plotlyjs=add_js, output_type="div",)
        html += inner_html
        html += "</section>"
        html += f"<hr>"
        add_js = False

    html += "</body></html>\n"

    with open(path, "w") as f:
        f.write(html)

    return None
