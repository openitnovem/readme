from fbd_interpreter.visualization.plots import interpretation_plots_to_html_report
import plotly.graph_objects as go
from fbd_interpreter.logger import logger


def test_plots() -> None:
    # Create dummy figure to plot as html with available content of sections
    f = go.FigureWidget()
    f.add_scatter(y=[2, 1, 4, 3]);
    dict_figures = {"DUMMY":f}
    logger.info("f")
    html = interpretation_plots_to_html_report(dict_figures)
    print(html)
    assert 1==1

#test_plots()