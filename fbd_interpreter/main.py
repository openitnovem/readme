from pprint import pformat
import click

from fbd_interpreter.explainers.core import Interpreter
from fbd_interpreter.utils import _parse_config
from fbd_interpreter.loger import logger


@click.command()
@click.option(
    "--interpret-type",
    default="global",
    show_default=True,
    metavar="",
    help="Interpretability type: Choose global, local or mix",
)
@click.option(
    "--use-ale",
    default=True,
    show_default=True,
    metavar="",
    help="Computes and plots ALE",
)
@click.option(
    "--use-pdp-ice",
    default=True,
    show_default=True,
    metavar="",
    help="Computes and plots PDP & ICE",
)
@click.option(
    "--use-shap",
    default=False,
    show_default=True,
    metavar="",
    help="Computes and plots shapely values for global & local explanation",
)
def interept(interpret_type, use_ale, use_pdp_ice, use_shap):
    config_values = _parse_config()
    logger.info(f"Configuration settings :\n"+pformat(config_values))

    exp = Interpreter(
        model_path=config_values["model_path"],
        data_path=config_values["data_path"],
        task_name=config_values["task_name"],
        features_name=config_values["features_name"],
        target_col=config_values["target_col"],
        out_path=config_values["out_path"],
    )
    if interpret_type == "global" or interpret_type == "mix":
        logger.info(f"Interpretability type : {interpret_type}")
        if use_pdp_ice:
            exp.global_pdp_ice()
        if use_ale:
            exp.global_ale()
        if use_shap:
            raise NotImplemented
            # exp.global_shap()

    elif interpret_type == "local" or interpret_type == "mix":
        logger.info(f"Interpretability type : {interpret_type}")
        raise NotImplemented

    else:
        raise Exception  # Not supported


if __name__ == "__main__":
    interept()
