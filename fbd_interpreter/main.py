import logging

import click

from fbd_interpreter.modules.core import Interpreter
from fbd_interpreter.utils import _parse_config

logging.getLogger().setLevel(logging.INFO)


@click.command()
@click.option(
    "--interpret-type",
    default="global",
    metavar="",
    help="Type d'interprétabilité: choisir global, local ou mix",
)
@click.option(
    "--use-ale",
    default=True,
    metavar="",
    help="Calculer et afficher les plots ALE, par défaut calculés",
)
@click.option(
    "--use-pdp-ice",
    default=True,
    metavar="",
    help="Calculer et afficher les plots PDP & ICE, par défaut calculés",
)
@click.option(
    "--use-shap",
    default=False,
    metavar="",
    help="Calculer et afficher les plots de feature importance SHAP, par défaut calculés",
)
def interept(interpret_type, use_ale, use_pdp_ice, use_shap):
    config_values = _parse_config()
    print(config_values)
    exp = Interpreter(
        model_path=config_values["model_path"],
        data_path=config_values["data_path"],
        task_name=config_values["task_name"],
        features_name=config_values["features_name"],
        target_col=config_values["target_col"],
        out_path=config_values["out_path"],
    )
    if interpret_type == "global" or interpret_type == "mix":
        logging.info(f"Type d'interprétabilité est : {interpret_type}e")
        if use_pdp_ice:
            exp.global_pdp_ice()
        if use_ale:
            exp.global_ale()
        if use_shap:
            raise NotImplemented
            # exp.global_shap()

    elif interpret_type == "local" or interpret_type == "mix":
        logging.info(f"Type d'interprétabilité est : {interpret_type}e")
        raise NotImplemented

    else:
        raise Exception  # Not supported


if __name__ == "__main__":
    interept()
