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
    help="Type d'interprétabilité: globale, locale ou les deux",
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
    default=True,
    metavar="",
    help="Calculer et afficher les plots SHAP, par défaut calculés",
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
    if interpret_type == "global":
        logging.info(f"Type d'interprétabilité est : {interpret_type}")
        # TODO: split interpret_globally to 4 class/methodes
        fig_ = exp.intepreter_globaly(
            use_ale=use_ale, use_pdp=use_pdp_ice, use_ice=use_pdp_ice, use_shap=use_shap
        )
        # figures = [fig.data[0]._data_objs for k, fig in fig_.items()]
        # exp.stack_figures(figures)

    elif interpret_type == "local":
        if interpret_type == "local":
            logging.info(f"Type d'interprétabilité est : {interpret_type}")
            # fig_ = exp.intepreter_locally()
        raise NotImplemented
    elif interpret_type == "mix":
        raise NotImplemented
    else:
        raise Exception  # Not supported


if __name__ == "__main__":
    interept()
