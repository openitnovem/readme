import os
from pprint import pformat

import click

from fbd_interpreter.resource.data_loader import (
    load_csv_resource,
    load_parquet_resource,
    load_pickle_resource,
)
from fbd_interpreter.explainers.core import Interpreter
from fbd_interpreter.logger import logger
from fbd_interpreter.utils import _parse_config, optimize


@click.command()
@click.option(
    "--interpret-type",
    default="mix",
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
    default=False,
    show_default=True,
    metavar="",
    help="Computes and plots PDP & ICE",
)
@click.option(
    "--use-shap",
    default=True,
    show_default=True,
    metavar="",
    help="Computes and plots shapely values for global & local explanation",
)
def interept(
    interpret_type: str = "mix",
    use_ale: bool = True,
    use_pdp_ice: bool = True,
    use_shap: bool = True,
):
    """
    Interpret locally, globally or both any ML model using PDP, ICE, ALE & SHAP.
    Before using this function, you must fill in the config file located in
    config/config_{type_env}.cfg.
    Note that to speed up computations, we apply the function `optimize` to reduce the pandas
    dataframes memory usage, by downcasting the columns automatically to the smallest possible
    datatype without losing any information.

    Parameters
    ----------
    interpret_type : str, optional
        Type of interpretability global, local or mix(both). (the default is "mix", which implies
        global and local interpretability)
    use_ale : bool, optional
        If True, computes ALE: Accumulated Local Effects.
        Can only be used for numerical features. (the default is True)
    use_pdp_ice : bool, optional
        If True, computes PDP & ICE: Partial Dependency & Individual Expectation plots.
        (the default is True)
    use_shap : bool, optional
        If True, computes SHAP plots. (the default is True)

    Returns
    -------
    None
    """
    config_values = _parse_config()
    logger.info("Configuration settings :\n" + pformat(config_values))
    logger.info("Loading model")
    model = load_pickle_resource(config_values["model_path"])
    tree_based_model = True if config_values["tree_based_model"] == "True" else False

    exp = Interpreter(
        model=model,
        task_name=config_values["task_name"],
        tree_based_model=tree_based_model,
        features_name=config_values["features_name"].split(","),
        features_to_interpret=config_values["features_to_interpret"].split(","),
        target_col=config_values["target_col"],
        out_path=config_values["out_path"],
    )
    if interpret_type == "global" or interpret_type == "mix":
        logger.info("Interpretability type : global")
        train_data_path = config_values["train_data_path"]
        train_data_format = config_values["train_data_format"]
        if train_data_path == "" or train_data_format == "":
            logger.error(
                f"Configuration file requires train data path and format, but is missing "
            )
            raise KeyError(
                "Missing train data path, please update conf file located in config/config_{type_env}.cfg"
                " by filling train_data_path "
            )
        elif train_data_format == "parquet":
            train_data = load_parquet_resource(train_data_path)

        else:
            train_data = load_csv_resource(train_data_path)

        logger.info("Reducing train dataframe memory usage to speed up computations")
        train_data = optimize(train_data)
        if use_pdp_ice:
            exp.global_pdp_ice(train_data)
        if use_ale:
            exp.global_ale(train_data)
        if use_shap:
            exp.global_shap(train_data)

    if interpret_type == "local" or interpret_type == "mix":
        logger.info("Interpretability type : local")
        test_data_path = config_values["test_data_path"]
        if test_data_path == "":
            logger.error(
                f"Configuration file requires test data path , but is missing "
            )
            raise KeyError(
                "Missing test data path, please update conf file located in config/config_{type_env}.cfg"
                " by filling test_data_path "
            )
        elif os.path.isdir(test_data_path):
            test_data = load_parquet_resource(test_data_path)
        else:
            test_data = load_csv_resource(test_data_path)
        logger.info("Reducing test dataframe memory usage to speed up computations")
        test_data = optimize(test_data)
        exp.local_shap(test_data)

    else:
        raise Exception  # Not supported


if __name__ == "__main__":
    interept()
