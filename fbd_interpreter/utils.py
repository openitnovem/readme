from typing import List

import pandas as pd

from fbd_interpreter import config
from fbd_interpreter.config import env
from fbd_interpreter.config.load import load_cfg_resource
from fbd_interpreter.logger import logger
from fbd_interpreter.resource.data_loader import (
    load_csv_resource,
    load_json_resource,
    load_parquet_resource,
)

# Get configuration as dict from config_{type_env}.cfg
config_ = load_cfg_resource(config, f"config_{env}.cfg")
configuration: dict = {s: dict(config_.items(s)) for s in config_.sections()}


def _parse_config():
    """Parse config from cfg file and return dictionnary with keys as config_params

    Returns
    -------
    dico_params : Dict[str, str]
        configuration as dictionnary

    Example
    -------
    >>> conf = _parse_config()
    >>> len(list(conf.keys())) > 0
    True
    """
    dico_params = {}
    # Get train data path as csv / parquet
    dico_params["train_data_path"] = configuration["PARAMS"]["train_data_path"]
    # Get train data format to load the data
    dico_params["train_data_format"] = configuration["PARAMS"]["train_data_format"]
    # Get test data path as csv / parquet
    dico_params["test_data_path"] = configuration["PARAMS"]["test_data_path"]
    # Get test data format to load the data
    dico_params["test_data_format"] = configuration["PARAMS"]["test_data_format"]
    # Get model path as pickle
    dico_params["model_path"] = configuration["PARAMS"]["model_path"]
    # Get features name as list
    dico_params["features_name"] = configuration["PARAMS"]["features_name"]
    # Get target_col as string
    dico_params["target_col"] = configuration["PARAMS"]["target_col"]
    # Get task name as str
    dico_params["task_name"] = configuration["PARAMS"]["task_name"]
    # Get model type (tree based model or not)
    dico_params["tree_based_model"] = configuration["PARAMS"]["tree_based_model"]
    # Get task name as str
    dico_params["target_col"] = configuration["PARAMS"]["target_col"]
    # Get features to interpret as list
    dico_params["features_to_interpret"] = configuration["PARAMS"][
        "features_to_interpret"
    ]
    # Get output path as str
    dico_params["out_path"] = configuration["PARAMS"]["output_path"]
    # Sanity check
    mandatory_conf = [
        "model_path",
        "features_name",
        "target_col",
        "features_to_interpret",
        "task_name",
        "tree_based_model",
    ]
    missing_conf = False
    for k in mandatory_conf:
        if dico_params[k] == "":
            logger.error(f"Configuration  requires {k} , but is missing ")
            missing_conf = True
    if missing_conf:
        raise KeyError(
            "Missing configuration , please update conf file located in config/config_{type_env}.cfg by "
            "filling in missing keys "
        )
    return dico_params


def check_and_load_data(data_path: str, data_format: str, data_type: str):
    if data_path == "" or data_format == "":
        logger.error(
            f"Configuration file requires {data_type} data path and format, but is missing "
        )
        raise KeyError(
            f"Missing {data_type} data path or format, please update conf file located in config/config_[type_env].cfg "
            f"by filling {data_type}_data_path "
        )
    elif data_format == "parquet":
        data = load_parquet_resource(data_path)
    elif data_format == "csv":
        data = load_csv_resource(data_path)
    else:
        data = load_json_resource(data_path)
    return data


def read_sections_from_txt(file_path: str):
    """
    Read html sections from txt file

    Parameters
    ----------
    file_path : str
        path to txt file containing html sections

    Returns
    -------
    dico_sections : Dict
        Dictionnary of sections with lines as values

    Example
    -------
    >>> import os
    >>> "COMMUN" in (list(read_sections_from_txt(os.path.abspath("fbd_interpreter/config/sections_html.txt")).keys()))
    True
    """
    with open(file_path, mode="r") as f:
        text = f.readlines()
    dico_sections = {}
    current_section = None
    for line in text:
        if line.startswith("#"):
            dico_sections[line.split("#")[1].strip()] = ""
            current_section = line.split("#")[1].strip()
        else:
            dico_sections[current_section] = (
                dico_sections[current_section] + "||" + line
            )
    # Get sentences as list for each section
    dico_sections = {
        k: [el for el in v.split("||")[1:] if el != "\n"]
        for k, v in dico_sections.items()
    }
    return dico_sections


def optimize(
    df: pd.DataFrame,
    datetime_features: List[str] = [],
    datetime_format: str = "%Y%m%d",
    prop_unique: float = 0.5,
):
    """
    Returns a pandas dataframe with better memory allocation by downcasting the columns
    automatically to the smallest possible datatype without losing any information.
    For strings we make use of the pandas category column type if the amount of unique
    strings cover less than half the total amount of strings.
    We cast date columns to the pandas datetime dtype. It does not reduce memory usage,
    but enables time based operations.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to reduce
    datetime_features : List[str]
        List of date columns to cast to the pandas datetime dtype
    datetime_format : str, optional
        datetime features format (the default is "%Y%m%d")
    prop_unique : float, optional = 0.5
        max proportion of unique values in object columns to allow casting to category type
        (the default is 0.5)

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': [1, 2, 3, 4], 'col2': [3.5, 4.89, 2.9, 3.1], 'col3': ["M", "F", "M", "F"]}
    >>> df = pd.DataFrame(d)
    >>> print(optimize(df).dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'int8', 'col2': 'float32', 'col3': 'category'}
    """
    optimized_df = optimize_floats(
        optimize_ints(
            optimize_objects(df, datetime_features, datetime_format, prop_unique)
        )
    )
    return optimized_df


def optimize_floats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe after downcasting the float columns to the smallest
     possible float datatype (float32, float64) using pd.to_numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to reduce

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation for float columns

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': [1, 2, 3, 4], 'col2': [3.5, 4.89, 2.9, 3.1], 'col3': ["M", "F", "M", "F"]}
    >>> df = pd.DataFrame(d)
    >>> print(optimize_floats(df).dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'int64', 'col2': 'float32', 'col3': 'object'}
    """
    floats = df.select_dtypes(include=["float64"]).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast="float")
    return df


def optimize_ints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pandas dataframe after downcasting the integer columns to the smallest
     possible int datatype (int8, int16, int32, int64) using pd.to_numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to reduce

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation for int columns

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': [1, 2, 3, 4], 'col2': [3.5, 4.89, 2.9, 3.1], 'col3': ["M", "F", "M", "F"]}
    >>> df = pd.DataFrame(d)
    >>> print(optimize_ints(df).dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'int8', 'col2': 'float64', 'col3': 'object'}
    """
    ints = df.select_dtypes(include=["int64"]).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast="integer")
    return df


def optimize_objects(
    df: pd.DataFrame,
    datetime_features: List[str],
    datetime_format: str = "%Y%m%d",
    prop_unique: float = 0.5,
) -> pd.DataFrame:
    """
    Returns a pandas dataframe after downcasting the object columns to the smallest
    possible datatype.
    For strings we make use of the pandas category column type if the amount of unique
    strings cover less than the proportion p (default 50%) of the total amount of strings.
    We cast date columns to the pandas datetime dtype. It does not reduce memory usage,
    but enables time based operations.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to reduce
    datetime_features : List[str]
        List of date columns to cast to the pandas datetime dtype
    datetime_format : str, optional
        datetime features format (the default is "%Y%m%d")
    prop_unique : float, optional
        max proportion of unique values to allow casting to category type
        (the default is 0.5)

    Returns
    -------
    optimized_df : pd.DataFrame
        Pandas dataframe with better memory allocation for object columns

    Example
    -------
    >>> import pandas as pd
    >>> d = {'col1': ["08:10", "10:15", "12:30", "06:00"], 'col2': ["M", "F", "M", "F"]}
    >>> df = pd.DataFrame(d)
    >>> print(optimize_objects(df, 'col1', "%H:%M").dtypes.apply(lambda x: x.name).to_dict())
    {'col1': 'datetime64[ns]', 'col2': 'category'}
    """

    for col in df.select_dtypes(include=["object"]):
        if col not in datetime_features:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if float(num_unique_values) / num_total_values <= prop_unique:
                df[col] = df[col].astype("category")
        else:
            df[col] = pd.to_datetime(df[col], format=datetime_format)
    return df
