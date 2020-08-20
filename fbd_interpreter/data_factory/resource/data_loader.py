import json
import pickle
from configparser import ConfigParser
from types import ModuleType
from typing import Dict, List, Union

import pandas as pd

import importlib_resources as pkg_resources
from fbd_interpreter.config.load import load_cfg_resource
from fbd_interpreter.data_factory import resource


def load_json_resource(
    resource_package: Union[ModuleType, str], resource_file_name: str
) -> Union[List, Dict]:
    text = pkg_resources.read_text(resource_package, resource_file_name)
    return json.loads(text)


def load_csv_resource(resource_file_name: str,) -> pd.DataFrame:
    df = pd.read_csv(resource_file_name)
    return df


def load_parquet_resource(resource_file_name: str,) -> pd.DataFrame:
    df = pd.read_parquet(resource_file_name)
    return df


def load_pickle_resource(resource_file_name: str):
    pickle_resource = pickle.load(open(resource_file_name, "rb"))
    print("Pickle loaded")
    return pickle_resource


parameters_: ConfigParser = load_cfg_resource(resource, "parameters.cfg")

parameters: dict = {s: dict(parameters_.items(s)) for s in parameters_.sections()}
