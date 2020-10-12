import json
import pickle
from types import ModuleType
from typing import Dict, List, Union

import importlib_resources as pkg_resources
import pandas as pd


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
    return pickle_resource
