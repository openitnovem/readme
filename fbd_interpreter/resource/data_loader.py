import json
import pickle
from typing import Dict

import pandas as pd


def load_json_resource(resource_file_name: str) -> Dict:
    f = open(resource_file_name)
    # returns JSON object as a dictionary
    data = json.load(f)
    f.close()
    return data


def load_csv_resource(resource_file_name: str,) -> pd.DataFrame:
    df = pd.read_csv(resource_file_name)
    return df


def load_parquet_resource(resource_file_name: str,) -> pd.DataFrame:
    df = pd.read_parquet(resource_file_name)
    return df


def load_pickle_resource(resource_file_name: str):
    pickle_resource = pickle.load(open(resource_file_name, "rb"))
    return pickle_resource
