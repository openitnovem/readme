from configparser import ConfigParser
from types import ModuleType
from typing import Union

import importlib_resources as pkg_resources

from fbd_interpreter import config


def load_cfg_resource(
    resource_package: Union[ModuleType, str], resource_file_name: str
) -> ConfigParser:
    text = pkg_resources.read_text(resource_package, resource_file_name)
    config = ConfigParser()
    config.read_string(text)
    return config


config_ = load_cfg_resource(config, "config.cfg")
configuration: dict = {s: dict(config_.items(s)) for s in config_.sections()}
