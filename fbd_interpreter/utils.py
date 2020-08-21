from pprint import pprint

from fbd_interpreter.config.load import configuration


def _parse_config():
    """
    Parse config from cfg file and return dictionnary with keys as config_params
    :return: configuration as dictionnary
    :rtype:
    Example
    -------
    >>> conf = _parse_config()
    >>> len(list(conf.keys())) > 0
    True
    """
    dico_params = {}
    # Get data path as csv / parquet
    dico_params["data_path"] = configuration["DEV"]["data_path"]
    # Get model path as pickle
    dico_params["model_path"] = configuration["DEV"]["model_path"]
    # Get features name as list
    dico_params["features_name"] = configuration["DEV"]["features_name"]
    # Get target_col as string
    dico_params["target_col"] = configuration["DEV"]["target_col"]
    # Get task name as str
    dico_params["task_name"] = configuration["DEV"]["task_name"]
    # Get task name as str
    dico_params["target_col"] = configuration["DEV"]["target_col"]
    # Get features to interpret as list
    dico_params["features_to_interpret"] = configuration["DEV"]["features_to_interpret"]
    # Get output path as str
    dico_params["out_path"] = configuration["DEV"]["output_path"]
    return dico_params


def read_sections_from_txt(file_path):
    """
    Read html sections from txt file
    :param file_path: path to txt file
    :type file_path: str
    :return: Dictionnary of sections with lines as values
    :rtype: dict
     Example
    -------
    >>> dico_sections = read_sections_from_txt("config/sections_html.txt")
    >>> "COMMUN" in (list(dico_sections.keys()))
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
