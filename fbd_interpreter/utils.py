from fbd_interpreter.config.load import configuration
from fbd_interpreter.logger import logger


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
    # Get train data path as csv / parquet
    dico_params["train_data_path"] = configuration["DEV"]["train_data_path"]
    # Get test data path as csv / parquet
    dico_params["test_data_path"] = configuration["DEV"]["test_data_path"]
    # Get model path as pickle
    dico_params["model_path"] = configuration["DEV"]["model_path"]
    # Get features name as list
    dico_params["features_name"] = configuration["DEV"]["features_name"]
    # Get target_col as string
    dico_params["target_col"] = configuration["DEV"]["target_col"]
    # Get task name as str
    dico_params["task_name"] = configuration["DEV"]["task_name"]
    # Get model type (tree based model or not)
    dico_params["tree_based_model"] = configuration["DEV"]["tree_based_model"]
    # Get task name as str
    dico_params["target_col"] = configuration["DEV"]["target_col"]
    # Get features to interpret as list
    dico_params["features_to_interpret"] = configuration["DEV"]["features_to_interpret"]
    # Get output path as str
    dico_params["out_path"] = configuration["DEV"]["output_path"]
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
            "Missing configuration , please update conf file located in config/config_{type_env}.cfg by filling missing keys "
        )
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


def _names_with_values(names, values):
    li = []
    for name, value in zip(names, values):
        if value == "":
            li.append("{0}".format(name))
        else:
            li.append("{0} ({1})".format(name, value))

    return li
