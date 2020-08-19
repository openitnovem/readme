from fbd_interpreter.config.load import configuration


def _parse_config():
    """
    Parse config from cfg file and return dictionnary with keys as config_params
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
    # Get output path as str
    dico_params["out_path"] = configuration["DEV"]["output_path"]
    return dico_params


def update_plotly_menus():
    list_updatemenus = [
        {
            "label": "Option 1",
            "method": "update",
            "args": [{"visible": [True, False, False]}, {"title": "Title is Option 1"}],
        },
        {
            "label": "Option 2",
            "method": "update",
            "args": [{"visible": [False, True, False]}, {"title": "Title is Option 2"}],
        },
        {
            "label": "Option 3",
            "method": "update",
            "args": [{"visible": [False, False, True]}, {"title": "Title is Option 3"}],
        },
    ]

    return list_updatemenus
