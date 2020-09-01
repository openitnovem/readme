import os

from fbd_interpreter.config.load import configuration
from fbd_interpreter.logger import logger


def initialize_dir(out_path: str):
    """
    Check if output dir tree exists if not it will be created.
    Output dir includes global_interpretation and local_interpretation folders
    """
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        logger.info("%s %s ", "Directory created", out_path)
    if not os.path.isdir(os.path.join(out_path, "local_interpretation")):
        os.makedirs(os.path.join(out_path, "local_interpretation"))
        logger.info(
            "%s %s ",
            "Directory created",
            os.path.join(out_path, "local_interpretation"),
        )
    if not os.path.isdir(os.path.join(out_path, "global_interpretation")):
        os.makedirs(os.path.join(out_path, "global_interpretation"))
        logger.info(
            "%s %s ",
            "Directory created",
            os.path.join(out_path, "global_interpretation"),
        )


output_path = configuration["DEV"]["output_path"]
initialize_dir(output_path)
