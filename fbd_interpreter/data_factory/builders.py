import logging
import os

from fbd_interpreter.config.load import configuration

logging.getLogger().setLevel(logging.INFO)


def initialize_dir(out_path):
    """
    Check if output dir tree exists if not it will be created
    """
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
        logging.info("%s %s ", "Directory created", out_path)
    if not os.path.isdir(os.path.join(out_path, "local_interpretation")):
        os.makedirs(os.path.join(out_path, "local_interpretation"))
        logging.info(
            "%s %s ",
            "Directory created",
            os.path.join(out_path, "local_interpretation"),
        )
    if not os.path.isdir(os.path.join(out_path, "global_interpretation")):
        os.makedirs(os.path.join(out_path, "global_interpretation"))
        logging.info(
            "%s %s ",
            "Directory created",
            os.path.join(out_path, "global_interpretation"),
        )


output_path = configuration["DEV"]["output_path"]
initialize_dir(output_path)
