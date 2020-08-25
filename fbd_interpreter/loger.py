import logging
from fbd_interpreter.config.load import configuration
import colorlog

def get_logger():
    """

    :return:
    :rtype:
    """


    # Create handlers
    log_path ='../outputs/logs.txt'
    f_handler = logging.FileHandler(log_path)

    log_format = '%(asctime)s --- %(name)s --- %(levelname)s --- %(message)s'
    bold_seq = '\033[1m'
    colorlog_format = (
        f'{bold_seq} '
        '%(log_color)s '
        f'{log_format}'
    )
    handler_color = colorlog.StreamHandler()
    handler_color.setFormatter(colorlog.ColoredFormatter(colorlog_format))

    # Create formatters and add it to handlers
    f_format = logging.Formatter(log_format)
    f_handler.setFormatter(f_format)

    # Create a custom logger
    logger = logging.getLogger("fbd_interpreter")
    logger.setLevel(logging.DEBUG)

    # Add handlers to the logger
    logger.addHandler(f_handler)
    logger.addHandler(handler_color)
    return logger

logger = get_logger()


