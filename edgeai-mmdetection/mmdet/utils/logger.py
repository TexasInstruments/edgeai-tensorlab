# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


# edgeailite
# make a logger look like a stream with the write method
# for example get_model_complexity_info needs a stream and not a logger
class LoggerStream():
    def __init__(self, l):
        self.logger = l

    def write(self, msg):
        self.logger.info(msg)