import os
import logging
import sys
from datetime import datetime, timezone, timedelta

from colorlog import ColoredFormatter

def setup_logger(log_dir, log_filename='log.log', logger_name='ColorLogger'):
    class KoreaTimeFormatter(ColoredFormatter):
        def formatTime(self, record, datefmt=None):
            KST = timezone(timedelta(hours=9))
            record_time = datetime.fromtimestamp(record.created, KST)
            return record_time.strftime(datefmt) if datefmt else record_time.isoformat()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # For terminal
    format_head = f"%(log_color)s[%(asctime)s] %(message)s%(reset)s"
    color_formatter = KoreaTimeFormatter(
        format_head,
        datefmt='%Y/%m/%d-%H:%M:%S',
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(color_formatter)

    # For file output
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt='%Y/%m/%d-%H:%M:%S'
    )
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), mode='a', encoding='utf-8')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate=False

    return logger