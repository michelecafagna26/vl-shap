import logging
from logging import Formatter
import json
import ast


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()
        #self.default_formatter = Formatter(fmt)

    def format(self, record):

        try:
            # the message is a stringifyed dict != json
            # So we do: str -> dict -> json
            msg = ast.literal_eval(record.getMessage())
        except (ValueError, SyntaxError):
            # jut keep it as it is
            msg = record.getMessage()

        record.message = msg
        record.asctime = self.formatTime(record, self.datefmt)
        out = {"time": record.asctime, "level": record.levelname, "message": record.message}

        return json.dumps(out)


def get_jsonlogger(filename):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename)
    formatter = JsonFormatter()
    #"%(asctime)s:%(levelname)s:%(message)"
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


