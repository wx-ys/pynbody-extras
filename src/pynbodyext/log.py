import logging
import sys

logger = logging.getLogger("pynext")

class DuplicateFilter(logging.Filter):
    """A filter that removes duplicated successive log entries."""

    # source    #yt
    # https://stackoverflow.com/questions/44691558/suppress-multiple-messages-with-same-content-in-python-logging-module-aka-log-co
    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg, record.args)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False

ufstring = "%(name)-3s: [%(levelname)-9s] %(asctime)s %(message)s"
cfstring = "%(name)-3s: [%(levelname)-18s] %(asctime)s %(message)s"


logger.setLevel(logging.INFO)
logger.addFilter(DuplicateFilter())

f = logging.Formatter(ufstring)

def stream_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(f)
    return handler


logger.addHandler(stream_handler())
