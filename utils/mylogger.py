import logging
from termcolor import colored
from datetime import datetime
import sys

# set up logger
class _MyFormatter(logging.Formatter):

    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)

def init_logger(log_file=None):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger('SXC')
    logger.propagate = False
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    # logger.addHandler(console_handler)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

if __name__=='__main__':
    logger = init_logger()
    logger.info("ok")