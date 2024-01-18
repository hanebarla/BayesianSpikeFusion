import os
import sys
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG
from termcolor import colored

def create_logger(savedir, name, filename='log.txt'):
    logger = getLogger(name)
    logger.setLevel(DEBUG)

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'
    
    console_handler = StreamHandler(sys.stdout)
    console_handler.setLevel(DEBUG)
    console_handler.setFormatter(Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    file_handler = FileHandler(os.path.join(savedir, filename), mode='a')
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger