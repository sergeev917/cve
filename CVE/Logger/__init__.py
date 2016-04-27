__all__ = (
    'ConsoleLog',
    'set_default_logger',
    'get_default_logger',
)

from .Console import (
    ConsoleLog,
    DevnullLog,
)

# default logger is not set by default
default_logger = DevnullLog()

def set_default_logger(logger):
    global default_logger
    default_logger = logger

def get_default_logger():
    global default_logger
    return default_logger
