"""loggers.py

Setup utility for adding consistent logging to all TCIA modules
"""
import os
import logging
import logging.handlers

def RotatingFile(logpath, logname, errorlogname=None):
    """initialize and return standard logger and error logger objects

    Args:
        logname (str): basename for the logfile that is initialized. Error log will have name:
            <logname>_Errors.log by default
    """
    logname = os.path.splitext(logname)[0]
    logfile_path = os.path.join(logpath, logname + '.log')#.format(time.strftime('%Y%b%d_%H:%M:%S')))
    # get a named logger - if not named, root will get messages from children loggers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create common formatter
    formatter = logging.Formatter(fmt='%(message)s')

    # create separate handlers for stream and file
    sh = logging.StreamHandler()  # defaults to sys.stderr
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    rfh = logging.handlers.RotatingFileHandler(logfile_path, mode='w', maxBytes=0, backupCount=10, delay=True)
    rfh.setFormatter(formatter)
    logger.addHandler(rfh)

    # error logger
    if not errorlogname:
        errorlogname = logname + '_Errors'
    else:
        errorlogname = os.path.splitext(errorlogname)[0]
    errlogfile_path = os.path.join(logpath, errorlogname + '.log')
    err_rfh = logging.handlers.RotatingFileHandler(errlogfile_path, mode='w', maxBytes=0, backupCount=10, delay=True)
    err_rfh.setFormatter(formatter)
    err_rfh.setLevel(logging.ERROR)
    logger.addHandler(err_rfh)

    # Initialize Loggers
    os.makedirs(logpath, exist_ok=True)
    rfh.doRollover()
    err_rfh.doRollover()

    # return loggers
    return logger

def AppendingRotatingFile(logpath, logname, errorlogname=None):
    """initialize and return standard logger and error logger objects

    Args:
        logname (str): basename for the logfile that is initialized. Error log will have name:
            <logname>_Errors.log by default
    """
    logname = os.path.splitext(logname)[0]
    logfile_path = os.path.join(logpath, logname + '.log')#.format(time.strftime('%Y%b%d_%H:%M:%S')))
    # get a named logger - if not named, root will get messages from children loggers
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create common formatter
    formatter = logging.Formatter(fmt='%(message)s')

    # create separate handlers for stream and file
    sh = logging.StreamHandler()  # defaults to sys.stderr
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    rfh = logging.handlers.RotatingFileHandler(logfile_path, mode='a', maxBytes=5e7, backupCount=5, delay=True)
    rfh.setFormatter(formatter)
    logger.addHandler(rfh)

    # error logger
    if not errorlogname:
        errorlogname = logname + '_Errors'
    else:
        errorlogname = os.path.splitext(errorlogname)[0]
    errlogfile_path = os.path.join(logpath, errorlogname + '.log')
    err_rfh = logging.handlers.RotatingFileHandler(errlogfile_path, mode='a', maxBytes=5e7, backupCount=5, delay=True)
    err_rfh.setFormatter(formatter)
    err_rfh.setLevel(logging.ERROR)
    logger.addHandler(err_rfh)

    # Initialize Loggers
    os.makedirs(logpath, exist_ok=True)

    # return loggers
    return logger
