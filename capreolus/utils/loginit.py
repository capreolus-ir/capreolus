import logging
import logging.handlers
import os

import colorlog

GLOBAL_LOGGING_CONF = {"level": os.environ.get("CAPREOLUS_LOGGING", logging.INFO)}


class RepeatFilter(logging.Filter):
    def __init__(self, logger, maxlevel=logging.DEBUG, max_repeats=5):
        super(RepeatFilter, self).__init__()
        self.logger = logger
        self.maxlevel = maxlevel
        self.max_repeats = max_repeats
        self.last = None
        self.last_count = 0
        self.notified = False

    def filter(self, record):
        # TODO is there a cleaner way to ignore messages produced by this filter?
        if record.funcName == "filter" and record.msg.startswith("RepeatFilter"):
            return True

        # message's level is higher than our max level, so log it
        if record.levelno > self.maxlevel:
            return True

        current = (record.module, record.funcName, record.levelno, record.msg)

        # message is new, so restart count and log it
        if current != self.last:
            self.last = current
            self.last_count = 1
            self.notified = False
            return True
        else:
            # message has been repeated less than max_repeats times, so log it
            if self.last_count < self.max_repeats:
                self.last_count += 1
                return True

            # message has reached the maximum number of repeats, so don't log the message
            # if we haven't yet logged a notification that RepeatFilter was triggered, log one
            if not self.notified:
                self.logger.log(
                    record.levelno, "RepeatFilter suppressing additional variations of past %s messages", self.last_count
                )
                self.notified = True
            return False


def _streamhandler():
    fmt = "%(thin_white)s%(asctime)s - %(reset)s%(log_color)s%(levelname)s - %(name)s.%(funcName)s - %(message)s"
    sh = colorlog.StreamHandler()
    sh.setFormatter(colorlog.ColoredFormatter(fmt))

    return sh


def get_logger(name=None):
    # create a root logger for warnings and above
    rlogger = logging.getLogger()
    if len(rlogger.handlers) == 0:
        rlevel = logging.WARNING
        rlogger.setLevel(rlevel)

        rlogger.addHandler(_streamhandler())

    # create a capreolus logger for debug messages
    logger = logging.getLogger("capreolus")
    if len(logger.handlers) == 0:
        logger.propagate = False

        level = logging.DEBUG
        logger.setLevel(level)

        sh = _streamhandler()
        sh.addFilter(RepeatFilter(logger))
        logger.addHandler(sh)

        logger.setLevel(GLOBAL_LOGGING_CONF["level"])

    if name is None:
        name = "capreolus"
    if not name.startswith("capreolus"):
        name = "capreolus." + name
    return logging.getLogger(name)
