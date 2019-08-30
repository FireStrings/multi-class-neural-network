#!/usr/bin/env python

import logging
import logging.handlers
import datetime


class LogController():
    def __init__(self):
        self.log = logging.getLogger("NeuralNetwork")
        self.log.setLevel(logging.DEBUG)
        logfile = "/var/log/NeuralNetwork.log"
        
        h1 = logging.handlers.RotatingFileHandler(logfile, mode='a', maxBytes=104857600, backupCount=5, encoding=None, delay=0)
        f = logging.Formatter("%(levelname)s [%(asctime)s] %(name)s: %(message)s")
        h1.setFormatter(f)
        self.log.addHandler(h1)
    
    def logInfo(self, message, quebra_linha=False):
        if quebra_linha:
            self.log.info(message + "\n")
        else:
            self.log.info(message)

    def logError(self, traceback):
        self.log.error(traceback)          








                                      