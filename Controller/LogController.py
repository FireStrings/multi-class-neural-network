#!/usr/bin/env python

import logging
import logging.handlers
import datetime

loggers = {}

class LogController():
    global loggers

    def __init__(self):
        pass

    
    def getLogger():
        global loggers

        if loggers.get("NeuralNetwork"):
            return loggers.get("NeuralNetwork")
        else:
            log = logging.getLogger("NeuralNetwork")
            log.setLevel(logging.DEBUG)
            logfile = "/var/log/NeuralNetwork.log"
            
            h1 = logging.handlers.RotatingFileHandler(logfile, mode='a', maxBytes=104857600, backupCount=5, encoding=None, delay=0)
            f = logging.Formatter("%(levelname)s [%(asctime)s] %(name)s: %(message)s")
            h1.setFormatter(f)
            log.addHandler(h1)
            loggers["NeuralNetwork"] = log
            
            return log        








                                      