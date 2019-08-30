import csv
import random
import sys
from Controller.LogController import LogController
from Controller.Controller import Controller
import traceback
import numpy as np

class CerebroSigmoidal():
    def __init__(self, pesos):
        self.log = LogController()
        self.log.logInfo("\n\n\n\n\n\n\n\n\n\n\n\n\n\nCérebro ligado!", True)  
        
        if isinstance(pesos, np.ndarray):
            self.pesos = pesos
            
        else:
            np.random.seed(1)
            self.pesos = 2 * np.random.random((pesos, 1)) - 1
            
            self.log.logInfo('Pesos aleatórios:')
            self.log.logInfo(str(self.pesos), True)
        
    def sigmoid(self, x):      
        norm = 1 / (1 + np.exp(-x))     
        norm = np.minimum(norm, 0.9999)
        norm = np.maximum(norm, 0.0001)

        return norm

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def training(self, inputTraining, outputTraining, training_iterations):
        for iteration in range(training_iterations):
            output = self.synapse(inputTraining)

            error = outputTraining - output
            adjustments = np.dot(inputTraining.T, error * self.calculateError(output))

            self.pesos += adjustments

    def synapse(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.pesos))
        return output

    def calculateError(self, x):
        return x * (1 - x)

    def initProcesso(self, inputTraining, outputTraining):
        self.training(inputTraining, outputTraining, 10000)

        self.log.logInfo('Pesos após treinamento:')
        self.log.logInfo(str(self.pesos), True)

        Controller().datasetToCsv(self.pesos, "pesos_atualizados")