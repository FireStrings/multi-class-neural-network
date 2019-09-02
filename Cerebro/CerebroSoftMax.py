import csv
import random
import sys
from Controller.LogController import LogController
from Controller.Controller import Controller
import traceback
import numpy as np

class CerebroSoftMax():

    def __init__(self):
        self.log = LogController()
        self.log.logInfo("\n\n\n\n\n\n\n\n\n\n\n\n\n\nCérebro Ligado!", True)      

        self.pesosH = None
        self.pesosO = None
        self.biasH = None
        self.biasO = None


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerivate(self, x):
        return self.sigmoid(x) *(1-self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)

    def test(self, newInput, pesosH, pesosO, biasH, biasO):        
        
        pesosHidden = pesosH
        pesosOut = pesosO
        biasHidden = biasH            
        biasOut = biasO


        inputsV = np.vstack([newInput])
        
        hiddenNodes = 4
        outputNodes = 3

        hiddenSum = np.dot(inputsV, pesosHidden) + biasHidden        
        hiddenActv = self.sigmoid(hiddenSum)
        
        outSum = np.dot(hiddenActv, pesosOut) + biasOut        
        outActv = self.softmax(outSum) 

        return outActv

    
    def training(self, inputTraining , outputTraining, iterations, pesos=None):    
        
        inputTraining = np.array([
        np.array([0.9, -0.1, -0.8, 0.3, 0.7, -0.2, 0.9, -0.5, 0.6, -0.1]), 
        np.array([-0.2, -0.4, 0.9, 0.3, -0.8, 0.5, -0.6, 0.4, -0.1, 1]),
        np.array([-1, 0.3, -0.8, 0.2, 0.4, -0.7, 0.9, -0.1, 0.3, -0.7]),
        np.array([0, 0.1, -0.6, -0.1, 0.3, -0.5, 0.8, -0.1, 0.4, -0.8]),
        np.array([0.2, 0.2, -0.3, -0.2, 0.4, -0.6, 0.9, -0.2, 0.7, -0.2]),
        np.array([0.3, -0.1, -0.4, -0.1, -0.4, 0.6, 0.8, -0.3, -0.5, 0.1]),
        np.array([0.1, 0.2, -0.8, -0.4, 0.1, 0.5, -0.6, 0.2, -0.5, -0.1]),
        np.array([-0.1, 0.4, 0.2, -0.9, 0.5, -0.1, 0.6, -0.3, 0.9, 0.1]),
        np.array([-0.5, -0.2, 0.3, -0.1, 0.4, 0.2, -0.5, 0.9, -0.7, 0.3]),
        np.array([0.7, 0.1, -0.5, 0.8, -0.3, 0.4, -0.8, 0.7, 0.2, -0.7])
        ])

        # print(outputTraining)
        # sys.exit()

        inputsV = np.vstack([inputTraining])   

        attributes = inputsV.shape[1]        
        
        hiddenNodes = 4
        outputNodes = len(inputTraining)

        if pesos == None:
            pesosHidden = np.random.rand(attributes,hiddenNodes)
            pesosOut = np.random.rand(hiddenNodes,outputNodes)   
            biasHidden = np.random.randn(hiddenNodes)                 
            biasOut = np.random.randn(outputNodes)   

        else:
            pesosHidden = pesos[0]
            pesosOut = pesos[1]
            biasHidden = pesos[2]
            biasOut = pesos[3]

        learningRate = 0.01
        learningRate = 10e-4

        errorCost = []
        outActv = None

        for epoch in range(iterations):  
            hiddenSum = np.dot(inputsV, pesosHidden) + biasHidden        
            hiddenActv = self.sigmoid(hiddenSum)
            
            outSum = np.dot(hiddenActv, pesosOut) + biasOut        
            outActv = self.softmax(outSum)   

            
            diffOutActv = outActv - outputTraining
            sumHiddenActvOut = np.dot(hiddenActv.T, diffOutActv)

            sumDiffOutActvPesosOut = np.dot(diffOutActv, pesosOut.T)            

            sigDer = self.sigmoidDerivate(hiddenSum)
            sumInputsSigDer = np.dot(inputsV.T, sigDer * sumDiffOutActvPesosOut)

            dotSigDer = sumDiffOutActvPesosOut * sigDer

            pesosHidden -= learningRate * sumInputsSigDer
            biasHidden -= learningRate * dotSigDer.sum(axis=0)

            pesosOut -= learningRate * sumHiddenActvOut
            biasOut -= learningRate * diffOutActv.sum(axis=0)

            if epoch % 200 == 0:
                loss = np.sum(-outputTraining * np.log(outActv))
                print('Valor da função de custo de erro: ', loss)
                errorCost.append(loss)

            self.pesosH = pesosHidden
            self.pesosO = pesosOut
            self.biasH = biasHidden
            self.biasO = biasOut
        # print()

        for i in outActv:
            
            print(
            "{:.16f}".format(float(str(i[0]))) + " " + 
            "{:.16f}".format(float(str(i[1]))) + " " + 
            "{:.16f}".format(float(str(i[2]))) + " " + 
            "{:.16f}".format(float(str(i[3]))) + " " + 
            "{:.16f}".format(float(str(i[4]))) + " " + 
            "{:.16f}".format(float(str(i[5]))) + " " + 
            "{:.16f}".format(float(str(i[6]))) + " " + 
            "{:.16f}".format(float(str(i[7]))) + " " + 
            "{:.16f}".format(float(str(i[8]))) + " " + 
            "{:.16f}".format(float(str(i[9])))
            )

        return loss

    def initProcesso(self, inputTraining, outputTraining, pesos=None):
        loss = self.training(inputTraining, outputTraining, 1000000, pesos)

        # self.log.logInfo('Pesos após treinamento:')
        # self.log.logInfo(str(self.pesos), True)

        # if loss < 5:
        #     Controller().datasetToCsv(self.pesosH, "pesosH")
        #     Controller().datasetToCsv(self.pesosO, "pesosO")
        #     Controller().datasetToCsv(self.biasH, "biasH")
        #     Controller().datasetToCsv(self.biasO, "biasO")