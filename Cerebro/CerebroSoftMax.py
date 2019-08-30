import csv
import random
import sys
from Controller.LogController import LogController
from Controller.Controller import Controller
import traceback
import numpy as np

class CerebroSoftMax():

    def __init__(self, pesosH=None, pesosO=None, biasH=None, biasO=None):
        self.log = LogController()
        self.log.logInfo("\n\n\n\n\n\n\n\n\n\n\n\n\n\nCérebro Ligado!", True)  
        
        self.pesosH = pesosH
        self.pesosO = pesosO
        self.biasH = biasH
        self.biasO = biasO

        # if isinstance(pesos, np.ndarray):
        #     self.pesos = pesos
            
        # else:
        #     np.random.seed(1)
        #     self.pesos = 2 * np.random.random((pesos, 1)) - 1
            
        #     self.log.logInfo('Pesos aleatórios:')
        #     self.log.logInfo(str(self.pesos), True)

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

    
    def training(self, inputTraining , outputTraining, iterations):      
       
        a = inputTraining[0]
        b= inputTraining[1]
        c = inputTraining[2]

        inputsV = np.vstack([a,b,c])        

        labels = np.array([0]*1 + [1]*1 + [2]*1)

        labelsResult = np.zeros((3, 3))

        for i in range(3):
            labelsResult[i, labels[i]] = 1
        
        print(labelsResult)
        sys.exit()
        attributes = inputsV.shape[1]
        
        hiddenNodes = 4
        outputNodes = 3

        pesosHidden = np.random.rand(attributes,hiddenNodes)
        biasHidden = np.random.randn(hiddenNodes)
        pesosOut = np.random.rand(hiddenNodes,outputNodes)        
        biasOut = np.random.randn(outputNodes)   

        # print(pesosHidden)
        # print(pesosOut)
        # print(biasHidden)
        # print(biasOut)
        # sys.exit()

        learningRate = 10e-4

        errorCost = []

        for epoch in range(20000):  
            hiddenSum = np.dot(inputsV, pesosHidden) + biasHidden        
            hiddenActv = self.sigmoid(hiddenSum)
            
            outSum = np.dot(hiddenActv, pesosOut) + biasOut        
            outActv = self.softmax(outSum)      
                    
            diffOutActv = outActv - labelsResult
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
                loss = np.sum(-labelsResult * np.log(outActv))
                print('Valor da função de custo de erro: ', loss)
                errorCost.append(loss)

            self.pesosH = pesosHidden
            self.pesosO = pesosOut
            self.biasH = biasHidden
            self.biasO = biasOut

            print(outActv)

    def initProcesso(self, inputTraining, outputTraining):
        self.training(inputTraining, outputTraining, 10000)

        # self.log.logInfo('Pesos após treinamento:')
        # self.log.logInfo(str(self.pesos), True)

        Controller().datasetToCsv(self.pesosH, "pesosH")
        Controller().datasetToCsv(self.pesosO, "pesosO")
        Controller().datasetToCsv(self.biasH, "biasH")
        Controller().datasetToCsv(self.biasO, "biasO")

        print("Ok'")