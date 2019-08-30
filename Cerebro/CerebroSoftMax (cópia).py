import csv
import random
import sys
from Controller.LogController import LogController
from Controller.Controller import Controller
import traceback
import numpy as np

class CerebroSoftMax():

    def __init__(self, pesos):
        self.log = LogController()
        self.log.logInfo("\n\n\n\n\n\n\n\n\n\n\n\n\n\nCérebro Ligado!", True)  

        if isinstance(pesos, np.ndarray):
            self.pesos = pesos
            
        else:
            np.random.seed(1)
            self.pesos = 2 * np.random.random((pesos, 1)) - 1
            
            self.log.logInfo('Pesos aleatórios:')
            self.log.logInfo(str(self.pesos), True)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def sigmoidDerivate(self, x):
        return self.sigmoid(x) *(1-self.sigmoid(x))

    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=1, keepdims=True)
    
    def training(self,inputTraining , outputTraining, iterations):

        # Imgs1 = np.random.randn(1, 10) + np.array([0.9, 0.1, 0.8, 0.3, 0.7, 0.2, 0.9, 0.5, 0.6, 0.1])
        # Imgs2 = np.random.randn(1, 10) + np.array([0.2, 0.4, 0.9, 0.3, 0.8, 0.5, 0.6, 0.4, 0.1, 1])
        # Imgs3 = np.random.randn(1, 10) + np.array([1, 0.3, 0.8, 0.2, 0.4, 0.7, 0.9, 0.1, 0.3, 0.7])


        # Imgs1 = np.array([0.9, -0.1, -0.8, 0.3, 0.7, -0.2, 0.9, -0.5, 0.6, -0.1])
        # Imgs2 = np.array([-0.2, -0.4, 0.9, 0.3, -0.8, 0.5, -0.6, 0.4, -0.1, 1])
        # Imgs3 = np.array([-1, 0.3, -0.8, 0.2, 0.4, -0.7, 0.9, -0.1, 0.3, -0.7])

        # Imgs1 = np.array([ 3,2.83529412, 1.44705882, -2.48235294, -3, -2.78823529, 0.27058824, 3, 2.95294118,  2.78823529,  2.97647059,  2.83529412])

        # Imgs2 = np.array([ -3, -2.95294118, 0.01176471, 2.83529412, 3, 2.97647059, 2.97647059, 3, 2.62352941, -0.22352941, -3, -2.76470588])

        # Imgs3 = np.array([-2.48235294, -3, -3, -2.48235294, 0.48235294, 3, 3, 2.76470588,  2.76470588,  2.97647059,  2.85882353,  2.41176471])
        # print(Imgs1)


        # Imgs1 = [ 1.52795055, -1.56111989,  1.20835292, -0.72835009,  1.71821638, -0.74617371, 2.87230326,  0.76781813, -0.26885896,  1.75875935]
        # Imgs2 = [-0.22725251,  0.27252389, -1.02379932,  1.56776474 , 1.50452187 , 0.42157303, -0.99762084, -0.13970009 , 1.49640531,  2.67072922]
        # Imgs3 = [ 0.65152886 , 0.53770509 , 0.99709455 ,-0.0606303 ,  1.96402263 , 0.440956, 1.32948997, -0.29257894,  1.81560036, -0.2820059 ]

        # Imgs1 = np.random.randn(1, 2) + np.array([0, -3])
        # Imgs2 = np.random.randn(1, 2) + np.array([3, 3])
        # Imgs3 = np.random.randn(1, 2) + np.array([-3, 3])

       
        a = inputTraining[0]
        b= inputTraining[1]
        c = inputTraining[2]
        # a = inputTraining[0][inputTraining[0]!=0][0:134]
        # b= inputTraining[1][inputTraining[1]!=0][0:134]
        # c = inputTraining[2][inputTraining[2]!=0][0:134]

        # a = a[0:15]
        
        # b = b[0:15]
        
        # c = c[0:15]
        # print(a)
        # print()
        # print(b)
        # print()
        # print(c)
        # sys.exit()    
        inputsV = np.vstack([a,b,c])

        # print(inputsV)
        
        # inputsV = np.vstack([Imgs1,Imgs2,Imgs3])
        # print(inputsV)
        # sys.exit()
        # inputsV = np.array([[-1.34776494, -1.52926014],
        #                         [ 3.33722094, 4.00806543],
        #                 [-2.21477308, 2.33513223]
        #                 ])

        # print(inputsV)
        # sys.exit()

        labels = np.array([0]*1 + [1]*1 + [2]*1)

        labelsResult = np.zeros((3, 3))


        for i in range(3):
            labelsResult[i, labels[i]] = 1

        # print(labelsResult)
        # sys.exit()
        attributes = inputsV.shape[1]
        
        hiddenNodes = 4
        outputNodes = 3

        pesosHidden = np.random.rand(attributes,hiddenNodes)
        biasHidden = np.random.randn(hiddenNodes)
        pesosOut = np.random.rand(hiddenNodes,outputNodes)        
        biasOut = np.random.randn(outputNodes)     

        learningRate = 10e-4

        errorCost = []

        for epoch in range(20000):  
            hiddenSum = np.dot(inputsV, pesosHidden) + biasHidden        
            hiddenActv = self.sigmoid(hiddenSum)
            
            outSum = np.dot(hiddenActv, pesosOut) + biasOut        
            outActv = self.softmax(outSum)      
            
            #-----------------------------------------
        
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
            
            print(outActv)

    def initProcesso(self, inputTraining, outputTraining):
        self.training(inputTraining, outputTraining, 10000)

        # self.log.logInfo('Pesos após treinamento:')
        # self.log.logInfo(str(self.pesos), True)

        # Controller().datasetToCsv(self.pesos, "pesos_atualizados")