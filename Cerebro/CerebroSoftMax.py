import csv
import random
import sys
from Controller.LogController import LogController
from Controller.Controller import Controller
import traceback
import numpy as np

log = LogController.getLogger()

class CerebroSoftMax():

    def __init__(self):
       
        log.info("Cérebro Ligado!")

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

    def predict(self, newInput, pesosH, pesosO, biasH, biasO):        
        
        pesosHidden = pesosH
        pesosOut = pesosO
        biasHidden = biasH            
        biasOut = biasO

        inputsV = np.vstack([newInput]) 

        
        hiddenSum = np.dot(inputsV, pesosHidden) + biasHidden        
        hiddenActv = self.sigmoid(hiddenSum)
        
        outSum = np.dot(hiddenActv, pesosOut) + biasOut        
        outActv = self.softmax(outSum) 

        return outActv

    def activation(self, inputsV):
        hiddenSum = np.dot(inputsV, self.pesosH) + self.biasH      
        hiddenActv = self.sigmoid(hiddenSum)
        
        outSum = np.dot(hiddenActv, self.pesosO) + self.biasO        
        outActv = self.softmax(outSum)  

        return  hiddenSum, hiddenActv, outActv
    
    def errorCostFunction(self, outputTraining, outActv):
        return np.sum(-outputTraining * np.log(outActv))
    
    def backropagation(self, outActv, outputTraining, hiddenActv, hiddenSum, inputsV, learningRate, epoch):
        diffOutActv = outActv - outputTraining
        sumHiddenActvOut = np.dot(hiddenActv.T, diffOutActv)

        sumDiffOutActvPesosOut = np.dot(diffOutActv, self.pesosO.T)            

        sigDer = self.sigmoidDerivate(hiddenSum)
        sumInputsSigDer = np.dot(inputsV.T, sigDer * sumDiffOutActvPesosOut)

        dotSigDer = sumDiffOutActvPesosOut * sigDer

        self.pesosH -= learningRate * sumInputsSigDer
        self.biasH -= learningRate * dotSigDer.sum(axis=0)

        self.pesosO -= learningRate * sumHiddenActvOut
        self.biasO -= learningRate * diffOutActv.sum(axis=0)

        if epoch % 200 == 0:
            loss = self.errorCostFunction(outputTraining, outActv)
            log.info('Valor da função de custo de erro: ' + str(loss))
            print('Valor da função de custo de erro: ' + str(loss), end="\r")

    def training(self, inputTraining , outputTraining, pesos=None):    

        log.info('Inicio do treinamento...')
        print('Inicio do treinamento...')
        inputsV = np.vstack([inputTraining])   

        attributes = inputsV.shape[1]  

        hiddenNodes = 4
        outputNodes = outputTraining.shape[1]

        if pesos == None:
            self.pesosH = np.random.rand(attributes,hiddenNodes)
            self.pesosO = np.random.rand(hiddenNodes,outputNodes)   
            self.biasH = np.random.randn(hiddenNodes)                 
            self.biasO = np.random.randn(outputNodes)   

        else:
            self.pesosH = pesos[0]
            self.pesosO = pesos[1]
            self.biasH = pesos[2]
            self.biasO = pesos[3]

        learningRate = 10e-4        

        for epoch in range(1000000):  
            
            hiddenSum, hiddenActv, outActv = self.activation(inputsV)

            self.backpropagation(outActv, outputTraining, hiddenActv, hiddenSum, inputsV, learningRate, epoch)

        
        self.storeLearning()       

    def printResultActv(self, outActv):
        for i in outActv:
            log.info(self.convertSciToDec(i[0])) 
            log.info(self.convertSciToDec(i[1])) 
            log.info(self.convertSciToDec(i[2])) 
            log.info(self.convertSciToDec(i[3])) 
            log.info(self.convertSciToDec(i[4])) 
            log.info(self.convertSciToDec(i[5])) 
            log.info(self.convertSciToDec(i[6])) 
            log.info(self.convertSciToDec(i[7])) 
            log.info(self.convertSciToDec(i[8])) 
            log.info(self.convertSciToDec(i[9])) 

    def convertSciToDec(self, value):
        return "{:.16f}".format(float(str(value)))

    def storeLearning(self):
        Controller().datasetToCsv(self.pesosH, 'Data/DataTraining/DataSet/pesosH.csv')
        Controller().datasetToCsv(self.pesosO, 'Data/DataTraining/DataSet/pesosO.csv')
        Controller().datasetToCsv(self.biasH, 'Data/DataTraining/DataSet/biasH.csv')
        Controller().datasetToCsv(self.biasO, 'Data/DataTraining/DataSet/biasO.csv')