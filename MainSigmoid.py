import os
import sys

from Controller.Controller import Controller 
from Cerebro.CerebroSigmoidal import CerebroSigmoidal 
from Controller.ImgController import ImgController 
from Controller.LogController import LogController 

import numpy as np

log = LogController()
c = Controller()

class Main():
    def __init__(self):
        super()    

    def training(self, listInputs):
        inputTraining = np.array(listInputs)
        
        outputTraining = np.array([[0,0,0,1]]).T

        CerebroSigmoidal(len(listInputs[0])).initProcesso(inputTraining, outputTraining)

    def predict(self, pathToNewInput):      

        _input = Controller().loadDataset(pathToNewInput)        
        pesos = np.array(Controller().loadDataset('Data/pesos_atualizados.csv'))

        _input = np.array([_input])
        result = CerebroSigmoidal(pesos).synapse(_input)

        precision = round(result[0], 3)*100
        log.logInfo("...." + str(precision) + "% de certeza que é um numero esperado...")



def main():
    option = sys.argv[1]

    log.logInfo("Processo Iniciado com o argumento " + option + "\n")

    if option == '0':

        listImg = c.getListFiles("jpg")

        for img in listImg:    
            imgDataset = ImgController().toDataset(img)
            imgName = img.split("Data/")[1]

            log.logInfo("Imagem " + imgName + " convertida para dataset")
            
            Controller().datasetToCsv(imgDataset, imgName)

            log.logInfo("Dataset " + imgName + " salvo")
    
            
        log.logInfo("Imagens convertidas em Dataset!\n")

    elif option == '1':
        
        listCsv = c.getListFiles("csv")

        listInputs = []

        for _csv in listCsv:    
            listInputs.append(Controller().loadDataset(_csv))
        
        main.training(listInputs)

    elif option == '2':
        newInput = sys.argv[2]

        imgDataset = ImgController().toDataset("Data/"+newInput)
        
        Controller().datasetToCsv(imgDataset, newInput)

        main.predict("Data/"+newInput + ".csv")


def tests():
    print(CerebroSigmoidal(2).softmax([0.8, 0.2, 0.5]))

tests()
    
    

