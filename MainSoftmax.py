import os
import sys

from Controller.Controller import Controller 
from Cerebro.CerebroSoftMax import CerebroSoftMax 
from Controller.ImgController import ImgController 
from Controller.LogController import LogController 

import numpy as np

log = LogController()
c = Controller()

class Main():
    def __init__(self):
        super()
    
    def createDataSets(self):
        listImg = c.getListFiles("jpg")

        for img in listImg:    
            imgDataset = ImgController().toDatasetSoftmax(img)
            imgName = img.split("Data/DataTraining/")[1]

            log.logInfo("Imagem " + imgName + " convertida para dataset")
            
            Controller().datasetToCsv(imgDataset, imgName)

            log.logInfo("Dataset " + imgName + " salvo")
    
            
        log.logInfo("Imagens convertidas em Dataset!\n")

    def training(self):
        listCsv = c.getListFiles("csv")

        listInputs = []

        for _csv in listCsv:   
            listInputs.append(Controller().loadDataset(_csv))

        inputTraining = np.array(listInputs)
        
        outputTraining = np.array(Controller().loadDataset('Data/DataTraining/labelsResult.csv', False))
        
        CerebroSoftMax(len(listInputs[0])).initProcesso(inputTraining, outputTraining)

    def predict(self, pathToNewInput):      
        
        _input = np.array([Controller().loadDataset(pathToNewInput)])  

        pesosH = np.array(Controller().loadDataset('Data/pesosH.csv', False))
        pesosO = np.array(Controller().loadDataset('Data/pesosO.csv', False))
        biasH = np.array(Controller().loadDataset('Data/biasH.csv'))
        biasO = np.array(Controller().loadDataset('Data/biasO.csv'))

        
        result = CerebroSoftMax().test(_input, pesosH, pesosO, biasH, biasO)
        print(result)

        precision = round(result[0], 3)*100
        log.logInfo("...." + str(precision) + "% de certeza que é um numero esperado...")


main = Main()
def _main():

    option = sys.argv[1]    

    log.logInfo("Processo Iniciado com o argumento " + option + "\n")

    if option == '0':

       main.createDataSets()

    elif option == '1':
        
        main.training()

    elif option == '2':
        
        newInput = sys.argv[2]

        imgDataset = ImgController().toDatasetSoftmax("Data/DataTraining/"+newInput)
        
        Controller().datasetToCsv(imgDataset, newInput)

        main.predict("Data/"+newInput + ".csv")



_main()
    
    
