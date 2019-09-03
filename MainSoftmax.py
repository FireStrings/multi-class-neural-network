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
            imgName = img.split("Data/DataTraining/Img/")[1]

            log.logInfo("Imagem " + imgName + " convertida para dataset")
            
            Controller().datasetToCsv(imgDataset, 'Data/DataTraining/DataSet/'+imgName+'.csv')

            log.logInfo("Dataset " + imgName + " salvo")
    
            
        log.logInfo("Imagens convertidas em Dataset!\n")

    def training(self, loadPesos=False):
        listCsv = c.getListFiles("csv")
       
        listInputs = []

        for _csv in listCsv:   
            listInputs.append(Controller().loadDataset(_csv))
        
        inputTraining = np.array(listInputs)
        
        outputTraining = np.array(Controller().loadDataset('Data/DataTraining/DataSet/labelsResult.csv', False))      
        
        if loadPesos:
            pesosH = np.array(Controller().loadDataset('Data/DataTraining/DataSet/pesosH.csv', False))
            pesosO = np.array(Controller().loadDataset('Data/DataTraining/DataSet/pesosO.csv', False))
            biasH = np.array(Controller().loadDataset('Data/DataTraining/DataSet/biasH.csv'))
            biasO = np.array(Controller().loadDataset('Data/DataTraining/DataSet/biasO.csv'))

            CerebroSoftMax().initProcesso(inputTraining, outputTraining, [pesosH, pesosO, biasH, biasO])

        else:
            CerebroSoftMax().initProcesso(inputTraining, outputTraining)

    def predict(self, pathToNewInput):      
        
        _input = np.array([Controller().loadDataset(pathToNewInput)])  

        pesosH = np.array(Controller().loadDataset('Data/DataTraining/DataSet/pesosH.csv', False))
        pesosO = np.array(Controller().loadDataset('Data/DataTraining/DataSet/pesosO.csv', False))
        biasH = np.array(Controller().loadDataset('Data/DataTraining/DataSet/biasH.csv'))
        biasO = np.array(Controller().loadDataset('Data/DataTraining/DataSet/biasO.csv'))

        
        result = CerebroSoftMax().test(_input, pesosH, pesosO, biasH, biasO)

        for i in result:
            for j in range(0,10):
                v = "{:.16f}".format(float(str(i[j])))[0:7]
                v = float(v)*100

                print(str(v) + "% de certeza que é um " + str(j) + "\n")


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

        imgDataset = ImgController().toDatasetSoftmax("Data/DataTest/Img/"+newInput)
        
        Controller().datasetToCsv(imgDataset, 'Data/DataTest/DataSet/'+newInput+'.csv')

        main.predict("Data/DataTest/DataSet/"+newInput + ".csv")



_main()