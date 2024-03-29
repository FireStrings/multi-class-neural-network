import csv
import random
import sys
import os
import numpy as np

class Controller():
    def __init__(self):
        pass
    
    def load_dataset_old(self, path):      
    
        dataset = []  

        with open(path) as _file:
            data = csv.reader(_file, delimiter=',')
            for line in data:
                line = [float(elemento) for elemento in line]
                dataset.append(line)

        return dataset

    
    def loadDataset(self, path, onlyOne=True):      
    
        dataset = []  

        with open(path) as _file:
            data = csv.reader(_file, delimiter=';')
            for line in data:
                line = [float(elemento) for elemento in line]
                if onlyOne:
                    dataset.append(line[0])
                else:
                    dataset.append(line)

        return dataset

    
    def datasetToCsv(self, dados, path):
        with open(path, 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')
                
            for r in dados:
                if isinstance(r, np.float64):
                    spamwriter.writerow([r])
                else:
                    spamwriter.writerow(r)
    
    def getListFiles(self, path):
        
        result = os.popen("ls " + path).read()
            

        listFiles = result.split('\n')

        del listFiles[-1]

        return listFiles



    
    
