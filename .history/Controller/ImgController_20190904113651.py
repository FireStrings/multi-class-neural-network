from PIL import Image
import numpy as np
import sys
import random

class ImgController():
    
    def __init__(self):
        pass
    
    def toDataset(self, path):
        img = Image.open(path)

        x = img.size[0]
        y = img.size[1]

        n = np.empty([0])      

        for ix in range(x):
            for iy in range(y):     
                print(self.sigmoid(np.sum(img.getpixel((ix,iy)))))
                a = np.array(img.getpixel((ix,iy))[0:3])
                result = self.getBlackPercent(a)               
                
                n = np.append(n, round(result,1))        
        
        return n

    def toDatasetSoftmax(self, path):
        
        img = Image.open(path)

        x = img.size[0]
        y = img.size[1]

        n = np.empty([0])      
        n2 = np.empty([0])      
        i = 0
        for ix in range(x):
            for iy in range(y):    
                a = self.norm(np.sum(img.getpixel((ix,iy))))        
                n = np.append(n, a)        
                i+=1
     
        return n
    
    
    def getBlackPercent(self, arrayRGB):
        l = round(2**(7.65 - np.sum(arrayRGB)/100))
        return l/200

    def getBlackPercentSoftMax(self, arrayRGB):
        l = round(2**(7.65 - np.sum(arrayRGB)/100), 4)-1    
        return l/200

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def softmax(self, A):
        expA = np.exp(A)
        return expA / expA.sum(axis=0, keepdims=True)

    def norm(self, x):        
    #    c = np.interp(x, (0, 765), (-1, 0.25))
       c = np.interp(x, (0, 765), (-0.1, 0.01))

       return c

    def norm3(self, x):        
    #    c = np.interp(x, (0, 765), (-1, 0.25))
       c = np.interp(x, (-10, 10), (-1, 1))

       return c

    def norm2(self, x):
        y = ((x - 3) / 3 - 3)/1000
        return y