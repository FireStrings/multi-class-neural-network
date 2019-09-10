
from PIL import Image, ImageOps
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

        if x != 28 or y != 28:
            img = self.resizeImg(img, True)
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

    def resizeImg(self, img, save=False):
        # wpercent = (28 / float(img.size[0]))
        # hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((28, 28), Image.ANTIALIAS)

        if save:
            img.save('Data/DataTest/Img/2_resized.jpg')
        return img

    def cropImg(self, img):
        import time
        img = Image.open('../Data/DataTest/Img/c1.jpg')
        x = img.size[0]
        y = img.size[1]

        greyscale_image = img.convert('L')
        greyscale_image.save('../Data/DataTest/Img/cb.jpg')


        # print(x)
        # print(y)
        for ix in range(x):
            for iy in range(y):
                print(img.getpixel((ix,iy)))
                print(iy)
                if iy != 0:   
                    if iy % 59 == 0:
                        print("Coluna " + str(ix))
                        print("###########################################")
                        print()
                        if ix == 2:
                            sys.exit()

        # for i in range()
        # imgCropped = img.crop((50,10,200,40)) #todo o  numero
        # imgCropped = img.crop((50,10,80,40)) #primeiro
        # imgCropped = img.crop((80,10,100,40)) #segundo
        # imgCropped = img.crop((100,10,125,40)) #terceiro
        # imgCropped = img.crop((125,10,150,40)) # quarto
        # imgCropped = img.crop((150,10,175,40)) #quinto
        # imgCropped = img.crop((175,10,200,40)) #sexta

        # imgCropped = img.crop((40,10,70,45)) #primeiro
        # imgCropped.save('../Data/DataTest/Img/cropped_captcha.jpg')

    
    def toNegative(self, pathOld, pathNew):
        img = Image.open(pathOld)
        im_invert = ImageOps.invert(img)
        im_invert.save(pathNew, quality=100)



ImgController().cropImg(None)

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)

# change_contrast(Image.open('../Data/DataTest/Img/c1.jpg'), 256).save('../Data/DataTest/Img/cb.jpg')