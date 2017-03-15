import cv2
import numpy as np
import os
import glob
import pickle

import matplotlib.image as mpimg
from skimage import color

import random

def load_image(infilename):
    #return mpimg.imread(infilename)
    return cv2.imread(infilename)
    
def separate_imgs(img):
    return np.split(img, 2, axis=1)


def get_files(path, ext):
    #"*.mp4"
    #"..\Data\"
    return glob.glob(os.path.join(path, ext)) 


def get_next_batch_from_disk_RGB(images_list, batch_size):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        #y is in 0 - 1
       
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray / 255
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        
        gray = gray[...][..., np.newaxis]
        X.append(gray)
        
        Y.append(img)
         
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y

def get_next_batch_from_disk2(images_list, batch_size):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        #y is in 0 - 1
        img_hsv = img[...,0:3] / 255
        
        y = img_hsv[..., 0]
        u = img_hsv[..., 1]
        v = img_hsv[..., 2]

        #To comply with tensorflow expected size
        y = y[...][..., np.newaxis]
        X.append(y)
        
        Y.append(np.dstack((u,v)))
         
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y


#Standardize X and Y
'''
meanX = np.mean(X, axis=(0, 1, 2))
stdX = np.std(X, axis=(0, 1, 2))

# We save the mean and the variance from the data since we need it later when we make predictions
f = open('meanX.pckl', 'wb')
pickle.dump(meanX, f)
f.close()

f = open('stdX.pckl', 'wb')
pickle.dump(stdX, f)
f.close()
print(meanX)
print(stdX)

#standardize Y
meanY = np.mean(Y, axis=(0, 1, 2))
stdY = np.std(Y, axis=(0, 1, 2))

# We save the mean and the variance from the data since we need it later when we make predictions
f = open('meanY.pckl', 'wb')
pickle.dump(meanY, f)
f.close()

f = open('stdY.pckl', 'wb')
pickle.dump(stdY, f)
f.close()

for i in range(0,3):
    X[...,i] = (X[...,i] - meanX[i]) / stdX[i]
    #Y[...,i] = (Y[...,i] - meanY[i]) / stdY[i]
    
    
    
#Destandardize
asd = X[2]

for i in range(0,3):
    asd[...,i] = (asd[...,i] + meanX[i]) * stdX[i]

'''