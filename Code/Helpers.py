import cv2
import numpy as np
import os
import glob
import pickle

import matplotlib.image as mpimg
from skimage import color

import random

def load_image(infilename, convert_lab=False):
    data = mpimg.imread(infilename)
    if convert_lab:
        data = color.rgb2lab(data)

    return data

def separate_imgs(img):
    return np.split(img, 2, axis=1)


def get_files(path, ext):
    #"*.mp4"
    #"..\Data\"
    return glob.glob(os.path.join(path, ext)) 


def get_next_batch_from_disk(images_list, batch_size):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i], convert_lab=False) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        y, x =  separate_imgs(img)

        x = cv2.normalize(x, x, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        y = cv2.normalize(y, y, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        X.append(x)
        Y.append(y)

    X = np.asarray(X)
    Y = np.asarray(Y)
    
    #print(type(X[1][1][1][1]))
    
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