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



def get_next_batch_from_disk_RGB_Gaussian(images_list, batch_size, crop_size=64):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        
        if img is not None:
            heightM, widthM = img.shape[:2];

            x = random.random() #[0- 1]
            y = random.random() #[0- 1]

            heightCropStart = int(x * (heightM - crop_size))
            widthCropStart = int(y * (widthM - crop_size))

            cropped = img[heightCropStart:heightCropStart+crop_size, widthCropStart:widthCropStart+crop_size]
            
            cropped_g = cv2.GaussianBlur(cropped, (3, 3), 0)

            gray = cv2.cvtColor(cropped_g, cv2.COLOR_BGR2GRAY)
            gray = (gray * 1./255).astype("float32");
            img = cv2.cvtColor(cropped_g, cv2.COLOR_BGR2RGB)
            img = (img * 1./255).astype("float32");

            gray = gray[...][..., np.newaxis]
            X.append(gray)

            Y.append(img)
        else:
            random.shuffle(images_list)
            imgs.append(load_image(images_list[42]))
        
         
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y


def get_next_batch_from_disk_RGB(images_list, batch_size, crop_size=64):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        
        if img is not None:
            heightM, widthM = img.shape[:2];

            x = random.random() #[0- 1]
            y = random.random() #[0- 1]

            heightCropStart = int(x * (heightM - crop_size))
            widthCropStart = int(y * (widthM - crop_size))

            cropped = img[heightCropStart:heightCropStart+crop_size, widthCropStart:widthCropStart+crop_size]

            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = (gray * 1./255).astype("float32");
            img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            img = (img * 1./255).astype("float32");

            gray = gray[...][..., np.newaxis]
            X.append(gray)

            Y.append(img)
        else:
            random.shuffle(images_list)
            imgs.append(load_image(images_list[42]))
        
         
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X, Y


def get_next_batch_from_disk_RGB_Nocrop(images_list, batch_size):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        
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


def get_next_batch_from_disk_Nocrop_HSV(images_list, batch_size):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        #convert image to 32 bit 
        img2 = (img * 1./255).astype("float32");
        hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV);
        
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        #To comply with tensorflow expected size
        v = v[...][..., np.newaxis]
        X.append(v)
        
        Y.append(np.dstack((h,s)))
         
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    return X, Y


'''
Use like:
a = get_next_batch_from_disk_RGB_Nocrop_HSV(images_list, 1)

h = a[1][0][..., 0]
s = a[1][0][..., 1]
v = a[0][0][..., 0]

hsv = np.dstack((h,s,v))
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
plt.imshow(rgb)
'''

def get_next_batch_from_disk_HSV(images_list, batch_size, crop_size=64):
    random.shuffle(images_list)
    imgs = [load_image(images_list[i]) for i in range(batch_size)]

    X = []
    Y = []

    for img in imgs:
        heightM, widthM = img.shape[:2];
        x = random.random() #[0- 1]
        y = random.random() #[0- 1]
           
        heightCropStart = int(x * (heightM - crop_size))
        widthCropStart = int(y * (widthM - crop_size))

        cropped = img[heightCropStart:heightCropStart+crop_size, widthCropStart:widthCropStart+crop_size]

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV);
        
        h = (hsv[..., 0] * 1./179).astype("float32");
        s = (hsv[..., 1] * 1./255).astype("float32");
        v = (hsv[..., 2] * 1./255).astype("float32");

        #To comply with tensorflow expected size
        v = v[...][..., np.newaxis]
        X.append(v)
        
        Y.append(np.dstack((h,s)))
        
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