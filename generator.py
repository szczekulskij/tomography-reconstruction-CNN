import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

from random import randint
from random import choice

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import KMeans
from skimage.transform import radon

import cv2 
import imutils


def single_square_data_generator(img_size = 64, # size of the img
                                 padding = 10, # the number of MINIMUM of black spaces around or square - so that it doesnt get cut during rotation
                                 own_sinogram_fnc = False,
                                     ) -> " (sinogram ,image) tuple " :


    img = np.zeros((img_size,img_size)) # create an empty image
    square_size = randint(3+padding,img_size-3-padding) # generate a random size of a square side
    x = randint(1,64-square_size-padding) # left starting point
    y = randint(1,64-square_size-padding) # up starting point
    
    # We got starting point and size of square, now, draw it ! :
    for i in range(square_size) : 
        for j in range(square_size) : 
            img[x+i,y+j] = 1 
            
    
    #Generate sinogram, either with the library or from own method :
    if own_sinogram_fnc :
        sinogram = sinogram_fnc(img)
    else : 
        sinogram = radon(img) 
        
    return (sinogram, img)


def square_data_generator(n : "number of data to be generated",
                          img_size : "size of the square's side",
                          ) -> " (x,y) where x - list of sinograms, y - list of corresponding images" :

    x, y = [], []    
    for _ in range(n):
        sinogram, img = single_square_data_generator(img_size)
        x.append(sinogram)
        y.append(img)
        
    x = np.array(x)
    y = np.array(y)
    
    #### To make it fit the NN we need to transform these to tensors, change their dimensions, and cast into double [float32]
    x = torch.from_numpy(x)
    x = x.view(-1,1,64,180)
    x = x.to(dtype=torch.float32)
    
    return (x,y)



# My own sinogram function : 
def sinogram_fnc(img):
    side_size = img.shape[1]
    
    sinogram = np.zeros((side_size,180)) 
    for i in range(180):
        sinogram[: ,i] = np.sum(img, axis = 0)
        img = imutils.rotate(img,1)
    
    img = imutils.rotate(img,180)
    return sinogram




def visualize(img) -> "draws the image" :
    plt.imshow(img, cmap=cm.gray)