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


def single_square_data_generator(img_size = 91, # size of the img
                                 min_square_size = 32,
                                 max_square_size = 50, 
                                 own_sinogram_fnc = False,
                                 rotation = False,
                                     ):
    
    
    padding = (img_size - max_square_size)//2


    img = np.zeros((img_size,img_size)) # create an empty image
    square_size = randint(min_square_size, max_square_size) # generate a random size of a square side
    x = randint(padding//1.5,padding) # left starting point
    y = randint(padding//1.5,padding) # up starting point
    
    # We got starting point and size of square, now, draw it ! :
    for i in range(square_size) : 
        for j in range(square_size) : 
            img[x+i,y+j] = 1 
            
            
    #Generate the best angles : 
    angles = np.array([0, 90])
            
            
    # If we want our data to be rotated :
    if rotation : 
        rotation_angle = randint(0,90)
        img = imutils.rotate(img, rotation_angle)
        angles+= rotation_angle
            
    
    #Generate sinogram, either with the library or from own method :
    if own_sinogram_fnc :
        sinogram = sinogram_fnc(img)
    else : 
        sinogram = radon(img) 
        
        
    
        
    return (sinogram, img, angles)




def square_data_generator(n = 1,  #number of data to be generated
                          img_size = 91,  # size of the image 
                          min_square_size = 32,
                          max_square_size = 50, 
                          own_sinogram_fnc = False,  # Which sinogram generator to use ? 
                          rotation = False, # Rotate the image ? 
                          ) -> " (sinogram, image, [angles] ) " :

    x, y, z = [], [], []    
    for _ in range(n):
        sinogram, img, angles = single_square_data_generator(img_size, min_square_size, max_square_size, own_sinogram_fnc, rotation)
        x.append(sinogram)
        y.append(img)
        z.append(angles)
        
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    #### To make it fit the NN we need to transform these to tensors, change their dimensions, and cast into double [float32]
    x = torch.from_numpy(x)
    x = x.view(-1,1,91,180)
    x = x.to(dtype=torch.float32)
    
    return (x, y, z)



# My own sinogram function : 
def sinogram_fnc(img):
    side_size = img.shape[1]
    
    sinogram = np.zeros((side_size,180)) 
    for i in range(180):
        sinogram[: ,i] = np.sum(img, axis = 0)
        img = imutils.rotate(img,1)
    
    img = imutils.rotate(img,180)
    return sinogram




def visualize(img) :
    plt.imshow(img, cmap=cm.gray)
