# adding src to the system path
import sys  
sys.path.insert(0, '/Users/szczekulskij/side_projects/tomography-reconstruction-CNN')

import numpy as np
import pandas as pd
from random import randint
import math
from PIL import Image, ImageDraw
from PIL import ImagePath 
import pickle
# from imutils import rotate
import matplotlib.pyplot as plt
from skimage.transform import radon

from src.mock_dataset_generator import create_dataset #, generate_polygon, generate_polygon_subroutine
from src.utils import mse_error, reconstruct, get_split, choose_top_angles, get_non_binary_angles, transform_angles_data

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt



images_list, sinograms_list, angles_list = create_dataset(25000, single_side_size = 6, img_size = 128)

# To load instead of creating - comment out previous line and uncomment next two!
# with open("pickle_file_128", 'rb') as file:
#     images_list, sinograms_list, angles_list = pickle.load(file)

with open("pickle_file_hexagon_128", 'wb') as file:
    pickle.dump([images_list, sinograms_list, angles_list], file)


reconstruct_list = reconstruct(sinograms_list)




print("average mse error for 50 hexagon-shaped pics:", mse_error(images_list, reconstruct_list))

def cut_last_40_angles_from_sinogram_list(sinograms_list):
    assert type(sinograms_list) == list
    output = []
    for sinogram in sinograms_list:
        cut_sinogram = sinogram[:, :-40]
        output.append(cut_sinogram)
    return output


def cut_last_40_angles_from_angles_list(angles_list):
    assert type(angles_list) == list
    output = []
    for angle in angles_list: 
        # Angle is a a liast of 180 angles
        new_angle = angle[:-40]
        output.append(new_angle)
    return output

def create_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (5, 5), activation='relu', kernel_initializer='random_normal',),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='random_normal',),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='random_normal',),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='random_normal',),
        layers.MaxPooling2D((2, 2)),

        # layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='random_normal',),
        # layers.MaxPooling2D((2, 2)), # Might increase performance because it decreases parameters (which is why CNN was created for in the first place)
        # Sams model has 85% of params in dense layer
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(1000, activation="relu", kernel_initializer='random_normal',),
        tf.keras.layers.Dense(500, activation="relu", kernel_initializer='random_normal',),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(140, activation="sigmoid", kernel_initializer='random_normal',),
    ])
    return model

print("Creating model!")
model = create_model()
model.build((None, 128, 140, 1))
print("Created model!")


print("Comping model")
model.compile(optimizer=Adam(),
              loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=['accuracy'])

checkpoint_filepath = '/tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

print("Compiled model")

def transform_angles_data(angles_list):
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    assert np.array(angles_list).ndim == 2
    angles_list_copy = np.array(angles_list).copy()
    output_angles = []
    
    for angles in angles_list_copy:
        angles_indices = get_non_binary_angles(angles)
        nr_angles = len(angles_indices)
        # Change from -10 to 10  (therefore value 20)
        CONST = int(180/2/nr_angles)
        for angle in angles_indices:
            for i in range(0,CONST + 1):
                if i == 0 : iter = 10
                else : iter = round(10 - (20 * i / CONST),2)
                angles[(angle + i)%180] = iter
                angles[angle - i] = iter
        # Transform values from -10 to 10 into more smooth line using sigmoid function
        # print("angles:", angles)
        angles_ = []
        for iter, angle in enumerate(angles):
            angles_.append(round(sigmoid(angle),4))
        output_angles.append(angles_)
        # print(angles_)
        # break
        
    return output_angles
    
print("Transforming data starts now")
transformed = transform_angles_data(angles_list)
sinograms_list_cut = cut_last_40_angles_from_sinogram_list(sinograms_list)
transformed_angles_list_cut = cut_last_40_angles_from_angles_list(transformed)
X_train, X_test, y_train, y_test = get_split(sinograms_list_cut, transformed_angles_list_cut)
print("Transforming data finished!")

print(np.array(X_train).shape)
print(np.array(y_train).shape)

print("About to start the training!")
history = model.fit(X_train,y_train, batch_size=200, epochs=300,validation_data=(X_test,y_test))
model.save('saved_models/128_limited_to_150_hexagon')