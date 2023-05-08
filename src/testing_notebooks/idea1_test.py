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


model = tf.keras.models.load_model('../ai_training_notebooks/saved_model/128_limited_to_150_polygon')

predictions = model.predict(X_test)


i = 10
for (truth, prediction) in zip(y_test, predictions):
    real_angles = choose_top_angles(truth)
    predicted_angles = choose_top_angles(prediction)
    print("truth:", real_angles)
    print("prediction:", predicted_angles)
    print()
    i+=1
    if i == 10 :
        break

