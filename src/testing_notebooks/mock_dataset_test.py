# adding src to the system path
import sys  
sys.path.insert(0, '/Users/szczekulskij/side_projects/research_projects/tomography-reconstruction-CNN')

import numpy as np
import pandas as pd
from random import randint
import math
import pickle
# from imutils import rotate
import matplotlib.pyplot as plt
from skimage.transform import radon

from src.mock_dataset_generator import create_dataset #, generate_polygon, generate_polygon_subroutine
from src.new_utils import find_best_angles_in_real, find_best_angles_in_prediction, cut_last_40_angles_from_sinogram_list, cut_last_40_angles_from_angles_list

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt


model = tf.keras.models.load_model('src/ai_training_notebooks/saved_models/128_limited_to_140_hexagon')
images_list, sinograms_list, angles_list = create_dataset(10, single_side_size = 6, img_size = 128)

sinograms_list_cut = cut_last_40_angles_from_sinogram_list(sinograms_list)
angles_list_cut = cut_last_40_angles_from_angles_list(angles_list)

X_test = np.array(sinograms_list_cut)
y_test = angles_list_cut 

predictions = model.predict(X_test)
predictions = np.round(np.array(predictions),2)

for (truth, prediction) in zip(y_test, predictions):
    real_angles = find_best_angles_in_real(truth)
    predicted_angles = find_best_angles_in_prediction(prediction)
    print("truth:", real_angles)
    print("prediction:", predicted_angles)
    print()