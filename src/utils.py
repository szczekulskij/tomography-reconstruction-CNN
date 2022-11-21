from skimage.transform import iradon
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import math



## Error calculation:
def mse_error(img_list1,img_list2):
    assert np.array(img_list1).ndim in [2,3]
    assert np.array(img_list2).ndim in [2,3]
    assert np.array(img_list1).ndim == np.array(img_list2).ndim

    if np.array(img_list1).ndim == 2 & np.array(img_list2).ndim == 2:
        img_list1 = [img_list1]
        img_list2 = [img_list2]

    error = 0
    n = len(img_list1)
    for img1, img2 in zip(img_list1,img_list2):
        error+=mse(img1, img2)
    
    return round(error/n,2)

def reconstruct(sinograms_list):
    assert np.array(sinograms_list).ndim in [2,3]
    if np.array(sinograms_list).ndim == 2 :
        return iradon(sinograms_list)

    reconstruct_list = []
    for sinogram in sinograms_list:
        reconstruct_list.append(iradon(sinogram))
    return reconstruct_list


def get_non_binary_angles(binary_angles):
    '''
    Take in list of length 180
    return indexes at which value=1
    '''
    angles_list = []
    for index,angle in enumerate(binary_angles):
        if angle == 1 :
            angles_list.append(index)
    return angles_list


def choose_top_angles(prediction):
    prediction_ = prediction.copy()
    assert prediction_.shape == (180,)
    CLOSENESS_THRESHOLD = 180/5 # Polygons have up to 
    i = -1
    indices_2dlist = []
    while i <179:
        i+=1
        val = prediction_[i]
        curr_1d_list = []

        while val > 0.9 and i < 179:
            curr_1d_list.append(i)
            i+=1
            val = prediction_[i]
        if curr_1d_list: # if not empty
            indices_2dlist.append(curr_1d_list)

    # print(indices_2dlist)
    output_angles = []
    for list in indices_2dlist:
        if len(list) < 10:
            continue
        
        # find middle part of indices_2dlist
        mid = int(len(list)/2)
        output_angles.append(list[mid])
    # print(output_angles)
    return output_angles

def get_split(X, y, split_pct = 70):
    assert type(X) in [np.ndarray, list] and type(y) in [np.ndarray, list]
    assert len(X) == len(y)
    assert split_pct > 50 and split_pct < 90

    if type(X) != np.array:
        X = np.array(X)
    if type(y) != np.array:
        y = np.array(y)

    index = int(split_pct/100 * len(X))
    X_train = X[:index]
    X_test = X[index:]

    y_train = y[:index]
    y_test = y[index:]

    return X_train, X_test, y_train, y_test

def transform_angles_data(angles_list):
    def sigmoid(x):
        return 2 / (1 + math.exp(-x)) - 1
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
        angles_ = []
        for iter, angle in enumerate(angles):
            angles_.append(round(sigmoid(angle),3))
        output_angles.append(angles_)

    return output_angles


def transform_angles_data_old(angles_list, high_nr = 100, neighbouring_angles = 10, drop = 10):
    assert np.array(angles_list).ndim == 2
    angles_list_copy = np.array(angles_list).copy()
    # print("neighbouring_angles:", neighbouring_angles)
    output_angles_list = []
    for angles in angles_list_copy:
        # print("angles:", angles)
        angles_indices = get_non_binary_angles(angles)
        # print("angles_indices:", angles_indices)
        for indice in angles_indices:
            # print("1st iterated")
            angles[indice] = high_nr
            for i in range(neighbouring_angles):
                # print("iterated")
                angles[indice-i] = high_nr - i * drop
                angles[(indice+i)%180] = high_nr - i * drop
        output_angles_list.append(angles)
    return np.array(output_angles_list)