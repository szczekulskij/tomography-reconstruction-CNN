from skimage.transform import iradon
from sklearn.metrics import mean_squared_error as mse
import numpy as np
import math


def find_best_angles_in_prediction(
                                   angles_list, 
                                   inclusion_threshold = 0.90, 
                                   difference_threshold = 3, 
                                   extra_tuning = False
                                   ):
    '''
    input: 
            [0.12 0.12 0.27 0.5  0.5  0.5  0.73 0.88 0.89 0.95 0.98 0.98 0.99 1.
            1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   1.   0.99 0.98
            0.98 0.95 0.88 0.88 0.73 0.5  0.5  0.5  0.27 0.12 0.11 0.05 0.02 0.02
            0.01 0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]

    output: [5, 65] - list of indices

    Algorithm:
    1. Find all ones
    2. Save their indices
    3. Group them together
    4. Find the middle one for each group
    5. EXTRA - extrapolate values/even out midds so they are all within same range difference
    '''
    # 1. Find all ones & 2. Save their indices
    indices_of_ones_list = []
    for index, value in enumerate(angles_list):
        if value >= inclusion_threshold : indices_of_ones_list.append(index)

    # 3.Group them together into subgroups
    list_2d = []
    subgroup_list = []
    for i in range(len(indices_of_ones_list) - 1):
        index = indices_of_ones_list[i]
        next_index = indices_of_ones_list[i+1]
        if next_index - index <= difference_threshold:
            subgroup_list.append(index)
        else :
            list_2d.append(subgroup_list)
            subgroup_list = []

        # Always add the last sublist in as well
        if i == len(indices_of_ones_list) - 2:
            list_2d.append(subgroup_list)
    
    # 4. Find the middle one for each group
    final_output = []
    for sublist in list_2d:
        middle_point = len(sublist) // 2 
        final_output.append(sublist[middle_point])

    # 5. Add extra tuning to the algorithm
    if extra_tuning:
        pass

    return final_output

def find_best_angles_in_real(angles_list):
    '''
    real dataset is simple - it's just like [0,0,0,0,0,1,0,0,]
    input: [0,0,0,0,0,1,0,0,0,0,1]
    output: [5, 9] - list of indices
    '''
    return [index for index, value  in enumerate(angles_list) if value == 1]