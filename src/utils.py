from skimage.transform import iradon
from sklearn.metrics import mean_squared_error as mse
import numpy as np
    



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