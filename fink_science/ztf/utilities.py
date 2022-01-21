import numpy as np


def fix_nans(arr):
    if len(arr.shape)>1:
        raise Exception("Only 1D arrays are supported.")
    
    if np.isnan(arr[0]):
        for elem in arr:
            if not np.isnan(elem):
                arr[0] = elem
                break
        else:
            raise ValueError("nans only!")

    last_value = arr[0]
    for idx, elem in enumerate(arr):
        if not np.isnan(elem):
            last_value = elem
        else:
            arr[idx] = last_value
