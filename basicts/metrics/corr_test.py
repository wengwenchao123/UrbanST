import numpy as np
import torch
def node_pcc(prediction, target, null_val: float = np.nan):
    if len(prediction.shape)  == 3:
        node = prediction.shape[1]
        prediction=prediction.transpose(1, 2).reshape(-1,node)
        target=target.transpose(1, 2).reshape(-1,node)
    elif len(prediction.shape)  == 4:
        node = prediction.shape[2]
        prediction = prediction.transpose(2, 3).reshape(-1, node)
        target = target.transpose(2, 3).reshape(-1, node)
    sigma_x = prediction.std(dim=0,unbiased=False)
    sigma_y = target.std(dim=0,unbiased=False)
    mean_x = prediction.mean(dim=0)
    mean_y = target.mean(dim=0)
    cor = ((prediction - mean_x) * (target - mean_y)).mean(0) / (sigma_x * sigma_y+0.00000001)
    return cor.mean()


import numpy as np
import torch


