import numpy as np
import torch

def MAE_torch(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    # if null_val != None:
    #     mask = torch.gt(target, null_val)
    #     prediction = torch.masked_select(prediction, mask)
    #     target = torch.masked_select(target, mask)
    return torch.mean(torch.abs(target-prediction))