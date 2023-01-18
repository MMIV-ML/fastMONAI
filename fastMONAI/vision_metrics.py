# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/05_vision_metrics.ipynb.

# %% auto 0
__all__ = ['calculate_dsc', 'calculate_haus', 'binary_dice_score', 'multi_dice_score', 'binary_hausdorff_distance',
           'multi_hausdorff_distance']

# %% ../nbs/05_vision_metrics.ipynb 1
import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance, compute_dice
from .vision_data import pred_to_binary_mask, batch_pred_to_multiclass_mask

# %% ../nbs/05_vision_metrics.ipynb 3
def calculate_dsc(pred, targ):
    ''' MONAI `compute_meandice`'''

    return torch.Tensor([compute_dice(p[None], t[None]) for p, t in list(zip(pred,targ))])

# %% ../nbs/05_vision_metrics.ipynb 4
def calculate_haus(pred, targ):
    ''' MONAI `compute_hausdorff_distance`'''

    return torch.Tensor([compute_hausdorff_distance(p[None], t[None]) for p, t in list(zip(pred,targ))])

# %% ../nbs/05_vision_metrics.ipynb 5
def binary_dice_score(act, # Activation tensor [B, C, W, H, D]
                      targ # Target masks [B, C, W, H, D]
                     ) -> torch.Tensor:
    '''Calculate the mean Dice score for binary semantic segmentation tasks.'''

    pred = pred_to_binary_mask(act)
    dsc = calculate_dsc(pred.cpu(), targ.cpu())

    return torch.mean(dsc)

# %% ../nbs/05_vision_metrics.ipynb 6
def multi_dice_score(act, # Activation values [B, C, W, H, D]
                     targ # Target masks [B, C, W, H, D]
                    ) -> torch.Tensor:
    '''Calculate the mean Dice score for each class in multi-class semantic segmentation tasks.'''


    pred, n_classes = batch_pred_to_multiclass_mask(act)
    binary_dice_scores = []

    for c in range(1, n_classes):
        c_pred, c_targ = torch.where(pred==c, 1, 0), torch.where(targ==c, 1, 0)
        dsc = calculate_dsc(c_pred, c_targ)
        binary_dice_scores.append(np.nanmean(dsc)) #TODO update torch to get torch.nanmean() to work

    return torch.Tensor(binary_dice_scores)

# %% ../nbs/05_vision_metrics.ipynb 7
def binary_hausdorff_distance(act, # Activation tensor [B, C, W, H, D]
                              targ # Target masks [B, C, W, H, D]
                             ) -> torch.Tensor:
    '''Calculate the mean Hausdorff distance for binary semantic segmentation tasks.'''

    pred = pred_to_binary_mask(act)

    haus = calculate_haus(pred.cpu(), targ.cpu())
    return torch.mean(haus)

# %% ../nbs/05_vision_metrics.ipynb 8
def multi_hausdorff_distance(act, # Activation tensor [B, C, W, H, D]
                             targ # Target masks [B, C, W, H, D]
                            ) -> torch.Tensor :
    '''Calculate the mean Hausdorff distance for each class in multi-class semantic segmentation tasks.'''

    pred, n_classes = batch_pred_to_multiclass_mask(act)
    binary_haus = []

    for c in range(1, n_classes):
        c_pred, c_targ = torch.where(pred==c, 1, 0), torch.where(targ==c, 1, 0)
        haus = calculate_haus(pred, targ)
        binary_haus.append(np.nanmean(haus))
    return torch.Tensor(binary_haus)
