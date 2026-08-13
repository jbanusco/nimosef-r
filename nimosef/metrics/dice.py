import torch.nn.functional as F
import numpy as np


def dice_coefficient(pred, target, smooth=1e-6, per_slice=True, per_frame=True, apply_softmax=False):
    """
    Compute Dice coefficient for segmentation results.
    """
    if apply_softmax:
        pred = F.softmax(pred, dim=1)
    
    pred = np.asarray(pred).astype(int)
    target = np.asarray(target).astype(int)

    labels = np.unique(target)
    dice_per_class = []

    for l in labels:
        if l == 0:  # skip background
            continue
        pred_l = (pred == l).astype(np.float32)
        target_l = (target == l).astype(np.float32)

        if np.sum(target_l) == 0 and np.sum(pred_l) == 0:
            dice = 1.0
        elif np.sum(target_l) == 0 or np.sum(pred_l) == 0:
            dice = 0.0
        else:
            if per_slice:
                intersection = np.sum(pred_l * target_l, axis=(0, 1, 3))
                union = np.sum(pred_l, axis=(0, 1, 3)) + np.sum(target_l, axis=(0, 1, 3))
                dice = (2.0 * intersection + smooth) / (union + smooth)
            elif per_frame:
                intersection = np.sum(pred_l * target_l, axis=(0, 1, 2))
                union = np.sum(pred_l, axis=(0, 1, 2)) + np.sum(target_l, axis=(0, 1, 2))
                dice = (2.0 * intersection + smooth) / (union + smooth)
            else:
                intersection = np.sum(pred_l * target_l)
                union = np.sum(pred_l) + np.sum(target_l)
                dice = (2.0 * intersection + smooth) / (union + smooth)

        dice_per_class.append(dice)

    return np.array(dice_per_class)
