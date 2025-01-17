# metrics.py
import torch
from torch import Tensor
from typing import Optional, Union
from kornia.losses import ssim as dssim

def mse(image_pred: Tensor, 
        image_gt: Tensor, 
        valid_mask: Optional[Tensor] = None, 
        reduction: str = 'mean') -> Tensor:
    """
    Calculate Mean Squared Error

    Args:
        image_pred: Predicted image tensor
        image_gt: Ground truth image tensor
        valid_mask: Optional mask for valid pixels
        reduction: Reduction method ('mean' or 'none')

    Returns:
        MSE loss value
    """
    value = (image_pred - image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value

def psnr(image_pred: Tensor, 
         image_gt: Tensor, 
         valid_mask: Optional[Tensor] = None, 
         reduction: str = 'mean') -> Tensor:
    """
    Calculate Peak Signal-to-Noise Ratio

    Args:
        image_pred: Predicted image tensor
        image_gt: Ground truth image tensor
        valid_mask: Optional mask for valid pixels
        reduction: Reduction method ('mean' or 'none')

    Returns:
        PSNR value
    """
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask, reduction))

def ssim(image_pred: Tensor, image_gt: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Calculate Structural Similarity Index

    Args:
        image_pred: Predicted image tensor (1, 3, H, W)
        image_gt: Ground truth image tensor (1, 3, H, W)
        reduction: Reduction method

    Returns:
        SSIM value in [-1, 1]
    """
    # Updated for kornia 0.8.0
    dssim_value = dssim(image_pred, 
                        image_gt, 
                        window_size=3, 
                        reduction=reduction,
                        max_val=1.0)
    return 1 - 2 * dssim_value