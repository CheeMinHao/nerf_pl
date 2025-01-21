# metrics.py
import torch
from kornia.losses import ssim as dssim

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
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

def psnr(image_pred, image_gt, valid_mask = None, reduction: str = 'mean'):
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

def ssim(image_pred, image_gt, reduction = 'mean'):
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
                        reduction=reduction)
    return 1 - 2 * dssim_value