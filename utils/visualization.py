import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if torch.is_tensor(depth):
        x = depth.cpu().numpy()
    else:
        x = np.array(depth)

    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    
    # Apply colormap and convert to PIL Image
    x_colored = cv2.applyColorMap(x, cmap)
    x_colored = cv2.cvtColor(x_colored, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    x_pil = Image.fromarray(x_colored)
    
    # Convert to tensor using newer transform syntax
    transform = T.ToTensor()
    x_tensor = transform(x_pil)  # (3, H, W)

    return x_tensor