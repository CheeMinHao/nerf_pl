# losses.py
import torch
from torch import nn

class ColorLoss(nn.Module):
    def __init__(self, coef = 1.0):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        """
        Calculate color loss for NeRF outputs
        
        Args:
            inputs: Dictionary containing rgb values
            targets: Ground truth RGB values
            
        Returns:
            Weighted loss value
        """
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss


class NerfWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        b_l: beta loss (2nd term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef = 1.0, lambda_u = 0.01):
        """
        Args:
            coef: Loss coefficient
            lambda_u: Uncertainty lambda from equation 13
        """
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u

    def forward(self, inputs, targets):
        """
        Calculate NeRF-W losses
        
        Args:
            inputs: Dictionary containing model outputs
            targets: Ground truth values
            
        Returns:
            Dictionary containing individual loss components
        """
        ret = {}
        ret['c_l'] = 0.5 * torch.mean((inputs['rgb_coarse'] - targets)**2)
        
        if 'rgb_fine' in inputs:
            if 'beta' not in inputs:  # no transient head, normal MSE loss
                ret['f_l'] = 0.5 * torch.mean((inputs['rgb_fine'] - targets)**2)
            else:
                beta_expanded = inputs['beta'].unsqueeze(1)
                ret['f_l'] = torch.mean(
                    (inputs['rgb_fine'] - targets)**2 / (2 * beta_expanded**2)
                )
                ret['b_l'] = 3 + torch.mean(torch.log(inputs['beta']))  # +3 to make it positive
                ret['s_l'] = self.lambda_u * torch.mean(inputs['transient_sigmas'])

        # Apply coefficient to all loss components
        ret = {k: self.coef * v for k, v in ret.items()}
        return ret


# Updated loss dictionary with type hint
loss_dict = {
    'color': ColorLoss,
    'nerfw': NerfWLoss
}