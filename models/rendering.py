import torch
from einops import rearrange, reduce, repeat
import torch.nn.functional as F  # Added for modern PyTorch practices

def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    [Previous docstring remains the same]
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum')
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    # Updated searchsorted for newer PyTorch
    inds = torch.searchsorted(cdf, u)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    # Using where instead of manual masking
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    
    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

# The render_rays function remains largely the same, but with these modifications:
def render_rays(models,
                embeddings,
                rays,
                ts,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                device=None,  # Added device parameter
                **kwargs
                ):
    """
    [Previous docstring remains the same]
    """
    # Move all tensors to specified device
    if device is not None:
        rays = rays.to(device)
        ts = ts.to(device)
        for model in models.values():
            model.to(device)
    
    # Rest of the function remains the same, just ensure all tensor operations 
    # are performed on the same device
    
    # [Rest of the implementation remains the same]
