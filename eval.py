import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio.v3 as imageio  # Updated to newer imageio version
from argparse import ArgumentParser
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple  # Added type hints

from models.rendering import render_rays
from models.nerf import *
from utils import load_ckpt
import metrics
from datasets import dataset_dict
from datasets.depth_utils import *

def get_opts():
    parser = ArgumentParser()
    # Previous arguments remain the same
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to run on')  # Added device argument
    
    return parser.parse_args()

@torch.no_grad()
def batched_inference(models: Dict[str, torch.nn.Module],
                     embeddings: Dict[str, torch.nn.Module],
                     rays: torch.Tensor,
                     ts: Optional[torch.Tensor],
                     N_samples: int,
                     N_importance: int,
                     use_disp: bool,
                     chunk: int,
                     white_back: bool,
                     device: torch.device,
                     **kwargs) -> Dict[str, torch.Tensor]:
    """Do batched inference on rays using chunk.
    
    Args:
        models: Dictionary of NeRF models
        embeddings: Dictionary of embedding models
        rays: Ray batch tensor
        ts: Timestamp tensor (optional)
        N_samples: Number of coarse samples
        N_importance: Number of fine samples
        use_disp: Whether to use disparity sampling
        chunk: Chunk size for processing
        white_back: Whether to use white background
        device: Device to run inference on
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing rendered results
    """
    B = rays.shape[0]
    results = defaultdict(list)
    
    # Move inputs to device
    rays = rays.to(device)
    if ts is not None:
        ts = ts.to(device)
    
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                       embeddings,
                       rays[i:i+chunk],
                       ts[i:i+chunk] if ts is not None else None,
                       N_samples,
                       use_disp,
                       0,
                       0,
                       N_importance,
                       chunk,
                       white_back,
                       test_time=True,
                       device=device,
                       **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

def main():
    args = get_opts()
    device = torch.device(args.device)

    # Dataset setup
    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache
    dataset = dataset_dict[args.dataset_name](**kwargs)
    scene = os.path.basename(args.root_dir.strip('/'))

    # Initialize embeddings
    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz, device=device)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir, device=device)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    
    if args.encode_a:
        embedding_a = torch.nn.Embedding(args.N_vocab, args.N_a).to(device)
        load_ckpt(embedding_a, args.ckpt_path, model_name='embedding_a')
        embeddings['a'] = embedding_a
    if args.encode_t:
        embedding_t = torch.nn.Embedding(args.N_vocab, args.N_tau).to(device)
        load_ckpt(embedding_t, args.ckpt_path, model_name='embedding_t')
        embeddings['t'] = embedding_t

    # Initialize models
    nerf_coarse = NeRF('coarse',
                       in_channels_xyz=6*args.N_emb_xyz+3,
                       in_channels_dir=6*args.N_emb_dir+3,
                       device=device).to(device)
    
    nerf_fine = NeRF('fine',
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     encode_transient=args.encode_t,
                     in_channels_t=args.N_tau,
                     beta_min=args.beta_min,
                     device=device).to(device)

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    # Evaluation loop
    imgs, psnrs = [], []
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'
    os.makedirs(dir_name, exist_ok=True)

    kwargs = {'device': device}
    
    # PhototTourism specific setup
    if args.dataset_name == 'phototourism' and args.split == 'test':
        setup_phototourism_test(dataset, args, scene)
        kwargs['output_transient'] = False

    # Main evaluation loop
    for i in tqdm(range(len(dataset)), desc='Evaluating'):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']
        
        results = batched_inference(models, embeddings, rays, ts,
                                  args.N_samples, args.N_importance, args.use_disp,
                                  args.chunk,
                                  dataset.white_back,
                                  device,
                                  **kwargs)

        w, h = args.img_wh if args.dataset_name == 'blender' else sample['img_wh']
        
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        
        # Save individual frame
        imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)

        if 'rgbs' in sample:
            rgbs = sample['rgbs']
            img_gt = rgbs.view(h, w, 3)
            psnrs += [metrics.psnr(img_gt, img_pred).item()]

    # Save video if applicable
    if args.dataset_name == 'blender' or \
       (args.dataset_name == 'phototourism' and args.split == 'test'):
        save_video(imgs, dir_name, args.scene_name, args.video_format)

    # Print metrics
    if psnrs:
        mean_psnr = np.mean(psnrs)
        print(f'Mean PSNR : {mean_psnr:.2f}')

def setup_phototourism_test(dataset, args, scene):
    """Setup PhotoTourism testing parameters."""
    dataset.test_img_w, dataset.test_img_h = args.img_wh
    dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6)  # fov=60 degrees
    dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                              [0, dataset.test_focal, dataset.test_img_h/2],
                              [0, 0, 1]])
    
    if scene == 'brandenburg_gate':
        dataset.test_appearance_idx = 1123  # 85572957_6053497857.jpg
        setup_brandenburg_gate_poses(dataset)
    else:
        raise NotImplementedError

def setup_brandenburg_gate_poses(dataset):
    """Setup camera poses for Brandenburg Gate scene."""
    N_frames = 30*4
    dx = np.linspace(0, 0.03, N_frames)
    dy = np.linspace(0, -0.1, N_frames)
    dz = np.linspace(0, 0.5, N_frames)
    
    dataset.poses_test = np.tile(dataset.poses_dict[1123], (N_frames, 1, 1))
    for i in range(N_frames):
        dataset.poses_test[i, 0, 3] += dx[i]
        dataset.poses_test[i, 1, 3] += dy[i]
        dataset.poses_test[i, 2, 3] += dz[i]

def save_video(imgs: List[np.ndarray], 
               dir_name: str, 
               scene_name: str, 
               video_format: str):
    """Save sequence of images as a video file."""
    output_path = os.path.join(dir_name, f'{scene_name}.{video_format}')
    if video_format == 'gif':
        imageio.mimwrite(output_path, imgs, fps=30)
    else:  # mp4
        imageio.mimwrite(output_path, imgs, fps=30, quality=8)

if __name__ == "__main__":
    main()
