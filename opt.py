# opt.py
import argparse
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TrainingConfig:
    """Configuration class for training parameters"""
    batch_size: int
    num_epochs: int
    optimizer: str
    lr: float
    num_gpus: int
    
def get_opts() -> argparse.Namespace:
    """
    Get command line arguments with proper organization and typing
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='NeRF Training Options')
    
    # Dataset Configuration
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--root_dir', type=str, required=True,
                           help='root directory of dataset')
    data_group.add_argument('--dataset_name', type=str, default='blender',
                           choices=['blender', 'phototourism'],
                           help='which dataset to train/val')
    data_group.add_argument('--data_perturb', nargs="+", type=str, default=[],
                           help='Available choices: [], ["color"], ["occ"] or ["color", "occ"]')
    data_group.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                           help='resolution (img_w, img_h) of the image')
    data_group.add_argument('--img_downscale', type=int, default=1,
                           help='how much to downscale the images for phototourism dataset')
    data_group.add_argument('--use_cache', action="store_true",
                           help='whether to use ray cache')

    # Model Configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--N_emb_xyz', type=int, default=10,
                            help='number of xyz embedding frequencies')
    model_group.add_argument('--N_emb_dir', type=int, default=4,
                            help='number of direction embedding frequencies')
    model_group.add_argument('--N_samples', type=int, default=64,
                            help='number of coarse samples')
    model_group.add_argument('--N_importance', type=int, default=128,
                            help='number of additional fine samples')
    model_group.add_argument('--use_disp', action="store_true",
                            help='use disparity depth sampling')
    model_group.add_argument('--perturb', type=float, default=1.0,
                            help='factor to perturb depth sampling points')
    model_group.add_argument('--noise_std', type=float, default=1.0,
                            help='std dev of noise added to regularize sigma')

    # NeRF-W Configuration
    nerfw_group = parser.add_argument_group('NeRF-W Configuration')
    nerfw_group.add_argument('--N_vocab', type=int, default=100,
                            help='number of vocabulary (images) for embedding')
    nerfw_group.add_argument('--encode_a', action="store_true",
                            help='whether to encode appearance (NeRF-A)')
    nerfw_group.add_argument('--N_a', type=int, default=48,
                            help='number of embeddings for appearance')
    nerfw_group.add_argument('--encode_t', action="store_true",
                            help='whether to encode transient object (NeRF-U)')
    nerfw_group.add_argument('--N_tau', type=int, default=16,
                            help='number of embeddings for transient objects')
    nerfw_group.add_argument('--beta_min', type=float, default=0.1,
                            help='minimum color variance for each ray')

    # Training Configuration
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--batch_size', type=int, default=1024,
                            help='batch size')
    train_group.add_argument('--chunk', type=int, default=32*1024,
                            help='chunk size to split input to avoid OOM')
    train_group.add_argument('--num_epochs', type=int, default=16,
                            help='number of training epochs')
    train_group.add_argument('--num_gpus', type=int, default=1,
                            help='number of gpus')
    train_group.add_argument('--ckpt_path', type=str, default=None,
                            help='pretrained checkpoint path to load')
    train_group.add_argument('--prefixes_to_ignore', nargs='+', type=str, 
                            default=['loss'],
                            help='prefixes to ignore in checkpoint state dict')

    # Optimizer Configuration
    opt_group = parser.add_argument_group('Optimizer Configuration')
    opt_group.add_argument('--optimizer', type=str, default='adam',
                          choices=['sgd', 'adam', 'radam', 'ranger'],
                          help='optimizer type')
    opt_group.add_argument('--lr', type=float, default=5e-4,
                          help='learning rate')
    opt_group.add_argument('--momentum', type=float, default=0.9,
                          help='learning rate momentum')
    opt_group.add_argument('--weight_decay', type=float, default=0,
                          help='weight decay')
    opt_group.add_argument('--lr_scheduler', type=str, default='steplr',
                          choices=['steplr', 'cosine', 'poly'],
                          help='scheduler type')
    
    # Warmup Configuration
    warmup_group = parser.add_argument_group('Warmup Configuration')
    warmup_group.add_argument('--warmup_multiplier', type=float, default=1.0,
                             help='lr multiplier after warmup')
    warmup_group.add_argument('--warmup_epochs', type=int, default=0,
                             help='epochs for warmup')
    
    # Scheduler Configuration
    sched_group = parser.add_argument_group('Scheduler Configuration')
    sched_group.add_argument('--decay_step', nargs='+', type=int, default=[20],
                            help='scheduler decay step')
    sched_group.add_argument('--decay_gamma', type=float, default=0.1,
                            help='learning rate decay amount')
    sched_group.add_argument('--poly_exp', type=float, default=0.9,
                            help='polynomial learning rate decay exponent')

    # Misc Configuration
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--exp_name', type=str, default='exp',
                           help='experiment name')
    misc_group.add_argument('--refresh_every', type=int, default=1,
                           help='print progress bar every X steps')

    return parser.parse_args()