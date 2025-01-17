import argparse
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pickle
from datasets import PhototourismDataset
import logging
from dataclasses import dataclass

@dataclass
class CacheConfig:
    """Configuration for cache preparation"""
    root_dir: Path
    img_downscale: int
    cache_dir: Path

    def __post_init__(self):
        self.cache_dir = self.root_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)

def setup_logging() -> None:
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_opts() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Prepare PhototourismDataset cache')
    
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=1,
                        help='how much to downscale the images for phototourism dataset')

    return parser.parse_args()

def save_pickle(data: Any, filepath: Path, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    Save data to a pickle file
    
    Args:
        data: Data to save
        filepath: Path to save the file
        protocol: Pickle protocol version
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol)

def save_numpy(data: np.ndarray, filepath: Path) -> None:
    """
    Save numpy array to file
    
    Args:
        data: Numpy array to save
        filepath: Path to save the file
    """
    np.save(filepath, data)

def prepare_cache(config: CacheConfig) -> None:
    """
    Prepare and save dataset cache
    
    Args:
        config: Cache configuration
    """
    logging.info(f'Preparing cache for scale {config.img_downscale}...')
    
    try:
        # Initialize dataset
        dataset = PhototourismDataset(str(config.root_dir), 'train', config.img_downscale)
        
        # Save basic information
        save_pickle(dataset.img_ids, config.cache_dir / 'img_ids.pkl')
        save_pickle(dataset.image_paths, config.cache_dir / 'image_paths.pkl')
        save_pickle(dataset.Ks, config.cache_dir / f'Ks{config.img_downscale}.pkl')
        
        # Save scene geometry
        save_numpy(dataset.xyz_world, config.cache_dir / 'xyz_world.npy')
        save_numpy(dataset.poses, config.cache_dir / 'poses.npy')
        
        # Save bounds
        save_pickle(dataset.nears, config.cache_dir / 'nears.pkl')
        save_pickle(dataset.fars, config.cache_dir / 'fars.pkl')
        
        # Save rays and RGB values
        save_numpy(dataset.all_rays.numpy(), 
                  config.cache_dir / f'rays{config.img_downscale}.npy')
        save_numpy(dataset.all_rgbs.numpy(), 
                  config.cache_dir / f'rgbs{config.img_downscale}.npy')
        
        logging.info(f"Data cache saved to {config.cache_dir} !")
        
    except Exception as e:
        logging.error(f"Error preparing cache: {str(e)}")
        raise

def main() -> None:
    """Main execution function"""
    setup_logging()
    args = get_opts()
    
    config = CacheConfig(
        root_dir=Path(args.root_dir),
        img_downscale=args.img_downscale,
        cache_dir=Path(args.root_dir) / 'cache'
    )
    
    prepare_cache(config)

if __name__ == '__main__':
    main()