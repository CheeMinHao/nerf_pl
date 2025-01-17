import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import Tensor
from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import NeRF
from models.rendering import render_rays, visualize_depth
from models.embeddings import PosEmbedding

# optimizer, scheduler, visualization
from utils import get_optimizer, get_scheduler, get_learning_rate

# losses and metrics
from losses import loss_dict
from metrics import psnr

# pytorch-lightning
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.types import STEP_OUTPUT


class NeRFSystem(L.LightningModule):
    def __init__(self, hparams: argparse.Namespace) -> None:
        """Initialize NeRF training system
        
        Args:
            hparams: Hyperparameters from command line arguments
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        
        # Initialize loss
        self.loss = loss_dict['nerfw'](coef=1)
        
        # Initialize embeddings
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embeddings: Dict[str, Any] = {
            'xyz': self.embedding_xyz,
            'dir': self.embedding_dir
        }
        
        self.models_to_train: List[Any] = []
        
        # Initialize appearance and transient embeddings if needed
        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train.append(self.embedding_a)
            
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train.append(self.embedding_t)

        # Initialize NeRF models
        self.nerf_coarse = NeRF(
            'coarse',
            in_channels_xyz=6*hparams.N_emb_xyz+3,
            in_channels_dir=6*hparams.N_emb_dir+3
        )
        self.models: Dict[str, NeRF] = {'coarse': self.nerf_coarse}
        
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF(
                'fine',
                in_channels_xyz=6*hparams.N_emb_xyz+3,
                in_channels_dir=6*hparams.N_emb_dir+3,
                encode_appearance=hparams.encode_a,
                in_channels_a=hparams.N_a,
                encode_transient=hparams.encode_t,
                in_channels_t=hparams.N_tau,
                beta_min=hparams.beta_min
            )
            self.models['fine'] = self.nerf_fine
            
        self.models_to_train.append(self.models)

    def forward(self, rays: Tensor, ts: Tensor) -> Dict[str, Tensor]:
        """Perform batched inference on rays
        
        Args:
            rays: Ray batch
            ts: Timestep batch
            
        Returns:
            Dictionary containing rendered results
        """
        B = rays.shape[0]
        results: Dict[str, List[Tensor]] = defaultdict(list)
        
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = render_rays(
                self.models,
                self.embeddings,
                rays[i:i+self.hparams.chunk],
                ts[i:i+self.hparams.chunk],
                self.hparams.N_samples,
                self.hparams.use_disp,
                self.hparams.perturb,
                self.hparams.noise_std,
                self.hparams.N_importance,
                self.hparams.chunk,
                self.train_dataset.white_back
            )

            for k, v in rendered_ray_chunks.items():
                results[k].append(v)

        return {k: torch.cat(v, 0) for k, v in results.items()}

    def setup(self, stage: str) -> None:
        """Setup datasets for training and validation
        
        Args:
            stage: Current stage ('fit' or 'test')
        """
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs: Dict[str, Any] = {'root_dir': self.hparams.root_dir}
        
        if self.hparams.dataset_name == 'phototourism':
            kwargs.update({
                'img_downscale': self.hparams.img_downscale,
                'val_num': self.hparams.num_gpus,
                'use_cache': self.hparams.use_cache
            })
        elif self.hparams.dataset_name == 'blender':
            kwargs.update({
                'img_wh': tuple(self.hparams.img_wh),
                'perturbation': self.hparams.data_perturb
            })
            
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configure optimizers and learning rate schedulers"""
        optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader"""
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=4,
            batch_size=self.hparams.batch_size,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=4,
            batch_size=1,  # validate one image (H*W rays) at a time
            pin_memory=True
        )
    
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Execute training step
        
        Args:
            batch: Batch of data
            batch_idx: Index of current batch
            
        Returns:
            Training loss
        """
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_val = psnr(results[f'rgb_{typ}'], rgbs)

        # Log metrics
        self.log('lr', get_learning_rate(self.optimizers()))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_val, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> STEP_OUTPUT:
        """Execute validation step
        
        Args:
            batch: Batch of data
            batch_idx: Index of current batch
            
        Returns:
            Dictionary containing validation metrics
        """
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)
        
        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_idx == 0:
            # Log example images
            if self.hparams.dataset_name == 'phototourism':
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                W, H = self.hparams.img_wh
                
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))
            
            self.logger.experiment.add_images(
                'val/GT_pred_depth',
                torch.stack([img_gt, img, depth]),
                self.global_step
            )

        psnr_val = psnr(results[f'rgb_{typ}'], rgbs)
        
        return {'val_loss': loss, 'val_psnr': psnr_val}

    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) -> None:
        """Process validation epoch results
        
        Args:
            outputs: List of validation step outputs
        """
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams: argparse.Namespace) -> None:
    """Main training function
    
    Args:
        hparams: Command line arguments
    """
    # Set up model
    system = NeRFSystem(hparams)
    
    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(f'ckpts/{hparams.exp_name}'),
        filename='{epoch:d}',
        monitor='val/psnr',
        mode='max',
        save_top_k=-1
    )

    # Configure logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name,
        default_hp_metric=False
    )

    # Configure trainer
    trainer = L.Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=hparams.ckpt_path,
        logger=logger,
        accelerator='gpu' if hparams.num_gpus > 0 else 'cpu',
        devices=hparams.num_gpus,
        strategy='ddp' if hparams.num_gpus > 1 else 'auto',
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams.num_gpus == 1 else None
    )

    trainer.fit(system)


if __name__ == '__main__':
    from opt import get_opts
    hparams = get_opts()
    main(hparams)