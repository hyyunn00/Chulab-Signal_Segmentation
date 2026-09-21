#!/usr/bin/env python3
"""
Tier-3 self-supervised backbone pretraining.

Trains a masked-volume-reconstruction pretext task across ALL available
unlabeled microscopy volumes (any biomarker, any dataset under
pretrain.data_path), producing one biomarker-agnostic backbone checkpoint per
architecture (models/factory.py model_type). A new biomarker's train.py run
then warm-starts from this checkpoint via the `train.pretrained_path` /
`train.freeze_backbone_epochs` config fields instead of training from scratch.

Works for any architecture registered in models/factory.py: it only relies on
BaseSegModel.forward_backbone() (see models/base.py), which auto-discovers
the backbone/head boundary via a forward-hook, with no architecture-specific
code required here.
"""
import warnings
# Suppress the cuda.cudart module deprecation warning (must be done before other imports)
warnings.filterwarnings("ignore", category=FutureWarning, module="cuda")

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import json
from typing import Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.data.dataloader import DataLoader

from IO import build_pretrain_dataset_from_config
from models import build_model_from_config, SSLReconstructionWrapper
from utils.concurrency import initialize_concurrency
from utils.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)

pretrain_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
])


def random_block_mask(shape: Tuple[int, ...], mask_ratio: float, block_size: int) -> torch.Tensor:
    """
    Random cuboid/block binary mask (True = masked-out) over a spatial shape,
    for the SimMIM/MAE-style masked-reconstruction pretext task. Masking is
    done in coarse blocks (not per-voxel) so the network can't trivially
    interpolate from immediate neighbors.
    """
    grid_shape = tuple(max(1, s // block_size) for s in shape)
    n_blocks = 1
    for g in grid_shape:
        n_blocks *= g
    n_masked = max(1, round(n_blocks * mask_ratio))

    flat = torch.zeros(n_blocks, dtype=torch.bool)
    flat[torch.randperm(n_blocks)[:n_masked]] = True
    mask = flat.view(*grid_shape)

    for dim, g in enumerate(grid_shape):
        factor = -(-shape[dim] // g)  # ceil division
        mask = mask.repeat_interleave(factor, dim=dim)
    slicer = tuple(slice(0, s) for s in shape)
    return mask[slicer]


def pretrain_epoch(
    wrapper: SSLReconstructionWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    block_size: int,
    epoch: int,
) -> float:
    wrapper.train()
    total_loss = 0.0
    n_batches = max(1, len(loader))

    progress = tqdm(
        loader,
        desc=f"Pretrain Epoch {epoch + 1}",
        leave=False,
        bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    for images in progress:
        images = images.to(device, non_blocking=True)
        spatial_shape = images.shape[2:]

        masks = torch.stack([
            random_block_mask(spatial_shape, mask_ratio, block_size) for _ in range(images.shape[0])
        ]).to(device)
        mask_b = masks.unsqueeze(1).expand_as(images)

        masked_images = images.clone()
        masked_images[mask_b] = 0.0

        optimizer.zero_grad(set_to_none=True)
        recon = wrapper(masked_images)
        loss = F.mse_loss(recon[mask_b], images[mask_b])
        loss.backward()
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        progress.set_postfix({"recon_loss": f"{batch_loss:.4f}"})

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Tier-3 cross-architecture self-supervised backbone pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)

    initialize_concurrency(full_config)
    config = full_config.get("pretrain", {})

    save_root = config.get("save_path")
    model_type = config.get("model_type", "unet")
    run_name = config.get("run_name", f"{model_type}_ssl")

    if not config.get("data_path") or not save_root:
        logging.error("Missing mandatory 'data_path' or 'save_path' in pretrain config.")
        return 1

    save_dir = os.path.join(save_root, run_name)
    os.makedirs(save_dir, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(os.path.join(save_dir, f"pretrain_{timestamp}.log"), encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    import shutil
    shutil.copy2(args.config, os.path.join(save_dir, "config.json"))

    dataset = build_pretrain_dataset_from_config(full_config, pretrain_transform)
    logger.info(f"Pretraining dataset: {len(dataset)} unlabeled patches.")

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 8),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        persistent_workers=True,
        pin_memory=True
    )

    patch_size = config.get("patch_size", [64, 64, 64])
    spatial_dims = 3 if patch_size[0] > 1 else 2

    model_cfg = dict(full_config.get("model", {}).get(model_type, {}))
    model_cfg["model_type"] = model_type
    model_cfg["spatial_dims"] = spatial_dims
    backbone = build_model_from_config(model_cfg)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    backbone.to(device)
    wrapper = SSLReconstructionWrapper(backbone).to(device)

    optimizer = optim.AdamW(
        wrapper.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5)
    )

    mask_ratio = config.get("mask_ratio", 0.6)
    block_size = config.get("mask_block_size", 8)
    epochs = config.get("epochs", 100)
    checkpoint_interval = config.get("checkpoint_interval", 10)

    logging.info(f"Starting Tier-3 self-supervised pretraining: model_type={model_type}, "
                 f"{len(dataset)} patches, mask_ratio={mask_ratio}, block_size={block_size}")

    for epoch in range(epochs):
        avg_loss = pretrain_epoch(wrapper, loader, optimizer, device, mask_ratio, block_size, epoch)
        logger.info(f"Epoch {epoch + 1}/{epochs} - reconstruction loss: {avg_loss:.4f}")

        is_last = (epoch + 1) == epochs
        if (epoch + 1) % checkpoint_interval == 0 or is_last:
            ckpt_path = os.path.join(save_dir, f"{run_name}_backbone.pth")
            save_checkpoint(
                backbone,
                ckpt_path,
                meta={
                    "model_type": model_type,
                    "in_channels": backbone.in_channels,
                    "out_channels": backbone.out_channels,
                    "pretrain_task": "masked_volume_reconstruction",
                    "epoch": epoch + 1,
                },
                state_dict=backbone.backbone_state_dict(),
            )

    logging.info("Self-supervised pretraining complete.")


if __name__ == "__main__":
    main()
