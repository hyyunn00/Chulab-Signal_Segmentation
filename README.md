# Microscopy Segmentation Trainer/Inferencer

High-performance 2D/3D microscopy image segmentation using MONAI, PyTorch, and Shared Memory for efficient processing of massive datasets (150GB+).

## Features

- **SwinUNETR Support:** State-of-the-art Transformer-based encoder for 3D segmentation, fully integrated with padding safety.
- **Automatic Model Padding:** Ensures all input patches are at least the size of `patch_size` and their dimensions are divisible by 32 (required for SwinUNETR/Transformer models).
- **Numba Acceleration:** High-performance JIT-compiled algorithms for 3D patch cropping, mask filtering, and volume stitching.
- **Shared Memory:** Utilizes `torch.multiprocessing` to prevent RAM duplication across workers, critical for large volumes.
- **Asynchronous Pipeline:** Optimized inference using a synchronized Disk Manager thread to maximize sequential I/O speed.
- **Pre-Packed Patches:** Zero-computation inference workers by pre-cropping patches into shared contiguous tensors.
- **Hybrid Loss:** Focal + Tversky + Dice loss combinations to handle extreme class imbalance in sparse microscopy signals.
- **Intensity Normalization:** Support for Z-score, Min-Max, and Global Histogram Equalization via `preprocess.py` and reader integration.
- **16-bit Logic:** Optimized for 16-bit (uint16) microscopy data with automatic scaling for visualization (65535 for foreground).
- **Transfer Learning:** Architecture-agnostic warm-starting, encoder/backbone freezing, differential learning rates, and cross-architecture self-supervised backbone pretraining - see [Transfer Learning](#transfer-learning) below. Adding a new biomarker no longer requires training a model from scratch.

## Structure

- `preprocess.py`: Configuration-driven intensity normalization (Z-score, Min-Max, Histogram).
- `train.py`: Main training script with functional epoch handlers.
- `pretrain.py`: Tier-3 self-supervised backbone pretraining (masked-volume reconstruction) across unlabeled volumes from any biomarker.
- `inference.py`: Optimized batch inference script with async Disk Manager.
- `converter.py`: Utility for format conversion (OME-Zarr, Zarr, Tiff, Nifti).
- `analysis.py`: Metrics calculation (F1, Precision, Recall) against Ground Truth.
- `IO/`: Unified readers, writers, and shared-memory dataset classes.
- `models/`: Model architecture factory (UNet, AttentionUNet, SwinUNETR, VNet) built on a common `BaseSegModel` transfer-learning interface (`models/base.py`).
- `utils/`: Numba-optimized stitcher, patch cropper, visualization tools, and `checkpoint.py` (state_dict-based checkpoint save/load).

## Installation

1. Create a Python 3.10+ environment (e.g., using Miniconda).
2. **Install PyTorch** following the official instructions for your platform (CUDA/CPU):
   [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   
   Example (CUDA 11.8):
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Install remaining dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `numba` is used for high-speed JIT acceleration.*

4. Make scripts executable (Linux):
   ```bash
   chmod +x *.py
   ```

## Preprocessing

Before training or inference, you can normalize your volume intensities:
```bash
python preprocess.py --input /path/to/raw --output /path/to/norm --config configs/config.json --mode inference
```

## Configuration Guide

The behavior of all scripts is controlled via a central `configs/config.json` file. 

### 1. Global Resources (`resources`)
- `numba_threads`: Number of threads for JIT-accelerated operations (default: 8).
- `io_workers`: Number of background workers for file reading/writing (default: 4).
- `memory_limit`: Soft memory limit in GB to avoid OOM during large volume reads.

### 2. Normalization (`normalization`)
Global defaults for intensity normalization:
- `z-score`: `std_multiplier` (default: 1.0).
- `histogram`: `bins` (default: 1024).
- `min-max`: (no parameters required).

### 3. Format Converter (`converter`)
- `output_type`: Target format(s). Options: `OME-Zarr`, `Zarr`, `Tiff`, `Nifti`, `Scroll-Tiff`, `Scroll-Nifti`.
- `scroll_axis`: Axis for per-slice exports. `0-2` for forward, `3-5` for reverse.

### 4. Model Architecture (`model`)
- `swin_unetr`: feature_size=48, spatial_dims=3.
- `attention_unet`: channels=[32, 64, 128, 256, 512].
- `unet`: standard MONAI UNet configuration.

### 5. Training (`train`)
- `preprocess`: `method` ("z-score", "min-max", "histogram"), `low_cut`, `high_cut`.
- `training_patch_size`: [64, 64, 64]. *Note: Will be automatically padded if not divisible by 32.*

### 6. Inference (`inference`)
- `output`: `type` ("Scroll-Tiff", "Zarr", etc.), `dtype` ("uint16", "uint8").
- `inference_patch_size`: [64, 64, 64]. 
- `inference_overlay`: [16, 16, 16]. 
- `batch_size`: patch count processed by GPU at once.

## Transfer Learning

Every model in `models/` (`UNet`, `AttentionUNet`, `VNet`, `SwinUNETR`, and any future architecture added to `models/factory.py`) subclasses `models.base.BaseSegModel`, which auto-discovers the "head" (the final layer whose output channels equal `out_channels`) via a forward hook - no hand-written per-architecture code is needed. This gives every architecture, uniformly:

- `model.freeze_backbone()` / `model.unfreeze_backbone()`
- `model.param_groups(lr, backbone_lr_mult)` - differential learning rates for backbone vs. head
- `model.load_pretrained(state_dict, strict_head=False)` - loads matching-shape weights, silently skipping the head if `out_channels` differs (e.g. a new biomarker)
- `model.backbone_state_dict()` / `model.forward_backbone(x)` - used by `pretrain.py`

Checkpoints are saved as `{state_dict, meta}` (`utils/checkpoint.py`) instead of pickled whole model objects, so they can be partially reloaded into a differently-configured model. Old whole-object `.pth` checkpoints are still readable (`train.py`/`inference.py` detect the format automatically).

### Warm-starting a new biomarker (Tier 1 + 2)

In the new biomarker's `train` config section:
```json
"pretrained_path": "/path/to/an/existing/checkpoint.pth",
"freeze_backbone_epochs": 15,
"backbone_lr_mult": 0.1
```
The backbone is frozen for the first `freeze_backbone_epochs` epochs (head-only adaptation), then unfrozen for joint fine-tuning at `backbone_lr_mult * learning_rate`. `pretrained_path` can point at an existing same-architecture biomarker checkpoint, or at a Tier-3 SSL backbone (below).

For `swin_unetr`, you can additionally load MONAI's official self-supervised-pretrained SwinViT encoder (trained on 5050 CT scans) via `model.swin_unetr.pretrained_ssl_path`, pointing at a downloaded `ssl_pretrained_weights.pth` (see [Project-MONAI/research-contributions SwinUNETR/BTCV](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BTCV)).

### Cross-architecture self-supervised pretraining (Tier 3)

`pretrain.py` trains a masked-volume-reconstruction pretext task on unlabeled volumes pooled across *any number of biomarkers* (no masks required), producing one biomarker-agnostic backbone checkpoint per architecture:
```bash
python pretrain.py --config configs/config_pretrain.json
```
See `configs/config_pretrain.json` for the config shape (`pretrain.data_path` is a list of raw volume roots spanning multiple biomarkers/datasets, `pretrain.model_type` selects the architecture). The resulting `*_backbone.pth` checkpoint is meant to be used as `train.pretrained_path` when starting a brand-new biomarker.

Known limitation: for `unet`/`attention_unet`, MONAI's built-in implementations collapse channels to `out_channels` slightly before the very last layer, so the auto-discovered backbone/head boundary is narrower there than for `vnet`/`swin_unetr` (which have separately-named decoder/head submodules). The mechanism is still architecture-agnostic and correct - it just pretrains a bit less of the final layer for those two architectures.

## Evaluation

Calculate metrics against Ground Truth:
```bash
python analysis.py --base_dir ./datas/path/to/results --gt_name images_mask --pred_prefix images_mask_
```
Outputs a `metrics.xlsx` with detailed performance statistics.
