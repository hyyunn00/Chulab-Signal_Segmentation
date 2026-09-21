import logging

import torch
import torch.nn as nn

from .base import BaseSegModel

logger = logging.getLogger(__name__)

_CONV_CLS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}


class SSLReconstructionWrapper(nn.Module):
    """
    Wraps any BaseSegModel with a lightweight 1x1(x1) reconstruction head for
    Tier-3 self-supervised (masked-volume-reconstruction) pretraining. Works
    for any architecture in models/factory.py, since it only relies on
    BaseSegModel.forward_backbone() (auto-discovered, architecture-agnostic).

    After pretraining, only backbone.backbone_state_dict() is saved - the
    reconstruction head is a throwaway pretext-task component, never used
    for segmentation.
    """
    def __init__(self, backbone: BaseSegModel):
        super().__init__()
        self.backbone = backbone

        was_training = backbone.training
        backbone.eval()
        with torch.no_grad():
            feat = backbone.forward_backbone(backbone._dummy_input())
        if was_training:
            backbone.train()

        feat_channels = feat.shape[1]
        conv_cls = _CONV_CLS[backbone.spatial_dims]
        self.recon_head = conv_cls(feat_channels, backbone.in_channels, kernel_size=1)
        logger.info(
            f"SSLReconstructionWrapper: backbone feature channels={feat_channels}, "
            f"reconstructing to in_channels={backbone.in_channels}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.forward_backbone(x)
        return self.recon_head(feat)
