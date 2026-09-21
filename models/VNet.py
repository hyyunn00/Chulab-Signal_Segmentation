from monai.networks.nets import VNet as MonaiVNet
import torch

from .base import BaseSegModel

class VNet(BaseSegModel):
    """
    A wrapper for MONAI's VNet.
    Designed for volumetric medical image segmentation with residual connections.
    """
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        dropout_prob=0.5,
        dropout_dim=3
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = MonaiVNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob_down=dropout_prob,
            dropout_prob_up=(dropout_prob, dropout_prob),
            dropout_dim=dropout_dim,
        )

    def forward(self, x):
        return self.model(x)
