import logging

from monai.networks.nets import SwinUNETR as MonaiSwinUNETR
import torch

from .base import BaseSegModel

logger = logging.getLogger(__name__)


class SwinUNETR(BaseSegModel):
    """
    A wrapper for MONAI's SwinUNETR.
    State-of-the-art Transformer-based encoder for 3D segmentation.
    Note: For MONAI 1.1+, img_size is no longer required.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        feature_size=48,
        use_checkpoint=False,
        spatial_dims=3,
        pretrained_ssl_path=None,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = MonaiSwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims
        )
        if pretrained_ssl_path:
            self.load_ssl_pretrained(pretrained_ssl_path)

    def forward(self, x):
        return self.model(x)

    def load_ssl_pretrained(self, ssl_weights_path: str):
        """
        Load MONAI's official self-supervised pretrained SwinViT encoder
        weights (ssl_pretrained_weights.pth, from Project-MONAI/research-
        contributions SwinUNETR/BTCV, pretrained on 5050 CT scans). Applies
        the same key-remap the reference implementation uses (strip 'module.'
        DDP prefix, rename 'swin_vit' -> 'swinViT') before delegating to
        BaseSegModel.load_pretrained, which skips any shape-mismatched keys
        (e.g. if feature_size differs from the checkpoint's).
        """
        checkpoint = torch.load(ssl_weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

        if state_dict and "module." in next(iter(state_dict.keys())):
            logger.info("SSL checkpoint: stripping 'module.' DDP prefix.")
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if state_dict and "swin_vit" in next(iter(state_dict.keys())):
            logger.info("SSL checkpoint: renaming 'swin_vit' -> 'swinViT'.")
            state_dict = {k.replace("swin_vit", "swinViT"): v for k, v in state_dict.items()}

        # Re-namespace into this wrapper's own state_dict keys (self.model.<key>).
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        return self.load_pretrained(state_dict, strict_head=False)
