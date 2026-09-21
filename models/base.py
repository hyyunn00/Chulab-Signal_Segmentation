import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)


class BaseSegModel(nn.Module):
    """
    Common transfer-learning interface shared by every segmentation model
    wrapper (UNet, AttentionUNet, VNet, SwinUNETR, and any future architecture
    added to models/factory.py).

    Subclasses must set self.in_channels, self.out_channels and
    self.spatial_dims in __init__ before this base class's helpers are used.

    The "head" (final output-mapping layer) is discovered automatically by
    running one dummy forward pass and recording which conv/conv-transpose
    module whose out_channels == self.out_channels fires last. This avoids
    hand-written per-architecture name patterns, which would break every time
    MONAI's internal module layout changes or a new architecture is added.
    """

    def __init__(self):
        super().__init__()
        self._head_param_names: Optional[List[str]] = None
        self._head_module_name: Optional[str] = None

    # --- head discovery ----------------------------------------------------

    def _dummy_input(self, size: int = 64) -> torch.Tensor:
        shape = (1, self.in_channels) + (size,) * self.spatial_dims
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        return torch.zeros(*shape, device=device)

    def head_param_names(self) -> List[str]:
        if self._head_param_names is not None:
            return self._head_param_names

        candidates: List[Tuple[str, nn.Module]] = [
            (name, m) for name, m in self.named_modules()
            if isinstance(m, _CONV_TYPES) and getattr(m, "out_channels", None) == self.out_channels
        ]
        if not candidates:
            logger.warning(
                f"[{self.__class__.__name__}] no conv module with out_channels="
                f"{self.out_channels} found; falling back to the last conv module overall."
            )
            candidates = [(name, m) for name, m in self.named_modules() if isinstance(m, _CONV_TYPES)]
        if not candidates:
            self._head_param_names = []
            return self._head_param_names

        call_order: List[str] = []
        handles = []
        for name, m in candidates:
            handles.append(m.register_forward_hook(lambda mod, inp, out, n=name: call_order.append(n)))

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                self(self._dummy_input())
        except Exception as e:
            logger.warning(
                f"[{self.__class__.__name__}] head auto-discovery forward pass failed ({e}); "
                f"falling back to the last-registered candidate conv as head."
            )
            call_order = [candidates[-1][0]]
        finally:
            for h in handles:
                h.remove()
            if was_training:
                self.train()

        head_module_name = call_order[-1] if call_order else candidates[-1][0]
        prefix = f"{head_module_name}."
        self._head_module_name = head_module_name
        self._head_param_names = [
            name for name, _ in self.named_parameters() if name.startswith(prefix)
        ]
        logger.info(
            f"[{self.__class__.__name__}] discovered head module '{head_module_name}' "
            f"({len(self._head_param_names)} params)"
        )
        return self._head_param_names

    def invalidate_head_cache(self):
        """Call this if out_channels/spatial_dims change after construction."""
        self._head_param_names = None
        self._head_module_name = None

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the feature map feeding into the discovered head module - i.e.
        the shared backbone/trunk output, before the final out_channels-shaped
        prediction layer. Used by Tier-3 self-supervised pretraining
        (models/pretrain_wrapper.py) to attach a reconstruction head that is
        independent of the segmentation head. Works uniformly across
        architectures via the same forward-hook mechanism as head_param_names().
        """
        if self._head_module_name is None:
            self.head_param_names()  # populates self._head_module_name as a side effect

        head_module = dict(self.named_modules())[self._head_module_name]
        captured = {}

        def hook(module, inputs, output):
            captured["feat"] = inputs[0]

        handle = head_module.register_forward_hook(hook)
        try:
            self(x)
        finally:
            handle.remove()

        if "feat" not in captured:
            raise RuntimeError(
                f"[{self.__class__.__name__}] forward_backbone: head module "
                f"'{self._head_module_name}' was not invoked during forward()."
            )
        return captured["feat"]

    # --- param grouping ------------------------------------------------------

    def backbone_parameters(self):
        head_names = set(self.head_param_names())
        for name, p in self.named_parameters():
            if name not in head_names:
                yield p

    def head_parameters(self):
        head_names = set(self.head_param_names())
        for name, p in self.named_parameters():
            if name in head_names:
                yield p

    def freeze_backbone(self):
        n = 0
        for p in self.backbone_parameters():
            p.requires_grad_(False)
            n += 1
        logger.info(f"[{self.__class__.__name__}] backbone frozen ({n} tensors).")

    def unfreeze_backbone(self):
        n = 0
        for p in self.backbone_parameters():
            p.requires_grad_(True)
            n += 1
        logger.info(f"[{self.__class__.__name__}] backbone unfrozen ({n} tensors).")

    def param_groups(self, base_lr: float, backbone_lr_mult: float = 1.0) -> List[Dict]:
        """Differential-LR param groups for the optimizer."""
        backbone_params = list(self.backbone_parameters())
        head_params = list(self.head_parameters())
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult, "name": "backbone"})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr, "name": "head"})
        if not groups:
            groups = [{"params": list(self.parameters()), "lr": base_lr, "name": "all"}]
        return groups

    # --- pretrained loading ---------------------------------------------------

    def load_pretrained(self, state_dict: Dict[str, torch.Tensor], strict_head: bool = False) -> Dict[str, List[str]]:
        """
        Load a state_dict, skipping any key whose shape mismatches the current
        model. This is what lets a new biomarker (possibly with a different
        out_channels) warm-start from an existing/pretrained checkpoint: the
        head layer is silently skipped when shapes disagree, everything else
        (the shared backbone) is loaded.
        """
        own_state = self.state_dict()
        loaded, skipped_shape, skipped_missing = [], [], []
        filtered = {}
        for k, v in state_dict.items():
            if k not in own_state:
                skipped_missing.append(k)
                continue
            if own_state[k].shape != v.shape:
                skipped_shape.append(k)
                continue
            filtered[k] = v
            loaded.append(k)

        self.load_state_dict(filtered, strict=False)

        logger.info(
            f"[{self.__class__.__name__}] load_pretrained: loaded={len(loaded)}, "
            f"skipped(shape mismatch)={len(skipped_shape)}, skipped(missing in model)={len(skipped_missing)}"
        )
        if skipped_shape:
            logger.info(f"  shape-mismatched keys (expected if out_channels/in_channels changed): {skipped_shape}")
        if skipped_missing:
            logger.debug(f"  keys present in checkpoint but not in model: {skipped_missing}")

        if strict_head:
            head_names = set(self.head_param_names())
            missing_head = head_names - set(loaded)
            if missing_head:
                raise RuntimeError(
                    f"strict_head=True but the following head params were not loaded: {missing_head}"
                )

        return {"loaded": loaded, "skipped_shape": skipped_shape, "skipped_missing": skipped_missing}

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """Used by Tier-3 self-supervised pretraining to save a head-free checkpoint."""
        head_names = set(self.head_param_names())
        full = self.state_dict()
        return {k: v for k, v in full.items() if k not in head_names}
