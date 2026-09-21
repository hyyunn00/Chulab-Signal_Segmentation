import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    model: nn.Module,
    path: str,
    meta: Optional[Dict[str, Any]] = None,
    state_dict: Optional[Dict[str, Any]] = None,
):
    """
    Saves a model as a plain state_dict + metadata dict, instead of pickling
    the whole nn.Module. This is what makes transfer learning safe: a
    state_dict can be partially loaded into a differently-configured model
    (e.g. a new biomarker with a different out_channels), whereas a pickled
    module object cannot survive any change to the class definition and
    forces an exact architecture match.

    Pass `state_dict` explicitly to save a subset of the model's weights -
    e.g. Tier-3 pretraining saves only model.backbone_state_dict() so the
    (untouched, randomly-initialized) segmentation head is never mistaken
    for a pretrained weight.
    """
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "state_dict": state_dict if state_dict is not None else model.state_dict(),
        "meta": meta or {},
    }
    torch.save(payload, path)
    logger.info(f"[OK] Checkpoint saved to {path}")


def load_checkpoint(path: str, map_location="cpu") -> Dict[str, Any]:
    """
    Loads a checkpoint written by save_checkpoint(), and stays backward
    compatible with the legacy format used by earlier versions of this repo
    (torch.save(model, path), i.e. a pickled nn.Module) so existing
    production weights keep working.

    Returns a dict with keys: "state_dict" (Dict[str, Tensor] or None),
    "meta" (dict), and "legacy_model" (the pickled nn.Module, only set for
    old-format checkpoints).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    loaded = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(loaded, nn.Module):
        logger.info(f"[{path}] legacy whole-model checkpoint detected (pickled nn.Module).")
        return {"state_dict": None, "meta": {}, "legacy_model": loaded}

    if isinstance(loaded, dict) and "state_dict" in loaded:
        return {
            "state_dict": loaded["state_dict"],
            "meta": loaded.get("meta", {}),
            "legacy_model": None,
        }

    # Fallback: a bare state_dict was saved directly (no wrapper dict).
    if isinstance(loaded, dict):
        logger.info(f"[{path}] bare state_dict checkpoint detected (no meta).")
        return {"state_dict": loaded, "meta": {}, "legacy_model": None}

    raise ValueError(f"Unrecognized checkpoint format at {path}: {type(loaded)}")


def load_pretrained_into(model: nn.Module, ckpt_path: str, strict_head: bool = False) -> Dict[str, Any]:
    """
    Loads a checkpoint (either format) into `model`, using model.load_pretrained
    (see models/base.py BaseSegModel) so shape-mismatched keys - typically the
    final head layer when out_channels differs - are skipped instead of
    raising, letting the rest of the (shared) backbone warm-start the new model.
    """
    ckpt = load_checkpoint(ckpt_path)
    if ckpt["legacy_model"] is not None:
        state_dict = ckpt["legacy_model"].state_dict()
    else:
        state_dict = ckpt["state_dict"]

    if not hasattr(model, "load_pretrained"):
        raise TypeError(
            f"{type(model).__name__} does not implement load_pretrained(); "
            f"it must subclass models.base.BaseSegModel."
        )
    result = model.load_pretrained(state_dict, strict_head=strict_head)
    logger.info(f"Loaded pretrained weights from {ckpt_path} (meta={ckpt['meta']})")
    return result
