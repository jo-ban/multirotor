"""Checkpoint format for the joint cylindrical-velocity model."""
from pathlib import Path

import numpy as np
import torch

from data_utils_cylindrical import TARGET_SCALE, validate_checkpoint
from model_cylindrical import CylindricalUNet2D

MODEL_FILENAME = "rotor_unet_cyl2d_joint_best.pth"
MODEL_FORMAT = "joint_cylindrical_velocity_v1"
OUTPUT_CHANNELS = ("ur", "utheta", "uz")


def load_joint_model(device, model_dir=None, expected_rd=None, expected_radius_over_rd=2.0):
    path = Path(model_dir or ".") / MODEL_FILENAME
    if not path.is_file():
        legacy = list(path.parent.glob("rotor_unet_cyl2d_[uvw].pth"))
        hint = " Legacy U/V/W checkpoints cannot be used; retrain the joint model." if legacy else " Run train_nd.py first."
        raise FileNotFoundError(f"Joint checkpoint missing: {path}.{hint}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_joint_checkpoint(checkpoint, expected_rd, expected_radius_over_rd)
    model = CylindricalUNet2D(in_channels=4, out_channels=3,
                             base_channels=int(checkpoint["base_channels"])).to(device)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError as exc:
        raise ValueError("Joint checkpoint weights do not match its declared U-Net architecture") from exc
    model.eval()
    return model, float(checkpoint["target_scale"])


def validate_joint_checkpoint(checkpoint, expected_rd=None, expected_radius_over_rd=None):
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a joint cylindrical checkpoint; raw/legacy weights are incompatible")
    state = checkpoint.get("state_dict", {})
    output = state.get("out_conv.weight")
    if (checkpoint.get("model_format") != MODEL_FORMAT
            or tuple(checkpoint.get("output_channels", ())) != OUTPUT_CHANNELS
            or checkpoint.get("out_channels") != 3
            or output is None or output.shape[0] != 3):
        raise ValueError("Expected joint cylindrical (ur, utheta, uz) 3-channel checkpoint; legacy U/V/W weights are incompatible")
    if (not np.isclose(checkpoint.get("target_scale", np.nan), TARGET_SCALE)
            or "disk_radius_m" not in checkpoint or "radius_over_rd" not in checkpoint
            or "enc1.block.0.conv.weight" not in state):
        raise ValueError("Joint checkpoint has missing/inconsistent scale or geometry metadata")
    validate_checkpoint(checkpoint, expected_rd, expected_radius_over_rd)
