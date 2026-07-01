"""
Model loading for AnemiCheck.

Both models are loaded with weights_only=True. This is the safe mode for
torch.load: it refuses to unpickle arbitrary Python objects and only
reconstructs tensors, preventing a malicious .pth file from executing code
on load (OWASP-relevant when checkpoints might ever come from an upload,
a shared drive, or a less-trusted source than the original training run).
"""
import os
import logging

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models

import config

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if config.FORCE_CPU:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_file(path: str, friendly_name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{friendly_name} weights not found at '{path}'. "
            f"Set the corresponding environment variable to the correct "
            f"path or place the file there."
        )


@st.cache_resource(show_spinner=False)
def load_unet_model():
    import segmentation_models_pytorch as smp

    device = get_device()
    _require_file(config.UNET_WEIGHTS_PATH, "U-Net")

    unet = smp.Unet(
        encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1
    )
    state_dict = torch.load(
        config.UNET_WEIGHTS_PATH, map_location=device, weights_only=True
    )
    unet.load_state_dict(state_dict)
    unet.to(device)
    unet.eval()

    return unet, device


@st.cache_resource(show_spinner=False)
def load_classifier_model():
    device = get_device()
    _require_file(config.CLASSIFIER_WEIGHTS_PATH, "EfficientNet-B3")

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    checkpoint = torch.load(
        config.CLASSIFIER_WEIGHTS_PATH, map_location=device, weights_only=True
    )

    # Defensive: tolerate checkpoints saved with a different final-layer
    # shape (e.g. multi-class head) by dropping just that layer's weights
    # and letting the freshly-initialized nn.Linear above take over.
    checkpoint.pop("classifier.1.weight", None)
    checkpoint.pop("classifier.1.bias", None)

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        logger.warning("Classifier checkpoint missing keys: %s", missing)
    if unexpected:
        logger.warning("Classifier checkpoint had unexpected keys: %s", unexpected)

    model.to(device)
    model.eval()

    return model, device
