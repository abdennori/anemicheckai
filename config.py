"""
Centralized configuration for AnemiCheck.
All tunables come from environment variables so behavior can change
per-environment (local / staging / production) without editing code.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Model files -----------------------------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
UNET_WEIGHTS_PATH = os.environ.get(
    "UNET_WEIGHTS_PATH", os.path.join(MODELS_DIR, "unet_model.pth")
)
CLASSIFIER_WEIGHTS_PATH = os.environ.get(
    "CLASSIFIER_WEIGHTS_PATH", os.path.join(MODELS_DIR, "efficientnet_b3.pth")
)
LOGO_PATH = os.environ.get("LOGO_PATH", os.path.join(BASE_DIR, "logo.png"))

# ---- Inference behavior ----------------------------------------------
FORCE_CPU = os.environ.get("FORCE_CPU", "false").lower() == "true"

# IMPORTANT / KNOWN ISSUE:
# The original training run produced a classifier whose sigmoid output is
# inverted relative to the intended label convention (high score = healthy,
# low score = anemic, instead of the other way round). Rather than silently
# burying a `1 - prediction` flip inside the prediction function, it is
# exposed here as an explicit, documented switch.
#
# This is a workaround, not a fix. The correct fix is to re-check the label
# mapping used during training (e.g. whether class 0/1 got swapped in the
# DataLoader or in the final Linear layer) and retrain/re-export the model
# so this flag can be set to False. Shipping a "production-ready" medical
# screening tool with an unexplained inverted-output patch is a real risk:
# if the inversion assumption doesn't hold for some checkpoint update or
# edge case, the tool will report the opposite of the correct diagnosis.
INVERT_CLASSIFIER_OUTPUT = (
    os.environ.get("INVERT_CLASSIFIER_OUTPUT", "true").lower() == "true"
)

# ---- UI -----------------------------------------------------------
# Optional LAN URL hint shown to users on the local network. Leave unset in
# production deployments (it is meaningless outside the developer's LAN).
LOCAL_NETWORK_HINT = os.environ.get("LOCAL_NETWORK_HINT", "")

MASK_MIN_AREA = int(os.environ.get("MASK_MIN_AREA", "500"))
BBOX_PADDING = int(os.environ.get("BBOX_PADDING", "15"))
