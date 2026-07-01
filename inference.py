"""
Image processing: mask cleanup, conjunctiva crop extraction, and the
classifier inference step.
"""
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config

_CLASSIFIER_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def clean_mask(mask: np.ndarray, min_area: int = None) -> np.ndarray:
    """Remove small contour artifacts and smooth the segmentation mask."""
    min_area = config.MASK_MIN_AREA if min_area is None else min_area

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)

    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(clean, [contour], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    return clean


def extract_best_conjunctiva(
    img: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Crop the image to the largest segmented region, with padding."""
    mask = clean_mask(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        padding = config.BBOX_PADDING
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)

        if w > 0 and h > 0:
            cropped = img[y : y + h, x : x + w]
            return cropped, mask, (x, y, w, h)

    conjunctiva = cv2.bitwise_and(img, img, mask=mask)
    return conjunctiva, mask, None


def predict_anemia(model, image, device) -> Tuple[str, float, float, float]:
    """
    Run the classifier and return (label, confidence_pct, corrected_prob,
    raw_prob).

    See config.INVERT_CLASSIFIER_OUTPUT for an explanation of why the raw
    sigmoid output may be flipped here -- this is a documented workaround
    for a known label-mapping issue in the current checkpoint, not an
    arbitrary design choice. Set that environment variable to "false" once
    the underlying model has been retrained with the correct mapping.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    img_tensor = _CLASSIFIER_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        raw_prediction = torch.sigmoid(output).item()

    corrected_prediction = (
        1 - raw_prediction if config.INVERT_CLASSIFIER_OUTPUT else raw_prediction
    )

    if corrected_prediction >= 0.5:
        result = "Anemic"
        confidence = corrected_prediction * 100
    else:
        result = "Non Anemic"
        confidence = (1 - corrected_prediction) * 100

    return result, confidence, corrected_prediction, raw_prediction
