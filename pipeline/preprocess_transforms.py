import cv2
import numpy as np
from typing import List

def binerize_images(images: List[np.ndarray]) -> List[np.ndarray]:
    return [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] for img in images]

def canny_images(images: List[np.ndarray]) -> List[np.ndarray]:
    return [cv2.Canny(img, 100, 200) for img in images]

# def align_images(images: List[np.ndarray]) -> List[np.ndarray]:
    