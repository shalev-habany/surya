from typing import Callable, List, Tuple
import cv2
import numpy as np
from PIL import Image
from surya.layout import LayoutPredictor, LayoutResult

DEFAULT_PAGE_LIMITS = (200, 0, 1600, 1500)


def crop_images(images: List[Image.Image], limits=DEFAULT_PAGE_LIMITS) -> List[Image.Image]:
    """
    Crop images to the given limits.
    """
    return [image.crop(limits) for image in images]


def preprocess_for_table_rec(cropped_images: List[Image.Image], transform_images: Callable[[List[np.ndarray]], List[np.ndarray]]) -> List[Image.Image]:
    np_cropped_images = [np.array(image) for image in cropped_images]
    preprocessed_images = transform_images(np_cropped_images)
    return preprocessed_images


def preprocess_for_ocr(cropped_images: List[Image.Image], transform_images: Callable[[List[np.ndarray]], List[np.ndarray]]) -> List[Image.Image]:
    np_cropped_images = [cv2.cvtColor(
        np.array(image), cv2.COLOR_BGR2GRAY) for image in cropped_images]
    preprocessed_images = transform_images(np_cropped_images)
    return preprocessed_images

def run_preprocess(images: List[Image.Image], transforms_for_ocr: Callable[[List[np.ndarray]], List[np.ndarray]], transforms_for_table_rec: Callable[[List[np.ndarray]], List[np.ndarray]]) -> Tuple[List[Image.Image], List[Image.Image], List[LayoutResult]]:
    """
    Run the preprocess pipeline for the given images.

    Args:
        images (List[Image.Image]): List of images to preprocess

    Returns:
        Tuple[List[Image.Image], List[Image.Image]]: 
        returns preprocessed images for OCR and table recognition (in this order)
    """
    cropped_images = crop_images(images)
    preprocessed_images_for_ocr = preprocess_for_ocr(cropped_images, transforms_for_ocr)
    preprocessed_images_for_table_rec = preprocess_for_table_rec(
        cropped_images, transforms_for_table_rec)
    layout_result = extract_layout(cropped_images)
    return preprocessed_images_for_ocr, preprocessed_images_for_table_rec, layout_result


def extract_layout(images: List[Image.Image]) -> List[LayoutResult]:
    layout_predictor = LayoutPredictor()
    return layout_predictor(images)
