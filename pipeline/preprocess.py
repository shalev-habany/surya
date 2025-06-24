from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image
from surya.layout import LayoutPredictor, LayoutResult

DEFAULT_PAGE_LIMITS = (200, 0, 1600, 1500)


def crop_images(images: list[Image.Image], limits=DEFAULT_PAGE_LIMITS) -> List[Image.Image]:
    """
    Crop images to the given limits.
    """
    return [image.crop(limits) for image in images]


def preprocess_for_table_rec(cropped_images: list[Image.Image]) -> List[Image.Image]:
    np_cropped_images = [np.array(image) for image in cropped_images]
    preprocessed_images = [Image.fromarray(cv2.Canny(image, 100, 200))
                           for image in np_cropped_images]
    return preprocessed_images


def preprocess_for_ocr(cropped_images: list[Image.Image]) -> List[Image.Image]:
    np_cropped_images = [cv2.cvtColor(
        np.array(image), cv2.COLOR_BGR2GRAY) for image in cropped_images]
    preprocessed_images = []
    for img in np_cropped_images:
        _, binary_image = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(Image.fromarray(binary_image))
    return preprocessed_images


def run_preprocess(images: list[Image.Image]) -> Tuple[List[Image.Image], List[Image.Image], List[LayoutResult]]:
    """
    Run the preprocess pipeline for the given images.

    Args:
        images (list[Image.Image]): list of images to preprocess

    Returns:
        Tuple[List[Image.Image], List[Image.Image]]: 
        returns preprocessed images for OCR and table recognition (in this order)
    """
    cropped_images = crop_images(images)
    preprocessed_images_for_ocr = preprocess_for_ocr(cropped_images)
    preprocessed_images_for_table_rec = preprocess_for_table_rec(
        cropped_images)
    layout_result = extract_layout(images)
    return preprocessed_images_for_ocr, preprocessed_images_for_table_rec, layout_result


def extract_layout(images: list[Image.Image]) -> List[LayoutResult]:
    layout_predictor = LayoutPredictor()
    return layout_predictor(images)
