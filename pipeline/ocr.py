from typing import List
from PIL import Image
from surya.detection import DetectionPredictor
from surya.recognition import OCRResult, RecognitionPredictor


def recognize_text(preprocessed_images: list[Image.Image], threshold: float = 0.1) -> List[OCRResult]:
    """
    Recognize text from the given preprocessed images.

    Args:
        preprocessed_images (list[Image.Image]): list of preprocessed images
        threshold (float, optional): threshold for the confidence score. Defaults to 0.1.

    Returns:
        List[OCRResult]: list of recognized texts
    """
    detection_predictor = DetectionPredictor()
    recognition_predictor = RecognitionPredictor()
    recognized_texts = recognition_predictor(
        preprocessed_images, det_predictor=detection_predictor, math_mode=False)
    recognized_texts = threshold_text(
        recognized_texts, threshold)
    return recognized_texts


def threshold_text(recognized_texts: List[OCRResult], threshold: float) -> List[OCRResult]:
    """
    Threshold the recognized texts based on the confidence score.

    Args:
        recognized_texts (List[OCRResult]): list of recognized texts
        threshold (float, optional): threshold for the confidence score. Defaults to 0.5.

    Returns:
        List[OCRResult]: list of recognized texts with confidence score greater than threshold
    """
    filtered_pages: List[OCRResult] = []
    for page in recognized_texts:
        page_copy = page.model_copy()
        for i, line in enumerate(page.text_lines):
            if line.confidence < threshold:
                page_copy.text_lines.pop(i)
        filtered_pages.append(page_copy)
    return filtered_pages
