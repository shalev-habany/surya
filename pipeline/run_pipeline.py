from typing import Callable, List
from PIL import Image
import numpy as np
from pipeline.ocr import recognize_text
from pipeline.preprocess import run_preprocess, DEFAULT_PAGE_LIMITS
from pipeline.table_recognition import run_table_recognition
from pipeline.pdf_utils import create_pdf


def run_pipeline(images: List[Image.Image], transforms_for_ocr: Callable[[List[np.ndarray]], List[np.ndarray]], transforms_for_table_rec: Callable[[List[np.ndarray]], List[np.ndarray]], out_path: str) -> None:
    preprocessed_images_for_ocr, preprocessed_images_for_table_rec, layout_result = run_preprocess(
        images, transforms_for_ocr, transforms_for_table_rec)
    ocr_results = recognize_text(preprocessed_images_for_ocr)
    tables, table_layouts_coordinates = run_table_recognition(
        preprocessed_images_for_table_rec, layout_result)
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = DEFAULT_PAGE_LIMITS
    w, h = bottom_right_x - top_left_x, bottom_right_y - top_left_y
    create_pdf(out_path, ocr_results, tables,
               table_layouts_coordinates, w, h)
    print(f"Created pdf at {out_path}")

def run_pipeline_without_pdf(images: List[Image.Image], transforms_for_ocr: Callable[[List[np.ndarray]], List[np.ndarray]], transforms_for_table_rec: Callable[[List[np.ndarray]], List[np.ndarray]]) -> Tuple[List[OCRResult], List[List[TableResult]], List[List[Tuple[int, int, int, int]]]]:
    preprocessed_images_for_ocr, preprocessed_images_for_table_rec, layout_result = run_preprocess(
        images, transforms_for_ocr, transforms_for_table_rec)
    ocr_results = recognize_text(preprocessed_images_for_ocr)
    tables, table_layouts_coordinates = run_table_recognition(
        preprocessed_images_for_table_rec, layout_result)
    return ocr_results, tables, table_layouts_coordinates