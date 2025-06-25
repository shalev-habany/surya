from PIL import Image
from surya.layout import LayoutResult
from typing import List, Tuple
from surya.table_rec import TableRecPredictor, TableResult


def crop_tables_from_images(images: List[Image.Image], layout_result: List[LayoutResult]) -> Tuple[List[List[Image.Image]], List[List[Tuple[int, int, int, int]]]]:
    """
    Crop tables from images based on layout result.

    Args:
        images (List[Image.Image]): list of images to crop tables from
        layout_result (List[LayoutResult]): layout result for the images

    Returns:
        Tuple[List[List[Image.Image]], List[List[Tuple[int, int, int, int]]]]: 
        list of cropped tables and list of table bounding box coordinates in the original image
    """
    table_crops = []
    table_bbox_coordinates = []
    for image, layout in zip(images, layout_result):
        single_table_coordinates = []
        single_page_results = []
        for box in layout.bboxes:
            if box.label == "Table":
                single_table_coordinates.append(box.bbox)
                single_page_results.append(image.crop(box.bbox))
        table_crops.append(single_page_results)
        table_bbox_coordinates.append(single_table_coordinates)
    return table_crops, table_bbox_coordinates


def recognize_tables(table_crops: List[List[Image.Image]]) -> List[List[TableResult]]:
    table_recognition_predictor = TableRecPredictor()
    return [table_recognition_predictor(table_crop) for table_crop in table_crops]


def run_table_recognition(images: List[Image.Image], layout_result: List[LayoutResult]) -> Tuple[List[List[TableResult]], List[List[Tuple[int, int, int, int]]]]:
    table_crops, table_bbox_coordinates = crop_tables_from_images(
        images, layout_result)
    return recognize_tables(table_crops), table_bbox_coordinates
