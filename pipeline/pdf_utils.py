from typing import List, Tuple
import fitz
from surya.recognition import OCRResult
from bidi.algorithm import get_display
import re
from surya.table_rec import TableResult


def create_pdf(pdf_path: str, ocr_results: List[OCRResult], tables: List[List[TableResult]], table_layouts_coordinates: List[List[Tuple[int, int, int, int]]], w: int, h: int) -> None:
    doc = fitz.open()
    i = 0
    for ocr_result, table, table_layouts_coordinate in zip(ocr_results, tables, table_layouts_coordinates):
        page = doc.new_page(width=w, height=h)
        draw_words_recognition_on_image(page, ocr_result)
        draw_tables_recognition_on_image(
            page, table, table_layouts_coordinate)
        print(f"Created page {i}")
        i += 1
    doc.save(pdf_path)
    doc.close()


def draw_words_recognition_on_image(pdf: fitz.Page, lines: OCRResult) -> None:
    bboxes = [line.bbox for line in lines.text_lines]
    words = [line.text for line in lines.text_lines]
    for word, bbox in zip(words, bboxes):
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        font_size = max(8, min(int(0.6 * h), 50))
        rect = fitz.Rect(x, y, x + w, y + h)
        pdf.insert_textbox(
            rect, get_display(strip_html_tags(word), base_dir='R'),
            fontname="figo",
            fontsize=font_size,
            color=(0, 0, 0),
            overlay=True,
            render_mode=0,
            align=2
        )


def draw_tables_recognition_on_image(pdf: fitz.Page, tables: List[TableResult], table_layouts_coordinates: List[Tuple[int, int, int, int]]) -> None:
    if not tables:
        return
    for table, coordinates in zip(tables, table_layouts_coordinates):
        for cell in table.cells:
            bbox = tuple(map(int, cell.bbox))
            x0, y0, x1, y1 = transform_coordinates_to_original_doc(
                bbox, coordinates)
            rect = fitz.Rect(x0, y0, x1, y1)
            pdf.draw_rect(rect, width=2, color=fitz.utils.getColor("blue"))


def transform_coordinates_to_original_doc(cell_bbox: Tuple[int, int, int, int], coordinates: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    top_left_x, top_left_y, _, _ = coordinates
    return top_left_x + cell_bbox[0], top_left_y + cell_bbox[1], top_left_x + cell_bbox[2], top_left_y + cell_bbox[3]


def strip_html_tags(text: str) -> str:
    pattern = re.compile(r"<[\w/][^>]*>")
    text_only = pattern.sub("", text)
    return text_only
