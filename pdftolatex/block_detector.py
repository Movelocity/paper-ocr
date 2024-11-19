import cv2
import os
from typing import List, Tuple
from .utils import BBox, pct_white, filter_overlapping_boxes

# Configuration
CONFIG = {
    'MIN_TEXT_SIZE': 10,
    'ADAPTIVE_THRESHOLD_BLOCK_SIZE': 11,
    'ADAPTIVE_THRESHOLD_C': 5,
    'GAUSSIAN_BLUR_KERNEL': (5, 5),
    'MORPHOLOGY_KERNEL_SIZE': (5, 5),
    'SMALL_BOX_THRESHOLD': 0.5,
    'SAME_ROW_THRESHOLD': 0.5
}

def split_bboxes(bboxes: List[BBox], page_width: int) -> Tuple[List[BBox], List[BBox]]:
    """Split bounding boxes into left and right columns."""
    mid_x = page_width / 2
    left_bboxes, right_bboxes = [], []
    
    for bbox in bboxes:
        if bbox.x + bbox.width / 2 < mid_x or bbox.width > page_width / 2:
            left_bboxes.append(bbox)
        else:
            right_bboxes.append(bbox)
    
    return sorted(left_bboxes, key=lambda b: b.y), sorted(right_bboxes, key=lambda b: b.y)

def is_small_box(bbox: BBox, avg_width: float) -> bool:
    return bbox.width < avg_width * CONFIG['SMALL_BOX_THRESHOLD']

def are_in_same_row(bbox1: BBox, bbox2: BBox, avg_height: float) -> bool:
    return abs(bbox1.y - bbox2.y) < avg_height * CONFIG['SAME_ROW_THRESHOLD']

def sort_bboxes(bboxes: List[BBox]) -> List[BBox]:
    """Sort bounding boxes from top to bottom, with small boxes in the same row sorted left to right."""
    bboxes.sort(key=lambda bbox: bbox.y)
    for i in range(1, len(bboxes)):
        prev_box, curr_box = bboxes[i-1], bboxes[i]
        if curr_box.x < prev_box.x and curr_box.y + curr_box.height < prev_box.y + prev_box.height:
            bboxes[i-1], bboxes[i] = curr_box, prev_box
    return bboxes

def preprocess_image(img: cv2.Mat) -> Tuple[cv2.Mat, cv2.Mat, cv2.Mat]:
    """Preprocess the input image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 
                                   CONFIG['ADAPTIVE_THRESHOLD_BLOCK_SIZE'], CONFIG['ADAPTIVE_THRESHOLD_C'])
    blur = cv2.GaussianBlur(img_bw, CONFIG['GAUSSIAN_BLUR_KERNEL'], 0)
    return gray, img_bw, blur

def apply_morphology(img: cv2.Mat) -> cv2.Mat:
    """Apply morphological operations to the image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG['MORPHOLOGY_KERNEL_SIZE'])
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def detect_contours(img: cv2.Mat) -> List[BBox]:
    """Detect contours in the image and convert them to BBox objects."""
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h >= CONFIG['MIN_TEXT_SIZE'] and pct_white(img[y:y+h, x:x+w]) < 1:
            bboxes.append(BBox(x, y, w, h))
    
    return filter_overlapping_boxes(bboxes)

def segment(img: cv2.Mat, two_col: bool = True, preview: bool = False) -> List[BBox]:
    """Segment the input image into content blocks."""
    img_width = img.shape[1]
    
    # Create preview directory if it doesn't exist
    if preview and not os.path.exists('./preview'):
        os.makedirs('./preview')

    # Preprocess image
    gray, img_bw, blur = preprocess_image(img)
    if preview:
        cv2.imwrite('./preview/1_gray.jpg', gray)
        cv2.imwrite('./preview/2_binary.jpg', img_bw)
        cv2.imwrite('./preview/3_blur.jpg', blur)

    # Apply morphology
    morph = apply_morphology(blur)
    if preview:
        cv2.imwrite('./preview/4_morphology.jpg', morph)

    # Detect contours and create bounding boxes
    bboxes = detect_contours(morph)

    # Split and sort bounding boxes
    if two_col:
        left_bboxes, right_bboxes = split_bboxes(bboxes, img_width)
        bboxes = sort_bboxes(left_bboxes) + sort_bboxes(right_bboxes)
    else:
        bboxes = sort_bboxes(bboxes)

    # Draw bounding boxes and indices on the image for preview
    if preview:
        preview_img = img.copy()
        for i, bbox in enumerate(bboxes):
            color = (0, 0, 255) if bbox in left_bboxes else (0, 255, 0)
            cv2.rectangle(preview_img, (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height), color, 2)
            cv2.putText(preview_img, str(i), (bbox.x, bbox.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
        cv2.imwrite('./preview/5_result.jpg', preview_img)

    return bboxes

"""
class BBox(
    x: int,
    y: int,
    width: int,
    height: int
)
"""
if __name__ == '__main__':
    img = cv2.imread('test.png')
    bboxes = segment(img, True, True)
    # for bbox in bboxes:
    #     print(bbox.x, bbox.y, bbox.width, bbox.height)