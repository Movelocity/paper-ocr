import cv2
import os
from .utils import BBox, pct_white, filter_overlapping_boxes
from typing import Collection
def handle_small_boxes(bboxes):
    # 处理可能的序号小框逻辑
    # 例如,将小框与相邻的大框合并
    # 这里需要根据具体情况实现
    return bboxes

def split_bboxes(bboxes, page_width) -> Collection[list[BBox]]:
    """按左右布局划分检测框"""
    mid_x = page_width / 2
    left_bboxes = []
    right_bboxes = []
    
    for bbox in bboxes:
        center_x = bbox.x + bbox.width / 2
        if center_x < mid_x or bbox.width > page_width / 2:
            left_bboxes.append(bbox)
        else:
            right_bboxes.append(bbox)
    
    # 按y坐标排序
    left_bboxes.sort(key=lambda b: b.y)
    right_bboxes.sort(key=lambda b: b.y)
    
    # 处理可能的序号小框
    left_bboxes = handle_small_boxes(left_bboxes)
    right_bboxes = handle_small_boxes(right_bboxes)
    
    return left_bboxes, right_bboxes

def is_small_box(bbox: BBox, avg_width: float, threshold: float = 0.5) -> bool:
    return bbox.width < avg_width * threshold

def are_in_same_row(bbox1: BBox, bbox2: BBox, avg_height: float, threshold: float = 0.5) -> bool:
    return abs(bbox1.y - bbox2.y) < avg_height * threshold

def sort_bboxes(bboxes: list[BBox]) -> list[BBox]:
    """从上到下进行框排序，同一行内的多个小框从左到右排序"""
    # 计算平均宽度和高度
    bboxes.sort(key=lambda bbox: bbox.y)
    for i in range(1, len(bboxes)):
        prev_box = bboxes[i-1]
        box = bboxes[i]
        if box.x < prev_box.x and box.y+box.height < prev_box.y+prev_box.height:
            bboxes[i-1], bboxes[i] = box, prev_box
    return bboxes


def segment(img, two_col=True, preview=False) -> list[BBox]:
    """Input: cv2 image of page. Output: BBox objects for content blocks in page"""
    MIN_TEXT_SIZE = 10
    # HORIZONTAL_POOLING = 25
    img_width = img.shape[1]
    
    # 创建保存目录
    if preview and not os.path.exists('./demo'):
        os.makedirs('./demo')

    # 使用自适应阈值法将灰度图像转换为二值图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if preview: 
        cv2.imwrite('./demo/1gray.jpg', gray)

    # 使用自适应阈值法将灰度图像转换为二值图像
    img_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    if preview: 
        cv2.imwrite('./demo/2img_bw.jpg', img_bw)

    # 对二值化后的图像进行高斯模糊处理
    blur = cv2.GaussianBlur(img_bw, (5,5), 0) 
    if preview: 
        cv2.imwrite('./demo/3blur.jpg', blur)

    # 形态学操作 使用梯度操作增强边缘
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m1 = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, k1)
    if preview: 
        cv2.imwrite('./demo/m1.jpg', m1)

    # # 形态学操作 使用闭运算填充小的孔洞
    # k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZONTAL_POOLING, 3))
    # m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k2)
    # if preview: 
    #     cv2.imwrite('./demo/m2.jpg', m2)

    # # 形态学操作 使用膨胀操作扩大边界
    # k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # m3 = cv2.dilate(m2, k3, iterations=2)
    # if preview: 
    #     cv2.imwrite('./demo/m3.jpg', m3)

    # 检测轮廓
    # contours = cv2.findContours(m3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = cv2.findContours(m1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    bboxes = []
    # 在原图上绘制边界框
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bh < MIN_TEXT_SIZE:
            continue
        if not pct_white(img[by:by+bh, bx:bx+bw]) < 1:
            continue 
        bboxes.append(BBox(bx, by, bw, bh))
    bboxes = filter_overlapping_boxes(bboxes)

    if two_col:
        left_bboxes, right_bboxes = split_bboxes(bboxes, img_width)
        if preview: 
            for bbox in left_bboxes:
                bx, by, bw, bh = bbox.x, bbox.y, bbox.width, bbox.height
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            for bbox in right_bboxes:
                bx, by, bw, bh = bbox.x, bbox.y, bbox.width, bbox.height
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            # cv2.imwrite('./demo/result1.jpg', img)

        left_bboxes = sort_bboxes(left_bboxes)
        right_bboxes = sort_bboxes(right_bboxes)
        bboxes = left_bboxes + right_bboxes
    else:
        bboxes = sort_bboxes(bboxes)

    # plot the index of sorted box in img
    if preview:
        for i, bbox in enumerate(bboxes):
            cv2.putText(img, str(i), (bbox.x, bbox.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 0), 2)
        if preview: 
            cv2.imwrite('./demo/result.jpg', img)
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
    bboxes = segment(img, True)
    # for bbox in bboxes:
    #     print(bbox.x, bbox.y, bbox.width, bbox.height)