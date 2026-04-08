
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter

    return inter / union if union > 0 else 0.0


def sub_box(person_box, y_start_ratio, y_end_ratio):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return [x1, y1 + h * y_start_ratio, x2, y1 + h * y_end_ratio]


def best_region_match(person_box, ppe_boxes, region="full", min_iou=0.1):
    if region == "helmet":
        target_box = sub_box(person_box, 0.0, 0.4)
    elif region == "vest":
        target_box = sub_box(person_box, 0.3, 0.75)
    else:
        target_box = person_box

    best = None
    best_score = 0.0
    for ppe in ppe_boxes:
        score = iou_xyxy(target_box, ppe[:4])
        if score > best_score and score >= min_iou:
            best_score = score
            best = ppe
    return best
