import cv2
import numpy as np

def mask_to_bbox(mask_np):
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x, y, x + w, y + h)

def xyxy_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x_center = (x1 + x2) / 2.0 / img_w
    y_center = (y1 + y2) / 2.0 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def nms_per_class(detections, iou_thresh=0.25):
    by_class = {}
    has_conf = False
    for entry in detections:
        if len(entry) == 2:
            cid, box = entry
            conf = 1.0
        else:
            cid, box, conf = entry
            has_conf = True
        by_class.setdefault(cid, []).append((box, conf))
    filtered = []
    for cid, boxes in by_class.items():
        if len(boxes) <= 1:
            if has_conf:
                filtered.extend([(cid, b, c) for b, c in boxes])
            else:
                filtered.extend([(cid, b) for b, c in boxes])
            continue
        keep = []
        boxes_list = list(boxes)
        while boxes_list:
            current = boxes_list.pop(0)
            current_box, current_conf = current
            keep.append(current)
            boxes_list = [b for b in boxes_list if compute_iou(current_box, b[0]) < iou_thresh]
        if has_conf:
            filtered.extend([(cid, b, c) for b, c in keep])
        else:
            filtered.extend([(cid, b) for b, c in keep])
    return filtered