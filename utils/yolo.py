import numpy as np
from typing import List
from utils.dataloader import class_to_num

def iou(bboxes1, bboxes2):
    """ calculate iou between each bbox in `bboxes1` with each bbox in `bboxes2`"""
    px, py, pw, ph = np.hsplit(bboxes1[...,:4].reshape(-1, 4), 4)
    lx, ly, lw, lh = np.hsplit(bboxes2[...,:4].reshape(-1, 4), 4)
    px1, py1, px2, py2 = px - 0.5 * pw, py - 0.5 * ph, px + 0.5 * pw, py + 0.5 * ph
    lx1, ly1, lx2, ly2 = lx - 0.5 * lw, ly - 0.5 * lh, lx + 0.5 * lw, ly + 0.5 * lh                
    dx = np.maximum(np.minimum(px2, lx2.T) - np.maximum(px1, lx1.T), [0])
    dy = np.maximum(np.minimum(py2, ly2.T) - np.maximum(py1, ly1.T), [0])
    intersections = dx * dy
    pa = (px2 - px1) * (py2 - py1) # area
    la = (lx2 - lx1) * (ly2 - ly1) # area
    unions = (pa + la.T) - intersections
    ious = (intersections/unions).reshape(*bboxes1.shape[:-1], *bboxes2.shape[:-1])
        
    return ious


def nms(filtered_array: List[np.ndarray], threshold: float) -> List[np.ndarray]:
    result = []
    for x in filtered_array:
        # Sort coordinates by descending confidence
        scores = np.sort(x[:, 4])        
        scores = scores[::-1]
        order = np.argsort(x[:, 4])
        order = order[::-1]        
        x = x[order]
        ious = iou(x,x) # get ious between each bbox in x

        # Filter based on iou
        keep = np.repeat(np.triu((ious > threshold).astype('int64'), 1).sum(0, keepdims=True).T, x.shape[1], axis=1) == 0
        result.append(np.ascontiguousarray(x[keep].reshape(-1, 6)))

    return result


def filter_boxes(output_array: np.ndarray, threshold, person_only=False) -> List[np.ndarray]:
    b, a, h, w, c = output_array.shape    
    x = np.ascontiguousarray(output_array).reshape(b, a * h * w, c)

    boxes = x[:, :, 0:4]
    confidence = x[:, :, 4]    
    scores = np.max(x[:, :, 5:], -1)
    idx = np.argmax(x[:, :, 5:], -1)

    if person_only:
        idx[...] = class_to_num("person")

    idx = idx.astype('float32')

    scores = scores * confidence
    mask = scores > threshold

    filtered = []
    for c, s, i, m in zip(boxes, scores, idx, mask):
        if m.any():
            detected = np.concatenate([c[m, :], s[m, None], i[m, None]], -1)
        else:
            detected = np.zeros((0, 6), dtype=x.dtype)
        filtered.append(detected)

    return filtered
