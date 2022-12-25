import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import numpy as np
from typing import List
import cv2
from utils.yolo import filter_boxes, nms
import math


CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    )


def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str:
            return idx


def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number:
            return string
    return 'none'


def to_uint8_img(img):
    img *= 255
    img = img.astype(np.uint8)
    return img


def display_result(image: np.ndarray, output: List[np.ndarray], file_path=None) -> None:
    _, ax = plt.subplots()
    bboxes = np.squeeze(np.stack(output, axis=0), axis=0)
    ax.imshow(image)

    # draw bounding boxes
    for bbox in bboxes:
        x_center, y_center, width, height, class_confidence, class_id = bbox
        if class_id > 0:
            cx = x_center * 320 - width * 320 // 2
            cy = y_center * 320 - height * 320 // 2
            w = width * 320
            h = height * 320

            rect = patches.Rectangle((cx, cy),
                                     w, h, linewidth=2, facecolor='none', edgecolor='r')
            ax.add_patch(rect)
            ax.annotate(num_to_class(int(class_id)) + " " + f"{float(class_confidence):.2f}", (cx, cy), color='r')

    plt.axis('off')
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
    plt.show()
    plt.close()


def get_result(image: np.ndarray, output: List[np.ndarray], overwrite_class_idx=-1):
    bboxes = np.squeeze(np.stack(output, axis=0), axis=0)
    image = to_uint8_img(image)
    image = image.copy()
    img_size_y = image.shape[0]
    img_size_x = image.shape[1]

    for bbox in bboxes:
        x_center, y_center, width, height, class_confidence, class_id = bbox
        if class_id > 0:
            cx = math.floor(x_center * img_size_x - width * img_size_x // 2)
            cy = math.floor(y_center * img_size_y - height * img_size_y // 2)
            w = math.floor(width * img_size_x)
            h = math.floor(height * img_size_y)
            cv2.rectangle(image, (cx, cy), (cx + w, cy + h), (255, 0, 0), 2)
            if overwrite_class_idx == -1:
                cv2.putText(image, num_to_class(int(class_id)) + " " + f"{float(class_confidence):.2f}",
                            (cx, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            else:
                cv2.putText(image, num_to_class(overwrite_class_idx) + " " + f"{float(class_confidence):.2f}",
                            (cx, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)

    return image


def plot_predictions(predictions, image,
                     box_threshold=0.1, nms_threshold=0.25,
                     return_array=False, person_only=False):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(image, torch.Tensor):
        image = image.numpy()
        try:
            image = np.squeeze(image, axis=0)
        except:
            pass
        if image.shape[0] == 3:
            image = image.transpose(1, 2, 0)

    # add batch dimensions
    if len(predictions.shape) == 4:
        predictions = np.expand_dims(predictions, axis=0)

    predictions = filter_boxes(predictions, box_threshold, person_only=person_only)
    # filter boxes based on overlap
    predictions = nms(predictions, nms_threshold)
    if return_array:
        return get_result(image, predictions)
    else:
        display_result(image, predictions)