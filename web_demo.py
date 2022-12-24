import os
import cv2
import onnxruntime as ort
from utils.viz import plot_predictions
import torch
import torchvision.transforms.functional as tf
from utils.dataloader import image_transform
from PIL import Image

import sys
repo_path = sys.path[0]
os.environ["PATH"] = f"{repo_path}:" + os.environ["PATH"]

import gradio as gr

ort_sess = ort.InferenceSession('data/pretrained.onnx')


def video_identity(video):
    vidcap = cv2.VideoCapture(video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)

    prediction_images = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        image = Image.fromarray(image)
        image = image_transform(image)[0]
        image = tf.to_tensor(image)
        image = torch.unsqueeze(image, 0)
        onnx_predictions = ort_sess.run(None, {"input.1": image.numpy()})[0]

        prediction_images.append(
            plot_predictions(onnx_predictions, image, return_array=True)
        )
    video_file = "output.mp4"
    try:
        os.remove(video_file)
    except:
        pass

    first = prediction_images[0]
    video_writer = cv2.VideoWriter(
        video_file,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (first.shape[1], first.shape[0])
    )
    for image in prediction_images:
        video_writer.write(image)
    video_writer.release()
    cv2.destroyAllWindows()

    return video_file


demo = gr.Interface(
    fn=video_identity,
    inputs=gr.Video(source="webcam", format="mp4"),
    outputs="playable_video",
)