import os
import tempfile
import time
import cv2
import onnxruntime as ort
from utils.viz import plot_predictions
import torch
import torchvision.transforms.functional as tf
from utils.dataloader import image_transform
from PIL import Image

import sys
repo_path = sys.path[0]
os.environ["PATH"] = f"{repo_path}/ffmpeg_bins:" + os.environ["PATH"]

import gradio as gr

created_files = []
available_models = [e for e in os.listdir("data") if e.endswith(".onnx")]


def get_video_frames(gr_video):
    vidcap = cv2.VideoCapture(gr_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    images = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        images.append(image)
    return images, fps


def get_predictions_images(images, model_file, nms_threshold, box_threshold):
    ort_sess = ort.InferenceSession('data/'+model_file)
    prediction_images = []
    person_only = model_file == "person_only.onnx"
    start = time.time_ns()
    for image in images:
        image = Image.fromarray(image)
        image = image_transform(image)[0]
        image = tf.to_tensor(image)
        image = torch.unsqueeze(image, 0)
        onnx_predictions = ort_sess.run(None, {"input.1": image.numpy()})[0]
        prediction_images.append(
            plot_predictions(onnx_predictions, image,
                             return_array=True, nms_threshold=nms_threshold,
                             box_threshold=box_threshold, person_only=person_only)
        )
    end = time.time_ns()
    ms_per_frame = ((end - start) / len(images)) / 1000000
    return prediction_images, ms_per_frame


def get_video_file_from_prediction_images(prediction_images, fps):
    global created_files
    video_file = tempfile.NamedTemporaryFile(suffix=".mp4")
    created_files.append(video_file)
    first = prediction_images[0]
    video_writer = cv2.VideoWriter(
        video_file.name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (first.shape[1], first.shape[0])
    )
    for image in prediction_images:
        video_writer.write(image)
    video_writer.release()
    cv2.destroyAllWindows()
    return video_file.name


def predict(gr_video, model_file, nms_threshold, box_threshold):
    images, fps = get_video_frames(gr_video)
    prediction_images, ms_per_frame = get_predictions_images(images, model_file, nms_threshold, box_threshold)
    video_file = get_video_file_from_prediction_images(prediction_images, fps)
    return video_file, ms_per_frame


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Video(source="webcam", format="mp4"),
        gr.Dropdown(choices=available_models, label="Model File", value="pretrained.onnx"),
        gr.Slider(label="NMS Threshold", minimum=0, maximum=1, step=0.01, value=0.25),
        gr.Slider(label="Box Threshold", minimum=0, maximum=1, step=0.01, value=0.1)
    ],
    outputs=["playable_video", gr.Number(label="milliseconds per frame", precision=2)],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()