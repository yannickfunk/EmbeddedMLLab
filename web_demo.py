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
import PIL
from models.tinyyolov2 import TinyYoloV2Original, TinyYoloV2PersonOnly
import numpy as np

import sys
repo_path = sys.path[0]
os.environ["PATH"] = f"{repo_path}/ffmpeg_bins:" + os.environ["PATH"]

import gradio as gr

created_files = []
available_models = [e for e in os.listdir("data") if e.endswith(".onnx") or e.endswith(".pt")]

last_model = None
last_opt = None
last_sess = None


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

def predict_onnx(image, ort_sess):
    image = image[:, :, ::-1]
    image = Image.fromarray(image)
    image = image_transform(image)[0]
    image = tf.to_tensor(image)
    image = torch.unsqueeze(image, 0)
    onnx_predictions = ort_sess.run(None, {"input.1": image.numpy()})[0]
    return onnx_predictions

def predict_torch(image, model):
    image = Image.fromarray(image)
    image = image_transform(image)[0]
    image = tf.to_tensor(image)
    image = torch.unsqueeze(image, 0)
    with torch.no_grad():
        predictions = model(image)
    return predictions

def get_predictions_images(images, model_file, nms_threshold, box_threshold, enable_opt):
    global last_model, last_opt, last_sess
    person_only = "person_only" in model_file
    use_onnx = ".onnx" in model_file

    if use_onnx:
        if last_model == model_file and last_opt == enable_opt:
            print("reusing session")
            ort_sess = last_sess
        else:
            sess_options = ort.SessionOptions()
            if enable_opt:
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort_sess = ort.InferenceSession('data/'+model_file, sess_options=sess_options)
            last_model = model_file
            last_opt = enable_opt
            last_sess = ort_sess

        predict = lambda image, ort_sess: predict_onnx(image, ort_sess)
    else:
        if person_only:
            model = TinyYoloV2PersonOnly()
            model.load_state_dict(torch.load("data/"+model_file))
        else:
            model = TinyYoloV2Original()
            model.load_pt_from_disk("data/voc_pretrained.pt", discard_last_layer=False)
        predict = lambda image, model: predict_torch(image, model)


    prediction_images = []
    start = time.time_ns()
    for image in images:
        if use_onnx:
            prediction_result = predict(image, ort_sess)
        else:
            prediction_result = predict(image, model)

        prediction_image = plot_predictions(prediction_result, image,
                             return_array=True, nms_threshold=nms_threshold,
                             box_threshold=box_threshold, person_only=person_only)

        #prediction_images.append(prediction_image)

        # SOMEHOW IT IS INVERTED SUDDENLY???
        prediction_images.append(np.array(PIL.ImageOps.invert(Image.fromarray(prediction_image))))
    end = time.time_ns()
    fps = len(images) / ((end - start) / 1e+9)
    return prediction_images, fps


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


def _predict(gr_video, model_file, nms_threshold, box_threshold, enable_opt):
    images, fps = get_video_frames(gr_video)
    prediction_images, ms_per_frame = get_predictions_images(images, model_file,
                                                             nms_threshold, box_threshold, enable_opt)
    video_file = get_video_file_from_prediction_images(prediction_images, fps)
    return video_file, ms_per_frame


demo = gr.Interface(
    fn=_predict,
    inputs=[
        gr.Video(source="webcam", format="mp4"),
        gr.Dropdown(choices=available_models, label="Model File", value="pretrained.onnx"),
        gr.Slider(label="NMS Threshold", minimum=0, maximum=1, step=0.01, value=0.25),
        gr.Slider(label="Box Threshold", minimum=0, maximum=1, step=0.01, value=0.1),
        gr.Checkbox(label="Enable Onnx Optimizations", value=True)
    ],
    outputs=["playable_video", gr.Number(label="FPS", precision=2)],
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()