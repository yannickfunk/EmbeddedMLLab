import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl

import numpy as np

from utils.loss import YoloLoss
from utils.yolo import nms, filter_boxes
from utils.ap import precision_recall_levels, ap, display_roc

from typing import Optional, Union, Dict, Any
from torch import Tensor

import nni
from nni.algorithms.compression.v2.pytorch import LightningEvaluator

STEP_OUTPUT = Union[Tensor, Dict[str, Any], None]

ANCHORS = (
    (1.08, 1.19),
    (3.42, 4.41),
    (6.63, 11.38),
    (9.42, 5.11),
    (16.62, 10.52),
)


class TinyYoloV2(pl.LightningModule):
    def __init__(self, num_classes: int =20, learning_rate: int = 0.001):
        super().__init__()
        self.register_buffer("anchors", torch.tensor(ANCHORS))

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.loss = YoloLoss(anchors=ANCHORS)

        self.pad = nn.ReflectionPad2d((0, 1, 0, 1))

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(512)

        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8 = nn.Conv2d(1024, 1024, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(1024)

        self.conv9 = nn.Conv2d(1024, len(ANCHORS) * (5 + num_classes), 1, 1, 0)

    def forward(self, x, yolo=True):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.pad(x)
        x = F.max_pool2d(x, kernel_size=2, stride=1)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.leaky_relu(x, negative_slope=0.1, inplace=True)

        x = self.conv9(x)

        if yolo:
            nB, _, nH, nW = x.shape

            x = x.view(nB, self.anchors.shape[0], -1, nH, nW).permute(0, 1, 3, 4, 2)

            anchors = self.anchors.to(dtype=x.dtype, device=x.device)
            range_y, range_x, = torch.meshgrid(
                torch.arange(nH, dtype=x.dtype, device=x.device), torch.arange(nW, dtype=x.dtype, device=x.device)
            )
            anchor_x, anchor_y = anchors[:, 0], anchors[:, 1]

            x_center = (x[..., 0:1].sigmoid() + range_x[None, None, :, :, None]) / nW
            y_center = (x[..., 1:2].sigmoid() + range_y[None, None, :, :, None]) / nH
            width = (x[..., 2:3].exp() * anchor_x[None, :, None, None, None]) / nW
            height = (x[..., 3:4].exp() * anchor_y[None, :, None, None, None]) / nH
            confidence = x[..., 4:5].sigmoid()
            class_confidences = x[..., 5:].softmax(-1)
            x = torch.cat([x_center, y_center, width, height, confidence, class_confidences], -1)

        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs, yolo=False)
        loss, _ = self.loss.forward(outputs, targets)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs, yolo=False)
        loss, _ = self.loss.forward(outputs, targets)
        self.log("val_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, _ = batch

        return self(inputs, yolo=True), inputs

    def test_step(self, batch, batch_idx, dataloader_idx=0) -> Optional[STEP_OUTPUT]:
        inputs, targets = batch
        outputs = self(inputs, yolo=True)
        outputs = filter_boxes(outputs, 0.0)
        outputs = nms(outputs, 0.5)
        outputs = torch.tensor(np.array(outputs))

        precision, recall = precision_recall_levels(targets[0], outputs[0])

        self.log("precision", precision)
        self.log("recall", recall)

    def configure_optimizers(self) -> Any:
        optimizer = nni.trace(torch.optim.AdamW)(self.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        scheduler = {
            "scheduler": nni.trace(torch.optim.lr_scheduler.OneCycleLR)(optimizer=optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches),
            "interval": "step",
            "name": "LR"
        }

        return [optimizer]

    # def on_train_start(self) -> None:
    #     self.logger.log_hyperparams()

    # def on_train_start(self) -> None:
    #     self.logger.log_hyperparams()

    def load_pt_from_disk(self, pt_file):
        """
        For loading the pretrained file provided by Kilian
        """
        sd = torch.load(pt_file)
        self.load_state_dict(sd, strict=False)

    def on_fit_end(self) -> None:
        self.logger.finalize("success")


class TinyYoloV2PersonOnly(TinyYoloV2):
    def __init__(self):
        super().__init__(num_classes=1)

        # We only train the last 2 layers (conv8 and conv9)
        for key, param in self.named_parameters():
            if key.split(".")[0][-1] not in ["8", "9"]:
                param.requires_grad = False

    def load_pt_from_disk(self, pt_file):
        """
        For loading the pretrained file provided by Kilian
        """
        sd = torch.load(pt_file)
        sd = {k: v for k, v in sd.items() if not "9" in k}
        self.load_state_dict(sd, strict=False)
