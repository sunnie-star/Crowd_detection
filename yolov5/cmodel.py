import argparse
import os
import sys
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from .models.experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages, letterbox
from .utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from .utils.torch_utils import select_device, load_classifier, time_synchronized

class Model:
    def __init__(self):
        self.name = 'yolov5'
        self.device = select_device('0')
        self.half = self.device.type != 'cpu'
        self.model = attempt_load('./yolov5/yolov5s_subway.pt', map_location=self.device)
        self.imgsz = check_img_size(640, s=self.model.stride.max())
        if self.half:
            self.model.half()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None

    def predict(self, img0):
        model = self.model
        img = letterbox(img0, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False)[0]
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        pred = non_max_suppression(pred, 0.4, 0.5, 0, agnostic=False)
        rects = []
        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    # xywh = xywh * gn.numpy()
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=[0, 255, 0], line_thickness=3)
                    rects.append(np.array(xyxy))
        return [np.array(rects), img0]


if __name__ == '__main__':
    yolo = Model()

    img = cv2.imread('./yolov5/inference/images/zidane.jpg')
    out = yolo.predict(img)
    for rect in out:
        print(rect.astype(np.int))


