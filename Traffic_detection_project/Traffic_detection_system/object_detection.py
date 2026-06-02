# object_detection
import torch
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path, device):
        self.model = YOLO(model_path).to(device)
        self.class_names = self.model.names

    # 在 object_detection.py 中
    def detect(self, image, conf_threshold=0.5):
        results = self.model(image, conf=conf_threshold)  # 使用传入的置信度阈值
        return results