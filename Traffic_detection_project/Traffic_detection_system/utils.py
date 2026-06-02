# utils
import cv2
from PyQt5.QtGui import QImage, QPixmap, QColor

def get_color(class_name):
    if class_name in ["Car", "car"]:
        return (0, 255, 0)  # 绿色
    elif class_name in ["Truck", "truck"]:
        return (0, 255, 255)  # 黄色
    elif class_name in ["Person", "person"]:
        return (255, 0, 255)  # 紫色
    elif class_name in ["Two wheeled vehicle", "two_wheeled_vehicle"]:
        return (255, 165, 0)  # 蓝色
    elif class_name in ["Bus", "bus"]:
        return (255, 255, 0)  # 青色
    else:
        return (255, 255, 255)  # 白色

def display_image(frame, label_image):
    h, w, _ = frame.shape
    label_width = label_image.width()
    label_height = label_image.height()
    scale = min(label_width / w, label_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(convert_to_Qt_format)
    label_image.setPixmap(pixmap)