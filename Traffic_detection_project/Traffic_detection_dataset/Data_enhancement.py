import cv2
import albumentations as A
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm  # 用于显示进度条

# 定义类别和对应的索引
class_mapping = {
    "Car": 0,
    "Truck": 1,
    "Person": 2,
    "Two wheeled vehicle": 3,
    "Bus": 4,
}

# 定义数据增强管道
def get_transform(image_height, image_width):
    return A.Compose([
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.RandomBrightnessContrast(p=0.2),  # 亮度对比度调整
        A.Rotate(limit=15, p=0.5),  # 旋转
        A.RandomCrop(width=min(640, image_width), height=min(640, image_height), p=0.4),  # 随机裁剪
        A.Blur(blur_limit=3, p=0.2),  # 模糊
        A.CLAHE(p=0.2),  # 对比度增强
        A.RandomGamma(p=0.2),  # 随机伽马校正
        A.RGBShift(p=0.2),  # 色彩抖动
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=10, min_visibility=0.1))

# 读取YOLO格式的标注文件
def read_yolo_annotations(annotation_path):
    if not os.path.exists(annotation_path):
        return np.array([]), np.array([])  # 如果标签文件不存在，返回空数组
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # 确保每行有 5 个值
            class_id, x_center, y_center, width, height = map(float, parts)
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return np.array(bboxes), np.array(class_labels)

# 写入YOLO格式的标注文件
def write_yolo_annotations(annotation_path, bboxes, class_labels):
    with open(annotation_path, 'w') as file:
        for bbox, class_label in zip(bboxes, class_labels):
            file.write(f"{class_label} {' '.join(map(str, bbox))}\n")

# 检查边界框是否有效
def is_valid_bbox(bbox):
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return 0 <= x_min <= 1 and 0 <= y_min <= 1 and 0 <= x_max <= 1 and 0 <= y_max <= 1

# 修正边界框坐标
def clip_bbox(bbox):
    x_center, y_center, width, height = bbox
    x_center = max(0.0, min(x_center, 1.0))
    y_center = max(0.0, min(y_center, 1.0))
    width = max(0.0, min(width, 1.0))
    height = max(0.0, min(height, 1.0))
    return [x_center, y_center, width, height]

# 数据增强函数
def augment_data(image_path, annotation_path, output_image_path, output_annotation_path):
    # 读取图像和标签
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return
    bboxes, class_labels = read_yolo_annotations(annotation_path)

    # 如果标签为空，跳过增强
    if len(bboxes) == 0:
        return

    # 动态调整增强管道
    transform = get_transform(image.shape[0], image.shape[1])

    # 应用数据增强
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        # 过滤和修正无效边界框
        valid_bboxes = []
        valid_class_labels = []
        for bbox, class_label in zip(transformed_bboxes, transformed_class_labels):
            if is_valid_bbox(bbox):
                clipped_bbox = clip_bbox(bbox)  # 修正边界框
                valid_bboxes.append(clipped_bbox)
                valid_class_labels.append(class_label)

        # 如果没有有效的边界框，则跳过保存
        if not valid_bboxes:
            return

        # 保存增强后的图像和标注
        cv2.imwrite(output_image_path, transformed_image)
        write_yolo_annotations(output_annotation_path, valid_bboxes, valid_class_labels)
    except Exception as e:
        print(f"Error augmenting {image_path}: {e}")

# 处理类别不均的问题
def balance_classes(image_dir, annotation_dir, output_dir, target_count_per_class):
    # 初始化类别计数器
    class_counts = {class_id: 0 for class_id in class_mapping.values()}

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
    random.shuffle(image_files)

    # 创建输出目录
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)  # 创建 images 文件夹
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)  # 创建 labels 文件夹

    # 遍历图像文件
    for image_file in tqdm(image_files, desc="Balancing classes"):
        image_path = os.path.join(image_dir, image_file)
        annotation_path = os.path.join(annotation_dir, image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        # 读取标签
        bboxes, class_labels = read_yolo_annotations(annotation_path)

        # 检查是否需要增强
        for class_label in class_labels:
            if class_counts[class_label] < target_count_per_class:
                # 生成增强后的文件名
                output_image_path = os.path.join(output_dir, 'images', f"aug_{class_label}_{class_counts[class_label]}.jpg")
                output_annotation_path = os.path.join(output_dir, 'labels', f"aug_{class_label}_{class_counts[class_label]}.txt")

                # 进行数据增强
                augment_data(image_path, annotation_path, output_image_path, output_annotation_path)

                # 更新类别计数器
                class_counts[class_label] += 1

        # 如果所有类别都达到目标数量，提前退出
        if all(count >= target_count_per_class for count in class_counts.values()):
            break

    # 打印最终的类别分布
    print("Final class distribution:")
    for class_name, class_id in class_mapping.items():
        print(f"{class_name}: {class_counts[class_id]}")

# 设置路径和目标数量
DATASET_ROOT = Path(__file__).resolve().parent / 'dataset_file/Traffic_Detection_Dataset'
image_dir = str(DATASET_ROOT / 'dataset/split_dataset/train/images')  # 替换为图像目录
annotation_dir = str(DATASET_ROOT / 'dataset/split_dataset/train/labels')  # 替换为标签目录
output_dir = str(DATASET_ROOT / 'xfhz2_6222000/dataenhancement')  # 替换为输出目录
target_count_per_class = 2000  # 每个类别的目标数量

# 平衡类别并增强数据
balance_classes(image_dir, annotation_dir, output_dir, target_count_per_class)
