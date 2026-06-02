import os
from pathlib import Path
from ultralytics import YOLO
import multiprocessing
import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRAIN_DIR.parent
REPO_ROOT = PROJECT_DIR.parent

def main():
    # 保存训练结果的文件夹
    save_root = TRAIN_DIR / "results"

    model = YOLO(str(REPO_ROOT / "ultralytics/cfg/models/v8/yolov8n.yaml"))
    model.load(str(PROJECT_DIR / "Traffic_detection_system/best_weights/yolov8n.pt"))
    # 加载模型
    # model = YOLO(str(PROJECT_DIR / "Traffic_detection_system/best_weights/yolov8n.pt"))  # 加载预训练模型

    # 训练模型
    model.train(
        data=str(TRAIN_DIR / "Traffic_Detection_Dataset.yaml"),
        epochs=300,  # 训练轮次
        imgsz=640,  # 输入图像尺寸调整为1024
        device=0,  # 使用GPU训练
        amp=True,  # 启用混合精度，减少显存占用并提高训练速度
        workers=8,  # 数据加载线程数，加速数据加载
        patience=30,  # 早停机制，如果验证集性能在指定 epoch 内没有提升，则提前停止训练
        batch=32,  # 增加批量大小以提高GPU利用率
        lr0=0.01,  # 初始学习率
        optimizer="SGD",  # 优化器
        project=str(save_root),
        name="Traffic_detection_yolov8n",
        # cos_lr=True,  # 使用余弦退火学习率调度器
        # warmup_epochs=5,  # 预热epochs
        # weight_decay=0.0005,  # 权重衰减
        # momentum=0.937,  # 动量
        # hsv_h=0.01,  # 色调增强
        # hsv_s=0.1,  # 饱和度增强
        # hsv_v=0.1,  # 亮度增强
        # degrees=5.0,  # 旋转增强
        # translate=0.1,  # 平移增强
        # scale=0.5,  # 缩放增强
        # shear=0.001,  # 剪切增强
        # fliplr=0.5,  # 左右翻转增强
        # mosaic=0.5,  # Mosaic数据增强
        # mixup=0.5,  # Mixup数据增强
    )

def evaluate_model(model, save_path):
    """评估模型并保存指标"""
    # 评估模型在验证集上的性能
    metrics = model.val()

    # 计算 F1 Score
    precision = np.mean(metrics.precision)
    recall = np.mean(metrics.recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # 打印评估指标
    print(f"mAP50-95: {metrics.AP}")
    print(f"mAP50: {metrics.AP50}")
    print(f"Mean Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print("Class-wise AP:")
    for i, ap in enumerate(metrics.AP_per_class):
        print(f"Class {i}: AP = {ap}")

    # 保存评估指标到文件
    save_metrics_to_file(metrics, save_path, f1_score)

def save_metrics_to_file(metrics, save_path, f1_score):
    """保存评估指标到文件"""
    os.makedirs(save_path, exist_ok=True)
    metrics_file = os.path.join(save_path, "evaluation_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"mAP50-95: {metrics.AP}\n")
        f.write(f"mAP50: {metrics.AP50}\n")
        f.write(f"Mean Precision: {np.mean(metrics.precision)}\n")
        f.write(f"Recall: {np.mean(metrics.recall)}\n")
        f.write(f"F1 Score: {f1_score}\n")
        f.write("Class-wise AP:\n")
        for i, ap in enumerate(metrics.AP_per_class):
            f.write(f"Class {i}: AP = {ap}\n")
    print(f"评估指标已保存到 {metrics_file}")

if __name__ == '__main__':
    multiprocessing.freeze_support()  # 确保在 Windows 上正确支持多进程
    main()
