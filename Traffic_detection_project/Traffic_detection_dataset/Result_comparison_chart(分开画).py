# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'Traffic_detection_train/results'

# 训练结果列表
results_files = [
    str(RESULTS_DIR / 'Traffic_detection_yolov8n（baseline）/results.csv'),
    str(RESULTS_DIR / 'Traffic_detection_P2_CBAM（最终优化模型YOLOv8n-Optimize）/results.csv'),

]

# 与results_files顺序对应
custom_labels = [
    'YOLOv8n',
    'YOLOv8n-Optimize',


]
def plot_metric_comparison(metric_key, metric_label,custom_labels):
    plt.figure(figsize=(10, 6))

    for file_path, custom_label in zip(results_files, custom_labels):
        exp_name = os.path.basename(os.path.dirname(file_path))
        df = pd.read_csv(file_path)

        # 清理列名中的多余空格
        df.columns = df.columns.str.strip()

        # 检查 'epoch' 列是否存在
        if 'epoch' not in df.columns:
            print(f"'epoch' column not found in {file_path}. Available columns: {df.columns}")
            continue

        # 检查目标指标列是否存在
        if metric_key not in df.columns:
            print(f"'{metric_key}' column not found in {file_path}. Available columns: {df.columns}")
            continue

        # 绘制没有圆点的线条
        plt.plot(df['epoch'], df[metric_key], label=f'{custom_label}')

    plt.title(f'{metric_label} ')
    plt.xlabel('Epochs')
    plt.ylabel(metric_label)
    plt.legend()
    plt.show()




if __name__ == '__main__':

    metrics = [
        ('metrics/precision(B)', 'Precision'),
        ('metrics/recall(B)', 'Recall'),
        ('metrics/mAP50(B)', 'mAP@50'),
        ('metrics/mAP50-95(B)', 'mAP@50-95')
    ]

    for metric, label in metrics:
        plot_metric_comparison(metric, label, custom_labels)

    loss_metrics = [
        ('train/box_loss', 'Train Box Loss'),
        ('train/cls_loss', 'Train Class Loss'),
        ('train/dfl_loss', 'Train DFL Loss'),
        ('val/box_loss', 'Val Box Loss'),
        ('val/cls_loss', 'Val Class Loss'),
        ('val/dfl_loss', 'Val DFL Loss')
    ]

    for metric, label in loss_metrics:
        plot_metric_comparison(metric, label, custom_labels)
