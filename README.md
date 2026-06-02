# CV Traffic Detect YOLOv8

基于 Ultralytics YOLOv8 的交通目标检测项目，支持交通场景中的车辆、行人和两轮车等目标检测，并提供训练脚本、模型结构配置、评估辅助脚本和 PyQt5 可视化检测系统。

本仓库基于 Ultralytics 源码进行二次开发，重点改动集中在交通目标检测任务适配、YOLOv8 结构改进实验、训练配置和可视化系统实现。

## 项目功能

- 使用 YOLOv8 进行交通目标检测。
- 支持 5 类目标：`Car`、`Truck`、`Person`、`Two wheeled vehicle`、`Bus`。
- 提供 PyQt5 GUI，可选择图片或视频进行检测。
- 支持切换权重文件、调节置信度阈值、显示类别/置信度/速度信息。
- 支持检测结果保存、检测报告生成和数据库写入。
- 提供数据划分、数据增强、数据分布可视化、实验结果对比和热力图脚本。
- 包含多种 YOLOv8 改进模型配置，例如 CBAM、ECA、CoordAttention、DCNv2、P2 检测头等。

## 目录结构

```text
.
├── Traffic_detection_project/
│   ├── Traffic_detection_dataset/      # 数据处理与分析脚本，数据集本体不提交
│   ├── Traffic_detection_system/       # PyQt5 可视化检测系统
│   └── Traffic_detection_train/        # 训练脚本、数据集 YAML、训练任务脚本
├── ultralytics/                        # Ultralytics 核心源码与改进模型配置
├── img/car/                            # 少量演示图片
├── label/car-test/                     # 少量演示标签
├── heatmap_script.py                   # 检测热力图脚本
├── export.py                           # 权重导出脚本
└── oldcar_train.py                     # 示例训练脚本
```

## 未提交内容

为避免仓库过大，本项目没有提交以下内容：

- 完整数据集：`Traffic_detection_project/Traffic_detection_dataset/dataset_file/`
- 测试图片和视频：`Traffic_detection_project/Traffic_detection_dataset/images_videos_test/`
- 训练结果：`Traffic_detection_project/Traffic_detection_train/results/`
- GUI 检测输出：`Traffic_detection_project/Traffic_detection_system/Target_detection_results/`
- 违章截图输出：`Traffic_detection_project/Traffic_detection_system/Traffic_violation_screenshot/`
- 模型权重和导出模型：`*.pt`、`*.onnx`、`*.engine` 等

这些内容已通过 `.gitignore` 排除。若需要复现实验，请自行下载或准备数据集和权重文件，并按下面路径放置。

## 数据集放置

训练配置文件位于：

```text
Traffic_detection_project/Traffic_detection_train/Traffic_Detection_Dataset.yaml
```

默认要求数据集放置为：

```text
Traffic_detection_project/
├── Traffic_detection_dataset/
│   └── dataset_file/
│       └── roboflow_721_x3/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           ├── valid/
│           │   ├── images/
│           │   └── labels/
│           └── test/
│               ├── images/
│               └── labels/
```

类别配置：

```yaml
nc: 5
names: ['Car', 'Truck', 'Person', 'Two wheeled vehicle', 'Bus']
```

## 权重放置

GUI 默认从以下目录读取权重：

```text
Traffic_detection_project/Traffic_detection_system/best_weights/
```

默认模型文件名：

```text
Traffic_detection_YOLOv8n-baseline.pt
```

如果没有该文件，可以在 GUI 中手动选择其他 `.pt` 权重，或将自己的权重文件放到上述目录。

## 环境准备

建议使用 Python 3.8+。项目依赖 Ultralytics、PyTorch、OpenCV、PyQt5、FilterPy、pyodbc 等库。

```bash
pip install -e .
pip install pyqt5 opencv-python filterpy pyodbc prettytable seaborn albumentations
```

PyTorch 建议根据本机 CUDA 版本从官方渠道安装。

## 训练模型

在仓库根目录执行：

```bash
python Traffic_detection_project/Traffic_detection_train/Traffic_Detection_Dataset.py
```

训练脚本会读取：

```text
Traffic_detection_project/Traffic_detection_train/Traffic_Detection_Dataset.yaml
```

训练输出默认保存到：

```text
Traffic_detection_project/Traffic_detection_train/results/
```

该目录不会提交到 GitHub。

## 运行检测系统

进入检测系统目录：

```bash
cd Traffic_detection_project/Traffic_detection_system
python main.py
```

系统启动后可以：

- 选择图片或视频文件。
- 切换模型权重。
- 调整置信度阈值。
- 保存检测结果和报告。
- 查看检测记录和违章截图。

测试媒体默认目录：

```text
Traffic_detection_project/Traffic_detection_dataset/images_videos_test/
```

该目录不随仓库提交，需要自行放置测试图片或视频。

## 辅助脚本

- `Traffic_detection_project/Traffic_detection_dataset/Dataset_division.py`：数据集划分。
- `Traffic_detection_project/Traffic_detection_dataset/Data_enhancement.py`：数据增强。
- `Traffic_detection_project/Traffic_detection_dataset/Draw_data_distribution_map.py`：数据分布可视化。
- `Traffic_detection_project/Traffic_detection_dataset/Paper_Data_Script.py`：论文实验指标汇总。
- `Traffic_detection_project/Traffic_detection_dataset/Result_comparison_chart(合并画).py`：实验曲线合并绘图。
- `Traffic_detection_project/Traffic_detection_dataset/Result_comparison_chart(分开画).py`：实验曲线分开绘图。
- `heatmap_script.py`：检测热力图生成。
- `export.py`：模型导出为 TensorRT engine。

## 说明

本项目保留了 Ultralytics 源码结构，并在其基础上加入交通检测相关配置、训练脚本和 GUI 系统。完整数据集、训练输出和模型权重未提交到仓库，使用时需要按 README 中的目录结构自行补齐。
