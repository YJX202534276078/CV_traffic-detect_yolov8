import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

# 定义路径
DATASET_DIR = Path(__file__).resolve().parent / "dataset_file"
base_dir = str(DATASET_DIR / "roboflow_721_5880")
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# 定义输出文件夹
output_dir = os.path.join(base_dir, "split_dataset")
train_dir = os.path.join(output_dir, "train")
valid_dir = os.path.join(output_dir, "valid")
test_dir = os.path.join(output_dir, "test")

# 创建输出文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有图像文件名（不带扩展名）
image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 检查图像文件和标签文件是否匹配
valid_image_files = []
for file_name in image_files:
    label_file = os.path.join(labels_dir, file_name + ".txt")
    if os.path.exists(label_file):
        valid_image_files.append(file_name)

# 更新数据集总数
total_images = len(valid_image_files)
print(f"数据集总数: {total_images} 张照片")

# 设置随机种子
random.seed(42)

# 打乱数据集
random.shuffle(valid_image_files)

# 定义划分比例
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

# 计算每个集合的样本数量
train_split = int(total_images * train_ratio)
valid_split = int(total_images * valid_ratio)

# 分配样本
train_files = valid_image_files[:train_split]
valid_files = valid_image_files[train_split:train_split + valid_split]
test_files = valid_image_files[train_split + valid_split:]

# 复制图像和标签到对应的文件夹
def copy_files(file_list, output_folder):
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "labels"), exist_ok=True)
    for file_name in file_list:
        # 复制图像文件
        for ext in ['.jpg', '.png', '.jpeg']:
            src_image = os.path.join(images_dir, file_name + ext)
            if os.path.exists(src_image):
                shutil.copy(src_image, os.path.join(output_folder, "images"))
                break
        # 复制标签文件
        src_label = os.path.join(labels_dir, file_name + ".txt")
        if os.path.exists(src_label):
            shutil.copy(src_label, os.path.join(output_folder, "labels"))

# 复制训练集
copy_files(train_files, train_dir)

# 复制验证集
copy_files(valid_files, valid_dir)

# 复制测试集
copy_files(test_files, test_dir)

# 打印结果
print("数据集划分完成！")
print(f"训练集: {len(train_files)} 个样本")
print(f"验证集: {len(valid_files)} 个样本")
print(f"测试集: {len(test_files)} 个样本")

# 统计每个类别的分布
def count_class_distribution(file_list):
    class_count = defaultdict(int)
    for file_name in file_list:
        label_file = os.path.join(labels_dir, file_name + ".txt")
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_count[class_id] += 1
    return class_count

# 打印训练集类别分布
print("\n训练集类别分布:")
train_class_count = count_class_distribution(train_files)
for class_id, count in sorted(train_class_count.items()):
    print(f"类别 {class_id}: {count} 个样本")

# 打印验证集类别分布
print("\n验证集类别分布:")
valid_class_count = count_class_distribution(valid_files)
for class_id, count in sorted(valid_class_count.items()):
    print(f"类别 {class_id}: {count} 个样本")

# 打印测试集类别分布
print("\n测试集类别分布:")
test_class_count = count_class_distribution(test_files)
for class_id, count in sorted(test_class_count.items()):
    print(f"类别 {class_id}: {count} 个样本")
