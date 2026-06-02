import os
import random
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保可重复性
random.seed(42)

# 定义数据集路径
data_dir = Path(__file__).resolve().parent / 'dataset_file/Traffic_Detection_Dataset/dataset/xzhf2'
subsets = ['train', 'valid', 'test']

# 定义保存图像的目录
save_dir = data_dir / 'Data_distribution_map'
save_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录

# 初始化一个全局的DataFrame来存储所有子集的数据
all_data = []

def read_labels(image_files, images_dir, labels_dir):
    """读取标签文件并返回包含标签数据的DataFrame"""
    data = []
    for img_file in image_files:
        # 根据图像文件扩展名替换为对应的标签文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = labels_dir / label_file

        # 读取标签文件
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = line.strip().split()
                    data.append([float(val) for val in values])
        except FileNotFoundError:
            print(f"Label file {label_file} not found.")
            continue
    return pd.DataFrame(data, columns=['class', 'x', 'y', 'width', 'height'])

def plot_distributions(df, subset, save_dir):
    """绘制数据分布图并保存到指定目录"""
    # 散点图和直方图
    sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.5})
    plt.savefig(save_dir / f'{subset}_pairplot.png')
    plt.close()

    # 类别分布柱状图
    plt.figure(figsize=(8, 6))
    class_counts = df['class'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='coolwarm', legend=False)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(f'{subset} Class Distribution')
    plt.savefig(save_dir / f'{subset}_class_distribution.png')
    plt.close()

    # 显示图像分布
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='x', y='y', data=df, alpha=0.5)
    plt.title(f'{subset} Image Distribution')
    plt.savefig(save_dir / f'{subset}_image_distribution.png')
    plt.close()

    # 显示宽高分布
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='width', y='height', data=df, alpha=0.5)
    max_dim = max(df['width'].max(), df['height'].max())  # 获取宽高的最大值
    plt.plot([0, max_dim], [0, max_dim], linestyle='--', color='gray')  # 添加对角线参考线
    plt.title(f'{subset} Width vs Height')
    plt.savefig(save_dir / f'{subset}_width_height_distribution.png')
    plt.close()

# 遍历每个子集
for subset in subsets:
    images_dir = data_dir / subset / 'images'
    labels_dir = data_dir / subset / 'labels'

    # 获取所有图像文件
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # 读取标签信息并存储在DataFrame中
    df = read_labels(image_files, images_dir, labels_dir)
    df['subset'] = subset  # 添加子集信息
    all_data.append(df)

    # 绘制数据分布图
    plot_distributions(df, subset, save_dir)

    print(f"Processing {subset} subset...")
    print(f"Found {len(image_files)} images and {len(df)} labels.")
    print(f"{subset} 数据分布图已保存到指定目录。")

# 合并所有子集的数据
all_df = pd.concat(all_data, ignore_index=True)

# 绘制整个数据集的数据分布图
plot_distributions(all_df, 'all', save_dir)

print("整个数据集的数据分布图已保存到指定目录。")
