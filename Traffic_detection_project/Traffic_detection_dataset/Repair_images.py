from PIL import Image
import os
from pathlib import Path

def repair_corrupt_jpegs(directory, delete_corrupt=False):
    """
    尝试修复指定目录下的所有 JPEG 图像文件。
    :param directory: 数据集的根目录路径（包含 images 和 labels 文件夹）。
    :param delete_corrupt: 如果为 True，则删除无法修复的图像文件及其标注文件。
    """
    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')

    # 支持的图像格式
    image_extensions = ('.jpg', '.jpeg')

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(images_dir, filename)
            try:
                with Image.open(file_path) as img:
                    # 尝试保存图像，Pillow 会尝试修复损坏的 JPEG
                    img.save(file_path)
                    print(f"图像 {filename} 修复并保存成功。")
            except Exception as e:
                print(f"无法修复图像 {filename}: {e}")
                if delete_corrupt:
                    # 删除损坏的图像文件
                    os.remove(file_path)
                    print(f"删除损坏的图像文件 {filename}")

                    # 删除对应的标注文件
                    base_name = os.path.splitext(filename)[0]  # 去掉扩展名
                    label_filename = f"{base_name}.txt"
                    label_file_path = os.path.join(labels_dir, label_filename)
                    if os.path.exists(label_file_path):
                        os.remove(label_file_path)
                        print(f"删除对应的标注文件 {label_filename}")

# 数据集路径
train_images_dir = Path(__file__).resolve().parent / "dataset_file/Vehicle_Dataset_for_YOLO/test"
repair_corrupt_jpegs(train_images_dir, delete_corrupt=True)
