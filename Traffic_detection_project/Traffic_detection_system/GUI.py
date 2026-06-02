import sys
import os
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QSlider, QCheckBox, QScrollArea,
                             QMessageBox, QProgressBar, QTextEdit, QSplitter, QGroupBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from ultralytics import YOLO
from collections import defaultdict
import traceback

# 全局变量存储类别名称
CLASS_NAMES = {}


class DetectionWorker(QObject):
    """后台检测线程，采用moveToThread模式"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, image_path, model_path, conf_threshold, selected_classes):
        super().__init__()
        self.image_path = image_path
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.selected_classes = selected_classes
        self.is_running = True

    def run(self):
        """线程执行的入口函数"""
        print("[DEBUG] Worker thread started.")
        result_data = None
        try:
            if not self.is_running:
                raise Exception("线程被中止")

            self.progress.emit(10)
            print("[DEBUG] Loading model...")
            model = YOLO(self.model_path)
            global CLASS_NAMES
            CLASS_NAMES = model.names
            print(f"[DEBUG] Model loaded. Classes: {len(CLASS_NAMES)}")

            self.progress.emit(30)
            print(f"[DEBUG] Loading image: {self.image_path}")
            image = cv2.imread(self.image_path)
            if image is None:
                raise ValueError(f"无法加载图像，请检查路径: {self.image_path}")
            print(f"[DEBUG] Image loaded. Shape: {image.shape}")

            self.progress.emit(50)
            print(f"[DEBUG] Starting detection with conf: {self.conf_threshold}")
            start_time = time.time()
            results = model(image, conf=self.conf_threshold, verbose=False)
            end_time = time.time()
            inference_time = round((end_time - start_time) * 1000, 2)
            print(f"[DEBUG] Detection finished. Time: {inference_time}ms")

            self.progress.emit(70)
            print("[DEBUG] Processing results...")
            annotated_image, class_counts = self.process_results(results, image)
            total_objects = sum(class_counts.values())
            print(f"[DEBUG] Results processed. Total objects: {total_objects}")

            self.progress.emit(100)
            result_data = {
                "annotated_image": annotated_image,
                "inference_time": inference_time,
                "class_counts": class_counts,
                "total_objects": total_objects
            }

        except Exception as e:
            error_msg = f"检测过程中发生错误: {str(e)}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()
            self.error.emit(error_msg)
        finally:
            if self.is_running:
                self.finished.emit(result_data)
            print("[DEBUG] Worker thread finished.")

    def process_results(self, results, image):
        annotated_image = image.copy()
        class_counts = defaultdict(int)
        selected_set = set(self.selected_classes)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                original_class_name = CLASS_NAMES.get(class_id, f"未知类别_{class_id}")
                display_class_name = str(original_class_name)
                conf = float(box.conf[0])

                if display_class_name not in selected_set:
                    continue

                class_counts[display_class_name] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{display_class_name} {conf:.2f}"

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y1 = y1 - label_size[1] - 5
                if label_y1 < 0:
                    label_y1 = y1 + label_size[1] + 5
                cv2.rectangle(annotated_image, (x1, label_y1), (x1 + label_size[0], label_y1 + label_size[1]),
                              (0, 0, 0), -1)
                cv2.putText(annotated_image, label, (x1, label_y1 + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)
        return annotated_image, dict(class_counts)

    def stop(self):
        """停止线程"""
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 目标检测 GUI")
        self.setGeometry(100, 100, 1400, 800)

        self.image_path = None
        self.model_path = None
        self.detection_thread = None
        self.worker = None
        self.result_data = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        control_layout = QHBoxLayout()
        self.model_btn = QPushButton("选择权重文件(.pt)")
        self.model_btn.clicked.connect(self.select_model)
        control_layout.addWidget(self.model_btn)

        self.image_btn = QPushButton("选择图像")
        self.image_btn.clicked.connect(self.select_image)
        control_layout.addWidget(self.image_btn)

        self.detect_btn = QPushButton("开始检测")
        self.detect_btn.clicked.connect(self.start_detection)
        self.detect_btn.setEnabled(False)
        control_layout.addWidget(self.detect_btn)

        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        control_layout.addWidget(self.save_btn)

        control_layout.addStretch()
        main_layout.addLayout(control_layout)

        # --- 主要改动区域开始 ---
        config_layout = QHBoxLayout()
        conf_group = QGroupBox("置信度筛选")
        conf_form = QFormLayout(conf_group)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_label = QLabel("50%")
        self.conf_slider.valueChanged.connect(lambda x: self.conf_label.setText(f"{x}%"))
        conf_form.addRow(self.conf_label, self.conf_slider)  # 调换顺序，让标签在左
        config_layout.addWidget(conf_group)

        class_group = QGroupBox("类别筛选")
        class_group_layout = QVBoxLayout(class_group)  # 外层是垂直布局

        self.class_scroll = QScrollArea()
        self.class_widget = QWidget()
        # 将原来的 QVBoxLayout 改为 QHBoxLayout
        self.class_layout = QHBoxLayout(self.class_widget)
        self.class_layout.setAlignment(Qt.AlignLeft)  # 复选框向左对齐
        self.class_layout.setSpacing(15)  # 设置复选框之间的间距

        self.class_scroll.setWidget(self.class_widget)
        self.class_scroll.setWidgetResizable(True)
        # 设置滚动条策略，只在需要时显示水平滚动条
        self.class_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.class_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        class_group_layout.addWidget(self.class_scroll)
        config_layout.addWidget(class_group, stretch=1)
        # --- 主要改动区域结束 ---

        main_layout.addLayout(config_layout)

        image_layout = QHBoxLayout()
        self.original_label = QLabel("原图")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 1px solid #cccccc;")
        self.result_label = QLabel("检测结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 1px solid #cccccc;")
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.original_label)
        splitter.addWidget(self.result_label)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        image_layout.addWidget(splitter)
        main_layout.addLayout(image_layout, stretch=1)

        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        main_layout.addWidget(result_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择YOLO权重文件", "", "权重文件 (*.pt)")
        if path:
            self.model_path = path
            self.model_btn.setText(f"权重：{os.path.basename(path)}")
            try:
                model = YOLO(path)
                global CLASS_NAMES
                CLASS_NAMES = model.names
                self.update_class_checkboxes()
                self.detect_btn.setEnabled(bool(self.image_path))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载权重失败：{str(e)}")

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.image_path = path
            self.image_btn.setText(f"图像：{os.path.basename(path)}")
            self.display_image(path, self.original_label)
            self.detect_btn.setEnabled(bool(self.model_path))
            self.reset_results()

    def update_class_checkboxes(self):
        """更新类别复选框，现在是水平排列"""
        # 清空现有复选框
        for i in reversed(range(self.class_layout.count())):
            widget = self.class_layout.takeAt(i).widget()
            if widget:
                widget.deleteLater()
        # 添加新的复选框
        for class_id, class_name in CLASS_NAMES.items():
            display_name = str(class_name)
            checkbox = QCheckBox(display_name)
            checkbox.setChecked(True)
            self.class_layout.addWidget(checkbox)
        # 添加一个拉伸的空白widget，将所有复选框推到左边
        self.class_layout.addStretch()

    def get_selected_classes(self):
        selected = []
        for i in range(self.class_layout.count()):
            item = self.class_layout.itemAt(i)
            if item:
                widget = item.widget()
                if isinstance(widget, QCheckBox) and widget.isChecked():
                    selected.append(widget.text())
        return selected

    def start_detection(self):
        if not self.image_path or not self.model_path:
            QMessageBox.warning(self, "警告", "请先选择权重和图像")
            return

        if self.detection_thread and self.detection_thread.isRunning():
            self.worker.stop()
            self.detection_thread.quit()
            self.detection_thread.wait()

        conf_threshold = self.conf_slider.value() / 100.0
        selected_classes = self.get_selected_classes()

        if not selected_classes:
            QMessageBox.warning(self, "警告", "请至少选择一个类别")
            return

        self.detect_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.result_text.clear()
        self.result_label.setText("检测中...")

        self.detection_thread = QThread()
        self.worker = DetectionWorker(
            self.image_path,
            self.model_path,
            conf_threshold,
            selected_classes
        )
        self.worker.moveToThread(self.detection_thread)

        self.detection_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.error.connect(self.on_detection_error)
        self.worker.progress.connect(self.progress_bar.setValue)

        self.worker.finished.connect(self.detection_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.detection_thread.finished.connect(self.detection_thread.deleteLater)

        print("[DEBUG] Starting detection thread...")
        self.detection_thread.start()

    def on_detection_finished(self, result_data):
        print("[DEBUG] on_detection_finished signal received.")
        if result_data:
            self.result_data = result_data
            self.display_image(result_data["annotated_image"], self.result_label)
            self.update_result_text()
            self.save_btn.setEnabled(True)
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_detection_error(self, error_msg):
        print(f"[DEBUG] on_detection_error signal received: {error_msg}")
        QMessageBox.critical(self, "检测错误", error_msg)
        self.detect_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.result_label.setText("检测失败")

    def display_image(self, image, label):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        if img is None:
            label.setText("无法加载图像")
            return
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qt_img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_result_text(self):
        if not self.result_data:
            return
        self.result_text.clear()
        self.result_text.append("检测结果统计：")
        self.result_text.append(f"检测时间：{self.result_data['inference_time']} ms")
        self.result_text.append(f"总检测目标数：{self.result_data['total_objects']}")
        self.result_text.append("各类别数量：")
        for class_name, count in self.result_data["class_counts"].items():
            self.result_text.append(f"  - {class_name}: {count}")

    def save_results(self):
        if not self.result_data:
            QMessageBox.warning(self, "警告", "暂无检测结果可保存")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        img_save_path = os.path.join(save_dir, f"{base_name}_result_{timestamp}.jpg")
        cv2.imwrite(img_save_path, self.result_data["annotated_image"])
        txt_save_path = os.path.join(save_dir, f"{base_name}_result_{timestamp}.txt")
        with open(txt_save_path, "w", encoding="utf-8") as f:
            f.write("YOLO 目标检测结果\n")
            f.write(f"检测时间：{self.result_data['inference_time']} ms\n")
            f.write(f"置信度阈值：{self.conf_slider.value()}%\n")
            f.write(f"选中类别：{', '.join(self.get_selected_classes())}\n")
            f.write(f"总检测目标数：{self.result_data['total_objects']}\n")
            f.write("各类别数量：\n")
            for class_name, count in self.result_data["class_counts"].items():
                f.write(f"  - {class_name}: {count}\n")
        QMessageBox.information(self, "成功", f"结果已保存至：\n{save_dir}")

    def reset_results(self):
        self.result_data = None
        self.result_label.setText("检测结果")
        self.result_text.clear()
        self.save_btn.setEnabled(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.image_path:
            self.display_image(self.image_path, self.original_label)
        if self.result_data:
            self.display_image(self.result_data["annotated_image"], self.result_label)

    def closeEvent(self, event):
        """关闭窗口时停止线程"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.worker.stop()
            self.detection_thread.quit()
            self.detection_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())