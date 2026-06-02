# main_window
import os
import cv2
import torch
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, QComboBox, QMessageBox,
    QTextEdit, QHBoxLayout, QDesktopWidget, QFrame, QSlider, QTableWidgetItem, QHeaderView, QTableWidget,
    QDialog, QDateEdit, QDoubleSpinBox
)
from PyQt5.QtGui import QIcon, QTextCursor, QColor
from PyQt5.QtCore import QTimer, Qt, QMutex, QDate
from database import connect_to_database, insert_violation_record
from object_detection import ObjectDetection
from violation_management import ViolationManagementDialog
from utils import get_color, display_image
from config import DEFAULT_MEDIA_DIR, DEFAULT_RESULTS_DIR, DEFAULT_VIOLATION_SCREENSHOT_DIR, DEFAULT_WEIGHTS_DIR
from collections import defaultdict
from filterpy.kalman import KalmanFilter
import time

class DatabaseViewDialog(QDialog):
    def __init__(self, records, conn, parent=None):
        super().__init__(parent)
        self.conn = conn
        self.original_records = records  # 保存原始记录
        self.filtered_records = records.copy()  # 当前筛选后的记录
        self.setWindowTitle("数据库检测结果")
        self.setGeometry(200, 200, 1200, 800)
        self.initUI()
        self.center()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def initUI(self):
        layout = QVBoxLayout()

        # 筛选条件布局
        filter_layout = QHBoxLayout()

        # 类别筛选下拉列表
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("所有类别")
        self.combo_box.addItems(self.get_unique_classes(self.original_records))
        self.combo_box.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.combo_box)

        # 置信度范围筛选
        filter_layout.addWidget(QLabel("最小置信度:", self))
        self.min_confidence_spin = QDoubleSpinBox(self)
        self.min_confidence_spin.setRange(0, 1)
        self.min_confidence_spin.setSingleStep(0.01)
        self.min_confidence_spin.setValue(0.5)
        self.min_confidence_spin.valueChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.min_confidence_spin)

        filter_layout.addWidget(QLabel("最大置信度:", self))
        self.max_confidence_spin = QDoubleSpinBox(self)
        self.max_confidence_spin.setRange(0, 1)
        self.max_confidence_spin.setSingleStep(0.01)
        self.max_confidence_spin.setValue(1.0)
        self.max_confidence_spin.valueChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.max_confidence_spin)

        # 日期范围筛选
        filter_layout.addWidget(QLabel("开始日期:", self))
        self.start_date_edit = QDateEdit(self)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))  # 默认7天前
        self.start_date_edit.dateChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.start_date_edit)

        filter_layout.addWidget(QLabel("结束日期:", self))
        self.end_date_edit = QDateEdit(self)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.dateChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.end_date_edit)

        # 重置筛选按钮
        self.reset_filter_btn = QPushButton("重置筛选", self)
        self.reset_filter_btn.clicked.connect(self.reset_filters)
        filter_layout.addWidget(self.reset_filter_btn)

        layout.addLayout(filter_layout)

        # 表格显示数据
        self.table = QTableWidget(self)
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["ID", "类别", "置信度", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "检测时间"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSortingEnabled(True)  # 启用排序

        # 连接表头点击信号以实现排序
        self.table.horizontalHeader().sectionClicked.connect(self.on_header_clicked)

        # 填充数据
        self.populate_table(self.filtered_records)

        layout.addWidget(self.table)

        # 关闭按钮
        self.btn_close = QPushButton("关闭", self)
        self.btn_close.clicked.connect(self.close)
        layout.addWidget(self.btn_close)

        self.setLayout(layout)

    def on_header_clicked(self, column):
        # 获取当前排序顺序
        sort_order = self.table.horizontalHeader().sortIndicatorOrder()

        # 反转排序顺序
        if sort_order == Qt.AscendingOrder:
            new_order = Qt.DescendingOrder
        else:
            new_order = Qt.AscendingOrder

        self.table.sortByColumn(column, new_order)

    def get_unique_classes(self, records):
        classes = set()
        for record in records:
            classes.add(record[1])
        return list(classes)

    def populate_table(self, records):
        self.table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.table.setItem(i, 0, QTableWidgetItem(str(record[0])))  # ID
            self.table.setItem(i, 1, QTableWidgetItem(record[1]))  # 类别
            self.table.setItem(i, 2, QTableWidgetItem(f"{record[2]:.2f}"))  # 置信度
            self.table.setItem(i, 3, QTableWidgetItem(str(record[3])))  # bbox_x1
            self.table.setItem(i, 4, QTableWidgetItem(str(record[4])))  # bbox_y1
            self.table.setItem(i, 5, QTableWidgetItem(str(record[5])))  # bbox_x2
            self.table.setItem(i, 6, QTableWidgetItem(str(record[6])))  # bbox_y2
            self.table.setItem(i, 7, QTableWidgetItem(str(record[7])))  # 检测时间

    def apply_filters(self):
        selected_class = self.combo_box.currentText()
        min_confidence = self.min_confidence_spin.value()
        max_confidence = self.max_confidence_spin.value()
        start_date = self.start_date_edit.date().toPyDate()
        end_date = self.end_date_edit.date().toPyDate()

        # 筛选记录
        filtered = []
        for record in self.original_records:
            # 检查类别
            if selected_class != "所有类别" and record[1] != selected_class:
                continue

            # 检查置信度
            confidence = record[2]
            if not (min_confidence <= confidence <= max_confidence):
                continue

            # 检查日期
            record_date = record[7].date()
            if not (start_date <= record_date <= end_date):
                continue

            filtered.append(record)

        self.filtered_records = filtered
        self.populate_table(self.filtered_records)

    # 重置所有筛选条件
    def reset_filters(self):
        self.combo_box.setCurrentIndex(0)
        self.min_confidence_spin.setValue(0.5)
        self.max_confidence_spin.setValue(1.0)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-7))
        self.end_date_edit.setDate(QDate.currentDate())
        self.filtered_records = self.original_records.copy()
        self.populate_table(self.filtered_records)

# 车辆速度检测算法
class SpeedEstimator:
    def __init__(self):
        self.trackers = defaultdict(self.create_kalman_filter)
        self.pixel_to_meter = self.get_pixel_to_meter_ratio()

    def create_kalman_filter(self):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])  # 状态转移矩阵
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 1000  # 协方差矩阵
        kf.R = np.eye(2) * 10
        kf.Q = np.eye(4) * 0.1
        return kf

    def get_pixel_to_meter_ratio(self):
        actual_length_meters = 3
        pixel_length = 300 # 车道线的实际长度约3米，在视频中的像素长度是300像素
        return actual_length_meters / pixel_length

    def estimate_speed(self, track_id, current_position, fps):
        kf = self.trackers[track_id]
        predicted = kf.predict()
        kf.update(current_position)

        velocity = np.linalg.norm(kf.x[2:]) * self.pixel_to_meter * fps
        return velocity

class TrafficMonitoringSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.mutex = QMutex()
        self.initUI()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        default_model_path = os.path.join(DEFAULT_WEIGHTS_DIR, "Traffic_detection_YOLOv8n-baseline.pt")
        self.model = ObjectDetection(default_model_path, self.device)
        self.conn = connect_to_database()
        self.class_names = self.model.class_names
        self.selected_class = None
        self.current_frame = None
        self.class_counts = {}
        self.is_detection_enabled = True
        self.is_video_paused = False
        self.show_class = True
        self.show_speed = False
        self.show_confidence = True
        self.is_image = False
        self.selected_box = None
        self.boxes = []
        self.current_model_path = default_model_path  # 初始权重文件
        self.tracked_objects = {}
        self.speed_estimator = SpeedEstimator()
        self.show_original_image = True
        self.last_displayed_speed = {}
        self.confidence_threshold = 0.5  # 初始置信度阈值0.5
        self.camera_active = False
        self.font_scale = 1.1 # 检测框字体大小
        self.thickness = 2 # 检测框字体粗细

    def initUI(self):
        self.setWindowTitle("基于YOLOv8n的交通目标检测系统_余骏轩_202534276078")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon("icon.png"))

        self.center()

        self.loading_label = QLabel("加载中，请稍候...", self)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size: 24px; color: #ECEFF4;")
        self.setCentralWidget(self.loading_label)

        QTimer.singleShot(100, self.load_model)

        self.showMaximized()

    # 延迟加载模型
    def load_model(self):
        self.class_names = self.model.class_names
        self.init_main_ui()

    def init_main_ui(self):
        self.loading_label.hide()

        # 全局样式表
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2E3440;  /* 背景颜色 */
            }
            QLabel {
                color: #ECEFF4;  /* 文字颜色 */
                font-size: 16px;
            }
            QPushButton {
                background-color: #4C566A;
                color: #ECEFF4;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
                border: 2px solid #3B4252;
            }
            QPushButton:hover {
                background-color: #5E81AC;
                border: 2px solid #88C0D0;
            }
            QPushButton:pressed {
                background-color: #81A1C1;
                border: 2px solid #88C0D0;
            }
            QComboBox {
                background-color: #4C566A;
                color: #ECEFF4;
                font-size: 14px;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QMessageBox {
                background-color: #2E3440;
                color: #ECEFF4;
                font-size: 16px;
            }
            QMessageBox QLabel {
                color: #ECEFF4;
                font-size: 16px;
            }
            QMessageBox QPushButton {
                background-color: #4C566A;
                color: #ECEFF4;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QMessageBox QPushButton:hover {
                background-color: #5E81AC;
            }
            QTextEdit {
                background-color: #3B4252;
                color: #ECEFF4;
                font-size: 16px;
                border: none;
                padding: 10px;
            }
            QSlider::groove:horizontal {
                background-color: #4C566A;
                height: 8px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background-color: #81A1C1;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background-color: #5E81AC;
                border-radius: 4px;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout()

        # 按钮布局
        button_layout = QHBoxLayout()
        self.btn_select = QPushButton("选择照片或视频", self)
        self.btn_select.clicked.connect(self.select_file)
        button_layout.addWidget(self.btn_select)

        # 选择筛选类别
        self.combo_box = QComboBox(self)
        self.combo_box.addItem("所有类别")
        self.combo_box.addItems(self.class_names.values())
        self.combo_box.currentTextChanged.connect(self.filter_class)
        button_layout.addWidget(self.combo_box)

        # 置信度阈值滑动条
        self.confidence_slider = QSlider(Qt.Horizontal, self)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(50)  # 默认值50对应0.5
        self.confidence_slider.setFixedWidth(200)
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        button_layout.addWidget(self.confidence_slider)

        # 置信度阈值标签
        self.confidence_label = QLabel("置信度阈值: 0.50", self)
        self.confidence_label.setStyleSheet("font-size: 15px; color: #ECEFF4; margin: 0px;")
        button_layout.addWidget(self.confidence_label)

        self.btn_switch_model = QPushButton("更换权重文件", self)
        self.btn_switch_model.clicked.connect(self.switch_model)
        self.btn_switch_model.setStyleSheet("margin: 0px;")
        button_layout.addWidget(self.btn_switch_model)

        self.btn_pause_video = QPushButton("暂停视频", self)
        self.btn_pause_video.clicked.connect(self.toggle_pause_video)
        button_layout.addWidget(self.btn_pause_video)

        self.btn_view_violations = QPushButton("查看违法记录", self)
        self.btn_view_violations.clicked.connect(self.view_violations)
        button_layout.addWidget(self.btn_view_violations)

        self.btn_view_screenshots = QPushButton("查看违法目标截图", self)
        self.btn_view_screenshots.clicked.connect(self.view_violation_screenshots)
        button_layout.addWidget(self.btn_view_screenshots)

        # 目标选择下拉列表
        self.target_combo = QComboBox(self)
        self.target_combo.setStyleSheet("background-color: #4C566A; color: #ECEFF4;")
        self.target_combo.setFixedWidth(150)
        self.target_combo.currentIndexChanged.connect(self.select_target_by_id)
        button_layout.addWidget(self.target_combo)

        # 设置按钮布局的间距
        button_layout.setSpacing(5)

        main_layout.addLayout(button_layout)

        horizontal_layout = QHBoxLayout()

        # 显示当前权重文件
        self.label_model = QLabel(f"当前权重文件: {os.path.basename(self.current_model_path)}", self)
        self.label_model.setStyleSheet("font-size: 18px; color: #88C0D0;")

        # 显示当前模式
        self.label_mode = QLabel("", self)  # 初始化为空
        self.label_mode.setStyleSheet("font-size: 18px; color: #88C0D0;")

        # 显示当前视频实时帧率
        self.label_fps = QLabel("当前视频帧率: 0.00 FPS", self)
        self.label_fps.setStyleSheet("font-size: 18px; color: #88C0D0;")

        horizontal_layout.addWidget(self.label_model)  # 权重文件标签
        horizontal_layout.addStretch(1)
        horizontal_layout.addWidget(self.label_mode)  # 检测模式标签
        horizontal_layout.addStretch(1)
        horizontal_layout.addWidget(self.label_fps)  # 帧率标签

        horizontal_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addLayout(horizontal_layout)

        # 更新模式信息
        self.update_mode_display()

        # 图像显示区域
        self.label_image = QLabel(self)
        self.label_image.setAlignment(Qt.AlignCenter)
        self.label_image.setStyleSheet("""
            QLabel {
                background-color: #3B4252;
                color: #ECEFF4;
                font-size: 24px;
                qproperty-alignment: 'AlignCenter';
            }
        """)
        self.label_image.setText("请点击左上角选择要检测的照片或视频\n或开启摄像头进行实时检测")
        self.label_image.setScaledContents(False)
        main_layout.addWidget(self.label_image, stretch=1)

        # 功能按钮
        function_button_layout = QHBoxLayout()
        self.btn_toggle_speed = QPushButton("开启速度估算功能", self)
        self.btn_toggle_speed.clicked.connect(self.toggle_speed_display)
        function_button_layout.addWidget(self.btn_toggle_speed)

        self.btn_toggle_camera = QPushButton("打开摄像头实时检测", self)
        self.btn_toggle_camera.clicked.connect(self.toggle_camera_detection)
        function_button_layout.addWidget(self.btn_toggle_camera)

        self.btn_save_results = QPushButton("保存检测结果", self)
        self.btn_save_results.clicked.connect(self.save_detection_results)
        function_button_layout.addWidget(self.btn_save_results)

        self.btn_view_reports = QPushButton("查看检测结果报告", self)
        self.btn_view_reports.clicked.connect(self.view_detection_reports)
        function_button_layout.addWidget(self.btn_view_reports)

        self.btn_view_database = QPushButton("查看数据库检测结果", self)
        self.btn_view_database.clicked.connect(self.view_detection_database)
        function_button_layout.addWidget(self.btn_view_database)

        main_layout.addLayout(function_button_layout)

        horizontal_layout = QHBoxLayout()

        frame_count = QFrame(self)
        layout_count = QVBoxLayout()
        self.label_count = QLabel("检测到的目标数量: 0", self)
        self.label_count.setStyleSheet("font-size: 20px; color: #88C0D0;")
        layout_count.addWidget(self.label_count)

        # 检测用时标签
        self.label_detection_time = QLabel("检测用时: 0.00 秒", self)
        self.label_detection_time.setStyleSheet("font-size: 20px; color: #88C0D0;")
        layout_count.addWidget(self.label_detection_time)

        frame_count.setLayout(layout_count)
        frame_count.setStyleSheet("border: 1px solid #88C0D0;")

        # 文本显示框
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setFixedHeight(100)

        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #3B4252;  /* 背景颜色 */
                color: #ECEFF4;            /* 文本颜色 */
                border: none;              /* 无边框 */
                padding: 10px;             /* 内边距 */
            }
            QTextEdit QWidget {
                border: 1px solid #88C0D0; /* 外边框 */
            }
        """)

        self.text_edit.setLineWrapMode(QTextEdit.WidgetWidth)

        frame_info = QFrame(self)
        layout_info = QVBoxLayout()
        self.label_selected_info = QLabel("选中目标的详细信息将显示在这里", self)
        self.label_selected_info.setStyleSheet("font-size: 16px; color: #88C0D0;")
        layout_info.addWidget(self.label_selected_info)
        frame_info.setLayout(layout_info)
        frame_info.setStyleSheet("border: 1px solid #88C0D0;")

        horizontal_layout.addWidget(frame_count)
        horizontal_layout.addWidget(self.text_edit)
        horizontal_layout.addWidget(frame_info)

        # 设置每个部件的伸缩比例为1:2:1
        horizontal_layout.setStretchFactor(frame_count, 1)
        horizontal_layout.setStretchFactor(self.text_edit, 2)
        horizontal_layout.setStretchFactor(frame_info, 1)

        main_layout.addLayout(horizontal_layout)

        # 违法标记按钮
        self.btn_mark_violation = QPushButton("标记为违停", self)
        self.btn_mark_violation.clicked.connect(lambda: self.mark_violation("违停"))
        main_layout.addWidget(self.btn_mark_violation)

        self.btn_mark_speeding = QPushButton("标记为超速", self)
        self.btn_mark_speeding.clicked.connect(lambda: self.mark_violation("超速"))
        main_layout.addWidget(self.btn_mark_speeding)

        # 主窗口布局
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # 更新目标选择下拉列表
    def update_target_combo(self):
        self.target_combo.clear()
        self.target_combo.addItem("未选择目标")

        # 只有在视频或照片模式下才显示目标列表
        if hasattr(self, 'boxes') and self.boxes:
            for box in self.boxes:
                self.target_combo.addItem(f"ID: {box['track_id']} - {box['class_name']}")

    # 通过ID选择目标
    def select_target_by_id(self, index):
        print(f"Selected index: {index}, current boxes: {len(self.boxes) if hasattr(self, 'boxes') else 0}")  # 调试信息

        if index == 0:  # "未选择目标"选项
            self.selected_box = None
            self.label_selected_info.setText("未选中目标")
            self.update_display_boxes()
            return

        # 获取选中的目标ID
        selected_text = self.target_combo.currentText()
        print(f"Selected text: {selected_text}")  # 调试信息

        try:
            target_id = int(selected_text.split("ID: ")[1].split(" - ")[0])
        except (IndexError, ValueError):
            print("Failed to parse target ID")  # 调试信息
            return

        # 查找对应的目标
        found = False
        for box in self.boxes:
            if box['track_id'] == target_id:
                self.selected_box = box
                found = True
                break

        if not found:
            print(f"Target ID {target_id} not found in boxes")  # 调试信息
            self.selected_box = None

        if self.selected_box:
            track_id = self.selected_box['track_id']
            speed_meter_per_sec = self.last_displayed_speed.get(track_id, 0)
            if self.is_image:
                speed_meter_per_sec = 0
            self.label_selected_info.setText(
                f"ID: {self.selected_box['track_id']}\n"
                f"目标类别: {self.selected_box['class_name']}\n"
                f"置信度: {self.selected_box['confidence']:.2f}\n"
                f"速度: {speed_meter_per_sec:.2f} m/s"
            )
        else:
            self.label_selected_info.setText("未选中目标")

        self.update_display_boxes()

    # 窗口居中显示
    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2,
            (screen.height() - size.height()) // 2
        )

    # 摄像头实时检测
    def toggle_camera_detection(self):
        if hasattr(self, 'camera_active') and self.camera_active:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if self.timer.isActive():
                self.timer.stop()
            self.camera_active = False
            self.btn_toggle_camera.setText("打开摄像头检测")
            self.update_mode_display("")

            # 清除显示画面
            self.label_image.clear()
            self.label_count.setText("检测到的目标数量: 0")
            self.label_detection_time.setText("检测用时: 0.00 秒")
            self.text_edit.clear()
            self.label_selected_info.setText("选中目标的详细信息将显示在这里")
        else:
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)  # 默认摄像头
            if not self.cap.isOpened():
                QMessageBox.warning(self, "摄像头错误", "无法打开摄像头！")
                return

            self.camera_active = True
            self.is_image = False
            self.btn_toggle_camera.setText("关闭摄像头检测")
            self.update_mode_display("摄像头实时检测")

            # 启动定时器
            self.timer.start(50)

    def save_detection_results(self):
        # 检查是否有可保存的帧（照片或视频模式）
        if not hasattr(self, 'current_frame') and not hasattr(self, 'predicted_frame'):
            QMessageBox.warning(self, "警告", "没有检测到的帧可供保存！")
            return

        # 指定保存路径
        base_save_path = DEFAULT_RESULTS_DIR
        if not os.path.exists(base_save_path):
            os.makedirs(base_save_path)

        # 获取当前时间
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 创建子文件夹
        subfolder_name = f"result{len(os.listdir(base_save_path)) + 1}"
        save_folder = os.path.join(base_save_path, subfolder_name)
        os.makedirs(save_folder)

        # 根据模式选择要保存的帧
        if self.is_image:
            # 照片模式：保存预测帧
            frame_to_save = self.predicted_frame if hasattr(self, 'predicted_frame') else self.current_frame
            save_image_path = os.path.join(save_folder, f"detection_result_{current_time}.jpg")

            # 绘制检测框到图片上
            frame_with_boxes = frame_to_save.copy()
            for box in self.boxes:
                x1, y1, x2, y2 = box["bbox"]
                class_name = box["class_name"]
                confidence = box["confidence"]
                track_id = box["track_id"]
                color = get_color(class_name)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id} {class_name} {confidence:.2f}"
                cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)
        else:
            # 视频模式：保存当前帧
            frame_to_save = self.current_frame
            save_image_path = os.path.join(save_folder, f"detection_result_frame_{current_time}.jpg")

            # 绘制检测框到帧上
            frame_with_boxes = frame_to_save.copy()
            for box in self.boxes:
                x1, y1, x2, y2 = box["bbox"]
                class_name = box["class_name"]
                confidence = box["confidence"]
                track_id = box["track_id"]
                color = get_color(class_name)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {track_id} {class_name} {confidence:.2f}"
                cv2.putText(frame_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)
                if not self.is_image and self.show_speed:
                    speed = self.last_displayed_speed.get(track_id, 0)
                    cv2.putText(frame_with_boxes, f"Speed: {speed:.2f} m/s",
                                (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

        # 保存图片
        cv2.imwrite(save_image_path, frame_with_boxes)

        # 获取检测用时
        detection_time_text = self.label_detection_time.text()
        detection_time = detection_time_text.split(": ")[1].split(" ")[0] if ":" in detection_time_text else "0.00"

        # 生成检测结果报告
        report_path = os.path.join(save_folder, f"detection_report_{current_time}.txt")
        with open(report_path, "w", encoding='utf-8') as report_file:
            report_file.write(f"检测时间: {current_time}\n")
            report_file.write(f"检测模式: {'照片' if self.is_image else '视频'}\n")
            report_file.write(f"使用的模型: {os.path.basename(self.current_model_path)}\n")
            report_file.write(f"检测到的目标数量: {sum(self.class_counts.values())}\n")
            report_file.write(f"检测用时: {detection_time} 秒\n\n")

            report_file.write("=== 按类别统计 ===\n")
            for class_name, count in self.class_counts.items():
                if count > 0:
                    report_file.write(f"{class_name}: {count}个\n")

            report_file.write("\n=== 详细检测结果 ===\n")
            for box in self.boxes:
                report_file.write(
                    f"目标ID: {box['track_id']}\n"
                    f"类别: {box['class_name']}\n"
                    f"置信度: {box['confidence']:.2f}\n"
                    f"边界框坐标: ({box['bbox'][0]}, {box['bbox'][1]}) -> ({box['bbox'][2]}, {box['bbox'][3]})\n"
                )
                if not self.is_image and self.show_speed:
                    speed = self.last_displayed_speed.get(box['track_id'], 0)
                    report_file.write(f"估算速度: {speed:.2f} m/s\n")
                report_file.write("-"*40 + "\n")

        # 将检测结果写入数据库
        if self.conn:
            try:
                cursor = self.conn.cursor()
                for box in self.boxes:
                    x1, y1, x2, y2 = box["bbox"]
                    class_name = box["class_name"]
                    confidence = box["confidence"]
                    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    query = """
                        INSERT INTO Detection_result_table 
                        (class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """
                    cursor.execute(query, (class_name, confidence, x1, y1, x2, y2, detection_time))

                self.conn.commit()
                print("检测结果已写入数据库")
            except Exception as e:
                print(f"写入数据库失败: {e}")
                self.conn.rollback()
        else:
            print("数据库连接不可用")

        QMessageBox.information(
            self,
            "保存成功",
            f"检测结果已保存到:\n"
            f"图片: {save_image_path}\n"
            f"报告: {report_path}\n"
            f"数据库表: Detection_result_table"
        )

    def view_detection_reports(self):
        reports_folder = DEFAULT_RESULTS_DIR
        if os.path.exists(reports_folder):
            os.startfile(reports_folder)  # 打开文件夹
        else:
            QMessageBox.warning(self, "文件夹不存在", "检测结果报告文件夹不存在，请检查路径是否正确。")

    # 查看数据库中的检测结果
    def view_detection_database(self):
        if not self.conn:
            QMessageBox.warning(self, "数据库错误", "无法连接到数据库")
            return

        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM Detection_result_table"
            cursor.execute(query)
            records = cursor.fetchall()

            # 创建并显示数据库查看对话框
            self.database_dialog = DatabaseViewDialog(records, self.conn)
            self.database_dialog.exec_()
        except Exception as e:
            print(f"查询数据库失败: {e}")
            QMessageBox.warning(self, "数据库错误", "查询检测结果失败")

    # 切换速度显示状态
    def toggle_speed_display(self):
        if self.is_image:
            self.selected_box = None
            self.label_selected_info.setText("速度估算功能仅适用于视频模式")
            QMessageBox.information(self, "提示", "速度估算功能仅适用于视频模式")
            return
        self.show_speed = not self.show_speed
        self.btn_toggle_speed.setText("关闭速度估算功能" if self.show_speed else "开启速度估算功能")

    def select_file(self):
        options = QFileDialog.Options()
        default_dir = DEFAULT_MEDIA_DIR
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择照片或视频",
            default_dir,
            "Images (*.png *.jpg *.jpeg);;Videos (*.mp4 *.avi *.mov *.mkv)",
            options=options
        )
        if file_name:
            # 停止并释放之前的视频资源
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            if self.timer.isActive():
                self.timer.stop()
            # 重置视频相关状态
            self.is_video_paused = False
            self.btn_pause_video.setText("暂停视频")

            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.is_image = True
                self.process_image(file_name)
                self.update_mode_display("照片分析")
                self.label_fps.setText("当前视频帧率: 0.00 FPS")  # 切换回照片模式时清零帧率
            elif file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.is_image = False
                self.cap = cv2.VideoCapture(file_name)
                if self.cap.isOpened():
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.timer.start(50)
                    self.update_mode_display("实时视频")

    def update_mode_display(self, mode_text=""):
        self.label_mode.setText(f"当前模式：{mode_text}")

    def process_image(self, file_name):
        self.current_frame_path = file_name
        original_frame = cv2.imread(file_name)
        if original_frame is None:
            QMessageBox.warning(self, "错误", "无法加载图像文件")
            return

        # 保持原始图像尺寸
        predicted_frame = original_frame.copy()

        # 保存原始帧和预测帧
        self.current_frame = original_frame
        self.predicted_frame = predicted_frame

        if self.is_detection_enabled:
            # 清空之前的检测结果
            self.boxes = []
            self.class_counts = {class_name: 0 for class_name in self.class_names.values()}

            # 记录检测开始时间
            start_time = time.perf_counter()

            # 执行目标检测
            results = self.model.detect(predicted_frame, conf_threshold=self.confidence_threshold)
            detected_objects = 0
            self.track_id_counter = 1  # 重置ID计数器

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = self.class_names[class_id]
                    confidence = float(box.conf)

                    # 过滤置信度低于阈值的目标
                    if confidence < self.confidence_threshold:
                        continue

                    if self.selected_class and class_name != self.selected_class:
                        continue

                    detected_objects += 1
                    self.class_counts[class_name] += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_color(class_name)
                    track_id = self.track_id_counter
                    self.track_id_counter += 1

                    # 在预测帧上绘制检测框
                    cv2.rectangle(predicted_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID: {track_id} {class_name} {confidence:.2f}"
                    cv2.putText(predicted_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

                    self.boxes.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2)
                    })

            # 更新预测帧
            self.predicted_frame = predicted_frame

            # 计算检测耗时
            detection_time = time.perf_counter() - start_time

            # 更新界面显示
            self.update_detection_time_display(detection_time)
            self.label_count.setText(f"检测到的目标数量: {detected_objects}")
            self.text_edit.clear()
            for class_name, count in self.class_counts.items():
                if count > 0:
                    color = get_color(class_name)
                    self.append_colored_text(f"{class_name}: {count}\n", color)

        # 显示逻辑
        if self.show_original_image:
            # 确保两幅图像尺寸相同
            predicted_frame = cv2.resize(predicted_frame, (original_frame.shape[1], original_frame.shape[0]))
            separator = np.zeros((original_frame.shape[0], 20, 3), dtype=np.uint8)
            combined_frame = np.hstack((original_frame, separator, predicted_frame))

            # 添加标签
            cv2.putText(combined_frame, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 2)
            cv2.putText(combined_frame, "Predicted",
                        (original_frame.shape[1] + separator.shape[1] + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)
            display_image(combined_frame, self.label_image)
        else:
            # 直接显示预测图像
            cv2.putText(predicted_frame, "Predicted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)
            display_image(predicted_frame, self.label_image)

        # 更新目标选择下拉列表
        self.update_target_combo()

    # 更新检测时间显示
    def update_detection_time_display(self, detection_time):
        self.label_detection_time.setText(f"检测用时: {detection_time:.2f} 秒")

    # 定时器触发的帧更新方法
    def update_frame(self):
        self.mutex.lock()
        if not self.is_video_paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.detect_and_display(frame)
                # 计算并显示实时帧率
                if hasattr(self, 'fps'):
                    self.label_fps.setText(f"当前视频实时帧率: {self.fps:.2f} FPS")
            else:
                self.timer.stop()
                self.cap.release()
        self.mutex.unlock()

        # 设置为视频模式
        self.is_image = False
        self.selected_box = None  # 清除选中目标

    # 更新置信度阈值
    def update_confidence_threshold(self):
        # 将滑条值(0-100)转换为0-1之间的浮点数
        self.confidence_threshold = self.confidence_slider.value() / 100.0
        self.confidence_label.setText(f"置信度阈值: {self.confidence_threshold:.2f}")

        # 强制重新处理当前帧
        if hasattr(self, 'current_frame_path') and self.is_image:
            self.process_image(self.current_frame_path)
        elif self.current_frame is not None:
            self.detect_and_display(self.current_frame)

    def detect_and_display(self, frame):
        if self.is_detection_enabled:
            start_time = datetime.now()  # 记录检测开始时间

            # 创建原始帧的干净副本（不包含任何检测框）
            original_frame = frame.copy()
            # 创建用于绘制的预测帧
            predicted_frame = frame.copy()

            # 执行目标检测
            results = self.model.detect(frame)
            detected_objects = 0
            self.class_counts = {class_name: 0 for class_name in self.class_names.values()}
            self.boxes = []
            self.track_id_counter = 1  # 用于生成唯一的ID

            current_time = datetime.now()  # 当前帧的时间

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = self.class_names[class_id]
                    confidence = float(box.conf)

                    # 过滤置信度低于阈值的目标
                    if confidence < self.confidence_threshold:
                        continue

                    if self.selected_class and class_name != self.selected_class:
                        continue

                    detected_objects += 1
                    self.class_counts[class_name] += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = get_color(class_name)

                    track_id = self.track_id_counter
                    self.track_id_counter += 1

                    if track_id not in self.tracked_objects:
                        self.tracked_objects[track_id] = {
                            'positions': [],
                            'timestamps': [],
                        }
                    self.tracked_objects[track_id]['positions'].append((x1, y1, x2, y2))
                    self.tracked_objects[track_id]['timestamps'].append(current_time)

                    # 在预测帧上绘制检测框
                    cv2.rectangle(predicted_frame, (x1, y1), (x2, y2), color, 2)

                    # 显示类别和置信度
                    label_class = f"{class_name} {confidence:.2f}" if self.show_confidence else class_name
                    cv2.putText(predicted_frame, label_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

                    # 仅在照片模式下显示ID
                    if self.is_image:
                        label_id = f"ID: {track_id}"
                        cv2.putText(predicted_frame, label_id, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

                    # 仅在视频模式下显示速度
                    if not self.is_image and self.show_speed:
                        if len(self.tracked_objects[track_id]['positions']) > 1:
                            current_position = (x1, (y1 + y2) // 2)
                            last_position = self.tracked_objects[track_id]['positions'][-2]
                            time_diff = (current_time - self.tracked_objects[track_id]['timestamps'][-2]).total_seconds()

                            speed_pixel_per_sec = np.sqrt((current_position[0] - last_position[0]) ** 2 +
                                                          (current_position[1] - last_position[1]) ** 2) / time_diff
                            speed_meter_per_sec = speed_pixel_per_sec * self.speed_estimator.pixel_to_meter

                            self.last_displayed_speed[track_id] = speed_meter_per_sec
                            cv2.putText(predicted_frame, f"Speed: {speed_meter_per_sec:.2f} m/s",
                                        (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

                    self.boxes.append({
                        "track_id": track_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2)
                    })

            # 记录检测结束时间并计算检测用时
            end_time = datetime.now()
            detection_time = (end_time - start_time).total_seconds()

            # 更新界面显示
            self.label_count.setText(f"检测到的目标数量: {detected_objects}")
            self.label_detection_time.setText(f"检测用时: {detection_time:.2f} 秒")
            self.text_edit.clear()
            for class_name, count in self.class_counts.items():
                if count > 0:
                    color = get_color(class_name)
                    self.append_colored_text(f"{class_name}: {count}\n", color)

            # 处理图像显示
            if not self.is_image and self.show_original_image:
                # 视频/摄像头模式下显示原始和预测对比
                separator = np.zeros((original_frame.shape[0], 20, 3), dtype=np.uint8)
                combined_frame = np.hstack((original_frame, separator, predicted_frame))

                # 标签
                cv2.putText(combined_frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 144, 255), 2)
                cv2.putText(combined_frame, "Predicted",
                            (original_frame.shape[1] + separator.shape[1] + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)

                display_image(combined_frame, self.label_image)
            else:
                # 直接显示检测结果
                if not self.is_image:
                    cv2.putText(predicted_frame, "Predicted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 205, 50), 2)
                display_image(predicted_frame, self.label_image)

            # 更新目标选择下拉列表
            self.update_target_combo()
        else:
            # 如果检测功能关闭
            self.label_count.setText("检测功能已关闭")
            self.text_edit.clear()
            display_image(frame, self.label_image)

        torch.cuda.empty_cache()

    def get_pixel_to_meter_ratio(self):
        # 车道线的实际长度是3米，在视频中的像素长度是300像素
        actual_length_meters = 3
        pixel_length = 300  # 需要根据实际情况测量
        pixel_to_meter = actual_length_meters / pixel_length
        return pixel_to_meter

    def append_colored_text(self, text, color):
        text_cursor = self.text_edit.textCursor()
        text_cursor.movePosition(QTextCursor.End)
        self.text_edit.setTextCursor(text_cursor)
        # 将 BGR 转换为 RGB
        rgb_color = QColor(color[2], color[1], color[0])
        self.text_edit.setTextColor(rgb_color)
        self.text_edit.insertPlainText(text)

    # 切换权重文件
    def switch_model(self):
        options = QFileDialog.Options()
        default_dir = DEFAULT_WEIGHTS_DIR
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "选择权重文件",
            default_dir,
            "PyTorch Model Files (*.pt)",
            options=options
        )
        if file_name:
            try:
                # 加载新的模型
                self.model = ObjectDetection(file_name, self.device)
                self.class_names = self.model.class_names
                self.current_model_path = file_name

                # 更新显示的权重文件名
                self.label_model.setText(f"当前权重文件: {os.path.basename(self.current_model_path)}")

                # 更新类别选项
                self.combo_box.clear()
                self.combo_box.addItem("所有类别")
                self.combo_box.addItems(self.class_names.values())

                # 重新检测并显示当前图像（如果是照片模式）
                if hasattr(self, 'current_frame_path') and self.current_frame_path and self.is_image:
                    self.process_image(self.current_frame_path)
                elif self.current_frame is not None:  # 如果是视频模式但当前有帧数据
                    self.detect_and_display(self.current_frame)

            except Exception as e:
                QMessageBox.warning(self, "错误", f"加载模型失败: {str(e)}")

    def toggle_pause_video(self):
        self.is_video_paused = not self.is_video_paused
        self.btn_pause_video.setText("播放视频" if self.is_video_paused else "暂停视频")

    def filter_class(self, text):
        self.selected_class = text if text != "所有类别" else None
        if self.current_frame is not None:
            if self.is_image:
                self.process_image(self.current_frame_path)
            else:
                self.detect_and_display(self.current_frame)

    def mark_violation(self, violation_type):
        # 检查是否有选中的目标
        if not self.selected_box and self.target_combo.currentIndex() > 0:
            # 从下拉菜单中获取选中的目标信息
            selected_text = self.target_combo.currentText()
            try:
                target_id = int(selected_text.split("ID: ")[1].split(" - ")[0])
                # 查找对应的目标
                for box in self.boxes:
                    if box['track_id'] == target_id:
                        self.selected_box = box
                        break
            except (IndexError, ValueError):
                pass

        # 再次检查是否有选中的目标
        if self.selected_box:
            class_name = self.selected_box["class_name"]
            confidence = self.selected_box["confidence"]
            insert_violation_record(self.conn, class_name, confidence, violation_type)

            # 获取当前时间
            violation_time = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 获取目标框的坐标
            x1, y1, x2, y2 = self.selected_box["bbox"]

            # 从当前帧中截取目标区域
            target_image = self.current_frame[y1:y2, x1:x2]

            # 根据违法类型选择保存路径
            if violation_type == "违停":
                save_folder = os.path.join(DEFAULT_VIOLATION_SCREENSHOT_DIR, "Target_parkingdisorder_screenshot")
            elif violation_type == "超速":
                save_folder = os.path.join(DEFAULT_VIOLATION_SCREENSHOT_DIR, "Target_overspeed_screenshot")
            else:
                QMessageBox.warning(self, "错误", "未知的违法类型")
                return

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            file_name = f"{class_name}_{violation_type}_{violation_time}.jpg"
            save_path = os.path.join(save_folder, file_name)

            # 保存截图
            cv2.imwrite(save_path, target_image)

            QMessageBox.information(self, "标记成功", f"已标记为{violation_type}，截图已保存。")
        else:
            QMessageBox.warning(self, "未选中目标", "请先选择一个目标")

    def view_violations(self):
        if not self.conn:
            QMessageBox.warning(self, "数据库错误", "无法连接到数据库")
            return

        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM Traffic_violation_information_table"
            cursor.execute(query)
            records = cursor.fetchall()

            self.violation_dialog = ViolationManagementDialog(records, self.conn)
            self.violation_dialog.exec_()
        except Exception as e:
            print(f"查询数据库失败: {e}")
            QMessageBox.warning(self, "数据库错误", "查询违法记录失败")

    # 查看违法目标截图
    def view_violation_screenshots(self):
        screenshot_folder = DEFAULT_VIOLATION_SCREENSHOT_DIR
        if os.path.exists(screenshot_folder):
            os.startfile(screenshot_folder)
        else:
            QMessageBox.warning(self, "文件夹不存在", "指定的截图文件夹不存在，请检查路径是否正确。")

    def update_display_boxes(self):
        if self.current_frame is None:
            return

        frame = self.current_frame.copy()
        for box in self.boxes:
            x1, y1, x2, y2 = box["bbox"]
            color = get_color(box["class_name"])
            if box == self.selected_box:
                color = (0, 0, 255)  # 选中的框变红色

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 显示类别和置信度
            label_class = f"{box['class_name']} {box['confidence']:.2f}" if self.show_confidence else box['class_name']
            cv2.putText(frame, label_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

            # 仅在照片模式下显示ID
            if self.is_image:
                label_id = f"ID: {box['track_id']}"
                cv2.putText(frame, label_id, (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

            # 仅在视频模式下显示速度
            if not self.is_image and self.show_speed:
                track_id = box["track_id"]
                speed_meter_per_sec = self.last_displayed_speed.get(track_id, 0)
                cv2.putText(frame, f"Speed: {speed_meter_per_sec:.2f} m/s",
                            (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness)

        display_image(frame, self.label_image)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "退出系统",
            "是否确认退出系统？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
