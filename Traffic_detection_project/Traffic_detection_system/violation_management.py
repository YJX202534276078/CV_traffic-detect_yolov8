# violation_management
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QMessageBox, QHeaderView, QLineEdit, QLabel, QDateEdit, QComboBox, QDesktopWidget
from PyQt5.QtCore import QDate

class ViolationManagementDialog(QDialog):
    def __init__(self, records, conn, parent=None):
        super().__init__(parent)
        self.conn = conn
        self.setWindowTitle("违法记录管理")
        self.setGeometry(200, 200, 1000, 600)
        self.initUI(records)
        self.center()

    def center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def initUI(self, records):
        layout = QVBoxLayout()

        search_filter_layout = QHBoxLayout()

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("搜索违法记录...")
        self.search_input.textChanged.connect(self.filter_records)
        search_filter_layout.addWidget(self.search_input)

        self.class_filter = QComboBox(self)
        self.class_filter.addItem("所有类别")
        self.class_filter.addItems(["Car", "Truck", "Person", "Two wheeled vehicle", "Bus"])  # 这里可以根据实际情况动态获取类别
        self.class_filter.currentTextChanged.connect(self.filter_records)
        search_filter_layout.addWidget(self.class_filter)

        self.violation_type_filter = QComboBox(self)
        self.violation_type_filter.addItem("所有类型")
        self.violation_type_filter.addItems(["违停", "超速"])
        self.violation_type_filter.currentTextChanged.connect(self.filter_records)
        search_filter_layout.addWidget(self.violation_type_filter)

        self.start_date_edit = QDateEdit(self)
        self.start_date_edit.setDate(QDate.currentDate().addMonths(-1))
        self.start_date_edit.setCalendarPopup(True)
        self.end_date_edit = QDateEdit(self)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.setCalendarPopup(True)
        self.start_date_edit.dateChanged.connect(self.filter_records)
        self.end_date_edit.dateChanged.connect(self.filter_records)

        date_filter_layout = QHBoxLayout()
        date_filter_layout.addWidget(QLabel("开始日期:"))
        date_filter_layout.addWidget(self.start_date_edit)
        date_filter_layout.addWidget(QLabel("结束日期:"))
        date_filter_layout.addWidget(self.end_date_edit)
        search_filter_layout.addLayout(date_filter_layout)

        layout.addLayout(search_filter_layout)

        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["ID", "类别", "置信度", "违法类型", "违法时间"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setRowCount(len(records))

        for i, record in enumerate(records):
            self.table.setItem(i, 0, QTableWidgetItem(str(record[0])))  # ID
            self.table.setItem(i, 1, QTableWidgetItem(record[1]))  # 类别
            self.table.setItem(i, 2, QTableWidgetItem(str(record[2])))  # 置信度
            self.table.setItem(i, 3, QTableWidgetItem(record[3]))  # 违法类型
            self.table.setItem(i, 4, QTableWidgetItem(str(record[4])))  # 违法时间

        self.table.setEditTriggers(QTableWidget.AllEditTriggers)

        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        self.btn_delete = QPushButton("删除记录", self)
        self.btn_delete.clicked.connect(self.delete_record)
        button_layout.addWidget(self.btn_delete)

        self.btn_refresh = QPushButton("刷新记录", self)
        self.btn_refresh.clicked.connect(self.refresh_records)
        button_layout.addWidget(self.btn_refresh)

        self.btn_save = QPushButton("保存修改", self)
        self.btn_save.clicked.connect(self.save_changes)
        button_layout.addWidget(self.btn_save)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def filter_records(self):
        search_text = self.search_input.text().strip().lower()
        class_name = self.class_filter.currentText()
        violation_type = self.violation_type_filter.currentText()
        start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
        end_date = self.end_date_edit.date().toString("yyyy-MM-dd")

        try:
            cursor = self.conn.cursor()
            query = """
                SELECT * FROM Traffic_violation_information_table 
                WHERE (LOWER(class_name) LIKE ? OR LOWER(violation_type) LIKE ?)
                AND (class_name = ? OR ? = '所有类别')
                AND (violation_type = ? OR ? = '所有类型')
                AND violation_time BETWEEN ? AND ?
            """
            cursor.execute(query, (
                f"%{search_text}%", f"%{search_text}%",
                class_name, class_name,
                violation_type, violation_type,
                start_date, end_date
            ))
            records = cursor.fetchall()

            self.table.setRowCount(len(records))
            for i, record in enumerate(records):
                self.table.setItem(i, 0, QTableWidgetItem(str(record[0])))  # ID
                self.table.setItem(i, 1, QTableWidgetItem(record[1]))  # 类别
                self.table.setItem(i, 2, QTableWidgetItem(str(record[2])))  # 置信度
                self.table.setItem(i, 3, QTableWidgetItem(record[3]))  # 违法类型
                self.table.setItem(i, 4, QTableWidgetItem(str(record[4])))  # 违法时间
        except Exception as e:
            print(f"筛选记录失败: {e}")
            QMessageBox.warning(self, "数据库错误", "筛选记录失败")

    def delete_record(self):
        selected_row = self.table.currentRow()
        if selected_row == -1:
            QMessageBox.warning(self, "未选中记录", "请先选择一条记录")
            return

        record_id = self.table.item(selected_row, 0).text()
        try:
            cursor = self.conn.cursor()
            query = "DELETE FROM Traffic_violation_information_table WHERE id = ?"
            cursor.execute(query, record_id)
            self.conn.commit()
            QMessageBox.information(self, "删除成功", "记录已删除")
            self.refresh_records()
        except Exception as e:
            print(f"删除记录失败: {e}")
            QMessageBox.warning(self, "数据库错误", "删除记录失败")

    def refresh_records(self):
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM Traffic_violation_information_table"
            cursor.execute(query)
            records = cursor.fetchall()

            self.table.setRowCount(len(records))
            for i, record in enumerate(records):
                self.table.setItem(i, 0, QTableWidgetItem(str(record[0])))  # ID
                self.table.setItem(i, 1, QTableWidgetItem(record[1]))  # 类别
                self.table.setItem(i, 2, QTableWidgetItem(str(record[2])))  # 置信度
                self.table.setItem(i, 3, QTableWidgetItem(record[3]))  # 违法类型
                self.table.setItem(i, 4, QTableWidgetItem(str(record[4])))  # 违法时间
        except Exception as e:
            print(f"刷新记录失败: {e}")
            QMessageBox.warning(self, "数据库错误", "刷新记录失败")

    def save_changes(self):
        try:
            cursor = self.conn.cursor()
            for row in range(self.table.rowCount()):
                record_id = self.table.item(row, 0).text()
                class_name = self.table.item(row, 1).text()
                confidence = self.table.item(row, 2).text()
                violation_type = self.table.item(row, 3).text()
                violation_time = self.table.item(row, 4).text()

                query = """
                    UPDATE Traffic_violation_information_table 
                    SET class_name = ?, confidence = ?, violation_type = ?, violation_time = ?
                    WHERE id = ?
                """
                cursor.execute(query, (class_name, confidence, violation_type, violation_time, record_id))

            self.conn.commit()
            QMessageBox.information(self, "保存成功", "修改已保存到数据库")
        except Exception as e:
            print(f"保存修改失败: {e}")
            QMessageBox.warning(self, "数据库错误", "保存修改失败")