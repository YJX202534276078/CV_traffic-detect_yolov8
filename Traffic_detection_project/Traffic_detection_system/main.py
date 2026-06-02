# main
import sys
from PyQt5.QtWidgets import QApplication
from main_window import TrafficMonitoringSystem

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrafficMonitoringSystem()
    window.show()
    sys.exit(app.exec_())