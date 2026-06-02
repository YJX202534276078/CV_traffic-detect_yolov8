#测试pytorch是否安装成功，是否与cuda连接成功
# import torch
# print(torch.__version__)  # 查看 PyTorch 版本
# print(torch.cuda.is_available())  # 检查 CUDA 是否可用
# x = torch.rand(5, 3).cuda()  # 将张量移动到 GPU
# print(x)

#测试pyqt5是否安装成功，安装成功才能启动labelimg
import sys
from PyQt5.QtWidgets import QWidget, QApplication

app = QApplication(sys.argv)
widget = QWidget()
widget.resize(640,480)
widget.setWindowTitle("Hello,PyQt5!")
widget.show()
sys.exit(app.exec())