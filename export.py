# 将输出pt文件改成engine格式
from ultralytics import YOLO
import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":
    #第一个命令行参数是脚本本身的路径，忽略它
    args = sys.argv[1] if len(sys.argv) > 1 else str(Path("weights") / "railcar.pt")
    # print(args)

    model = YOLO(args,task='detect')
    #加载预训练模型(建议用于训练)
    onnx = model.export(format="engine",dynamic=True,opset=12)# 对图像进行预测
