#利用yolov8训练数据集
from ultralytics import YOLO

# 加载模型
model = YOLO("yolov8s.yaml") #从头开始构建新模型
model = YOLO("weights/yolov8s.pt") #加载预训练模型（建议用于训练）

# 使用模型
model.train(data="ultralytics/cfg/datasets/car-test.yaml",epochs=3) #训练模型
metrics = model.val() # 在验证集上评估模型性能
# results = model.predict(source="ultralytics/assets/bus.jpg",save=True) #对图形进行预测
# success = model.export(format="onnx") # 将模型导出为ONNX格式


