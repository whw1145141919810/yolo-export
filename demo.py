from ultralytics import  YOLO

model = YOLO(model=r"best.pt")

model.export(format="onnx")

