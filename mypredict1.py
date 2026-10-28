from ultralytics import YOLO

model = YOLO(r"D:\deeplearning\weights\best.onnx")
model.predict(
    source=r"C:\Users\A\Downloads\QQ\d65fb6a2d145feb7f3deddaaf75a9efc.mp4",
    save=True,
    show=False,
)