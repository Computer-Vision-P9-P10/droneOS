from ultralytics import YOLO


model = YOLO("./yolo12m.pt")
model.export(format="onnx", imgsz=640,half=True,device=0,workspace=4)

# yolo export model=yolo12m.pt format=engine imgsz=640 device=0
