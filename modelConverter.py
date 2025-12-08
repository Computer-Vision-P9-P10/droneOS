from ultralytics import YOLO


model = YOLO("./yolo12n.pt")
model.export(format="onnx", imgsz=640,half=False,device=0,workspace=4)


# trtexec.exe --onnx=C:\Users\julia\Desktop\p9\droneOS\yolo12m.onnx --saveEngine=yolo12m.engine --fp16 --memPoolSize=workspace:6144MiB
