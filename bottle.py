from ultralytics import YOLO
model = YOLO('./bottle/best.pt')
results = model.predict(source=0, show=True)
