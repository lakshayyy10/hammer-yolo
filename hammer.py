from ultralytics import YOLO
model = YOLO('./hammer/best.pt')
results = model.predict(source=0, show=True)
