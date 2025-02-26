from ultralytics import YOLO
import cv2
import numpy as np
model1 = YOLO('./bottle/best.pt')
model2 = YOLO('./hammer/best.pt')
cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    results1 = model1.predict(source=frame, conf=0.25, show=False)[0]
    results2 = model2.predict(source=frame, conf=0.25, show=False)[0]
    for box in results1.boxes.data:
        x1, y1, x2, y2, score, class_id = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        label = f"{results1.names[int(class_id)]} {score:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for box in results2.boxes.data:
        x1, y1, x2, y2, score, class_id = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"{results2.names[int(class_id)]} {score:.2f}"
        cv2.putText(frame, label, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("yolov8 detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
