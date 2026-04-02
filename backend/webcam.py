from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    results = model(frame)
    frame = results[0].plot()

    cv2.imshow("Live YOLO", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()