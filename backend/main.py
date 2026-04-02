from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/")
def home():
    return {"message": "YOLO Live API Running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()

    return {"boxes": boxes, "classes": classes}