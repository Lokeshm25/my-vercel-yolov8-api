from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
import os
import requests

# Download model if not present
MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    print("🔽 Downloading YOLOv8n model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = YOLO(MODEL_PATH)

# Create FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    results = model(image)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "box": box.xyxy[0].tolist()
            })

    return {"detections": detections}
