# backend.py
import os
import subprocess
import io
import numpy as np
from PIL import Image

# 🩹 Fix for Streamlit Cloud: install OpenCV if missing
try:
    import cv2
except ImportError:
    subprocess.run(["pip", "install", "opencv-python-headless==4.8.1.78"], check=True)
    import cv2

from ultralytics import YOLO

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "models/best.pt"  # ensure this file exists in your repo
model = YOLO(MODEL_PATH)

# ✅ Optional: confirm model load in logs
print(f"✅ Model loaded successfully from {MODEL_PATH}")

# Class names must match your data.yaml
CLASS_NAMES = ['Crack', 'Dampness', 'Ground_settlement', 'Mould', 'Spalling']


def analyze_image_bytes(image_bytes):
    """
    Input: file-like object (BytesIO or uploaded file)
    Output: PIL annotated image, summary dict
    """
    pil = Image.open(image_bytes).convert("RGB")
    img = np.array(pil)  # RGB image as numpy array

    # Predict using YOLO model
    results = model.predict(source=img, imgsz=640, conf=0.25, device='cpu')
    r = results[0]

    boxes = []
    for box in r.boxes:
        cls = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        boxes.append({
            "class": CLASS_NAMES[cls],
            "conf": conf,
            "bbox": (x1, y1, x2, y2)
        })

    # If no detections found
    if not boxes:
        return pil, {"found": False, "description": "No defect detected", "class": None}

    # Pick the most confident detection
    boxes = sorted(boxes, key=lambda x: x["co]()
