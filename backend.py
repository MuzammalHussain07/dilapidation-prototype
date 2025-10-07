# backend.py
import os
import io
import subprocess

# 🩹 Fix for Streamlit Cloud: install OpenCV if missing
try:
    import cv2
except ImportError:
    subprocess.run(
        ["pip", "install", "opencv-python-headless==4.8.1.78"],
        check=True
    )
    import cv2

from PIL import Image
import numpy as np
from ultralytics import YOLO

# Path to your trained model
MODEL_PATH = "models/best.pt"

# Load model once at startup
model = YOLO(MODEL_PATH)

# These must match your data.yaml class names
CLASS_NAMES = ['Crack', 'Dampness', 'Ground_settlement', 'Mould', 'Spalling']

def analyze_image_bytes(image_bytes):
    """Run YOLO detection on an uploaded image."""
    pil = Image.open(image_bytes).convert("RGB")
    img = np.array(pil)
    h_img, w_img, _ = img.shape

    # Run prediction
    results = model.predict(source=img, imgsz=640, conf=0.25, device='cpu')
    r = results[0]

    boxes = []
    for box in r.boxes:
        cls = int(box.cls.cpu().numpy()[0])
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
        boxes.append({"class": CLASS_NAMES[cls], "conf": conf, "bbox": (x1, y1, x2, y2)})

    # No detection
    if not boxes:
        return pil, {"found": False, "description": "No defect detected", "class": None}

    # Pick highest-confidence box
    boxes = sorted(boxes, key=lambda x: x["conf"], reverse=True)
    top = boxes[0]
    cls = top["class"]
    x1, y1, x2, y2 = top["bbox"]

    annotated = img.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

    summary = {"found": True, "class": cls, "confidence": round(top["conf"], 3)}

    # Simple severity estimate for cracks
    if cls == "Crack":
        w_px = (x2 - x1)
        if w_px < 8:
            severity = "Fine (<1 mm approx)"
        elif w_px < 20:
            severity = "Medium (<3 mm approx)"
        elif w_px < 40:
            severity = "Wide (<5 mm approx)"
        else:
            severity = "Very wide (>5 mm approx)"
        summary["description"] = f"Crack detected — {severity}"
    else:
        summary["description"] = f"{cls.replace('_', ' ')} detected"

    out_pil = Image.fromarray(annotated)
    return out_pil, summary
