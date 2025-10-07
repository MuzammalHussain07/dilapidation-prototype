# backend.py
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

# Load model once globally
model = YOLO("models/best.pt")
print("✅ Model loaded successfully:", model)

def analyze_image_bytes(file_bytes):
    """
    Takes uploaded file bytes, runs YOLO detection, and returns PIL Image with boxes drawn.
    """
    # Load image
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array = np.array(image)

    # Run detection (lower conf if needed)
    results = model.predict(img_array, conf=0.25, verbose=True)

    # Save image with boxes drawn (YOLO returns a list of results)
    result = results[0]
    result_image = Image.fromarray(result.plot()[:, :, ::-1])  # convert BGR to RGB for PIL
    return result_image, result
