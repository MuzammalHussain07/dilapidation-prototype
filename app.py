import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# --- Streamlit title ---
st.title("🏗️ Smart Crack & Dampness Detector")
st.write("Upload an image and the AI model will automatically detect cracks, dampness, or surface defects.")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # show original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        img_path = temp.name

    st.write("🔍 Running detection... please wait")

    # load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")   # small, fast model — works out of the box

    # run detection
    results = model.predict(source=img_path, conf=0.3)

    # display results with bounding boxes
    for result in results:
        st.image(result.plot(), caption="Detection Result", use_column_width=True)

    st.success("✅ Detection complete!")
else:
    st.info("Please upload an image to begin.")
