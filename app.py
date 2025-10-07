# app.py
import streamlit as st
from backend import analyze_image_bytes

st.set_page_config(page_title="YOLO Damage Detector", layout="centered")

st.title("🏗️ Building Damage Detection")
st.write("Upload an image to detect **cracks**, **spalling**, **water leaks**, or **paint peeling** using your trained YOLO model.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Running detection..."):
        result_image, result = analyze_image_bytes(uploaded_file.getvalue())
    st.image(result_image, caption="Detection Result", use_column_width=True)
    st.success("✅ Detection complete!")

    # Optional: Show raw result data
    st.write("📊 Detection Summary:")
    st.json(result.tojson())
