import streamlit as st
from backend import process_image
from docx import Document
from docx.shared import Inches

st.title("Dilapidation Survey Prototype")

# --- Mode selection ---
mode = st.selectbox(
    "Select analysis mode:",
    ["Crack (with ruler)", "Other defect (no ruler)"]
)

# --- Crack calibration (only visible if Crack mode selected) ---
ref_mm = 0.3
if mode == "Crack (with ruler)":
    st.subheader("Calibration for Crack Measurement")
    ref_mm = st.number_input("Enter actual crack width on ruler (mm):", value=0.3, step=0.01)

# --- File upload ---
uploaded_files = st.file_uploader("Upload site photos", type=["jpg","png"], accept_multiple_files=True)

# --- Site info ---
location = st.text_input("Site Location")
inspector = st.text_input("Inspector Name")
date = st.date_input("Date")

if uploaded_files:
    reports = []
    for f in uploaded_files:
        if mode == "Crack (with ruler)":
            out_img, summary = process_image(f, mode="crack", ref_mm=ref_mm)
        else:
            out_img, summary = process_image(f, mode="defect")

        if summary["found"]:
            st.image(out_img, caption=summary["description"], use_column_width=True)
            st.json(summary)
            reports.append((f.name, out_img, summary))
        else:
            st.warning(f"No issue detected in {f.name}")

    # --- Generate Word report ---
    if st.button("Generate Word Report"):
        doc = Document()
        doc.add_heading("Dilapidation Survey Report", 0)
        doc.add_paragraph(f"Location: {location}")
        doc.add_paragraph(f"Inspector: {inspector}")
        doc.add_paragraph(f"Date: {date}")

        for idx,(name, out_img, summary) in enumerate(reports,1):
            doc.add_heading(f"Issue {idx}: {name}", level=1)
            doc.add_paragraph(summary["description"])
            out_img.save(f"temp_{idx}.png")
            doc.add_picture(f"temp_{idx}.png", width=Inches(5))

        report_filename = "dilapidation_report.docx"
        doc.save(report_filename)

        with open(report_filename, "rb") as file:
            st.download_button(
                label="📥 Download Report",
                data=file,
                file_name=report_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
