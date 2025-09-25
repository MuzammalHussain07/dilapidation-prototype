import cv2
import numpy as np
from PIL import Image

def process_image(uploaded_file, mode="auto", ref_mm=0.3):
    """
    mode = "crack"   → expects ruler+crack, output mm + arrow
    mode = "defect"  → no ruler, highlight biggest defect area only
    mode = "auto"    → guess (prototype)
    ref_mm = actual crack width in mm (entered by user)
    """

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Preprocess for better detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_found = False
    crack_width_pixels = 0
    biggest_crack = None
    defect_boxes = []

    # --- Detect contours ---
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h) if h > 0 else 0

        # Long thin contour → crack candidate
        if aspect_ratio > 3 and w > crack_width_pixels:
            crack_found = True
            crack_width_pixels = w
            biggest_crack = (x, y, w, h)
        else:
            defect_boxes.append((x, y, w, h))

    # -------------------------
    # CASE 1: Crack with ruler
    # -------------------------
    if mode == "crack" or (mode == "auto" and crack_found):
        if biggest_crack:
            x, y, w, h = biggest_crack

            # Draw arrow near crack
            arrow_start = (x + w // 2, y + h + 40)
            arrow_end   = (x + w // 2, y + h + 5)
            cv2.arrowedLine(img_array, arrow_start, arrow_end, (255, 0, 0), 3, tipLength=0.3)

            # Assume detected crack = reference mm
            crack_width_mm = round(ref_mm, 2)

            summary = {
                "type": "crack",
                "found": True,
                "description": f"Crack detected. Width: {crack_width_mm} mm",
                "est_crack_width_mm": crack_width_mm
            }
        else:
            summary = {
                "type": "crack",
                "found": False,
                "description": "No crack detected",
                "est_crack_width_mm": 0
            }

    # -------------------------
    # CASE 2: Other defects
    # -------------------------
    else:
        biggest_defect = None
        max_area = 0
        for (x, y, w, h) in defect_boxes:
            area = w * h
            if 800 < area < 80000 and area > max_area:  # filter & keep only largest
                max_area = area
                biggest_defect = (x, y, w, h)

        if biggest_defect:
            x, y, w, h = biggest_defect

            # 🔴 Expand rectangle a bit around defect
            pad = 10
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = w + pad * 2
            h = h + pad * 2

            # Draw RED rectangle
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 0, 255), 3)

            summary = {
                "type": "defect",
                "found": True,
                "description": "Defect area detected (peeling paint / stain / corrosion)",
                "est_crack_width_mm": 0
            }
        else:
            summary = {
                "type": "defect",
                "found": False,
                "description": "No defect detected",
                "est_crack_width_mm": 0
            }

    # --- Return processed image and report data ---
    out_img = Image.fromarray(img_array)
    return out_img, summary
