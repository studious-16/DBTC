
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import time
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from CannyEdgeDetector import *
import pytesseract
import easyocr
import difflib

# Streamlit setup
st.set_page_config(page_title="Smart Traffic Control", layout="centered")
# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# Load model
COCO_INSTANCE_CATEGORY_NAMES = [...]  # Same as before

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

@st.cache_resource
def load_model():
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights)
    model.eval()
    return model

model = load_model()

# Load the EAST text detector
east_net = cv2.dnn.readNet(r"C:\Users\Slnrockstones\Desktop\newmodel\frozen_east_text_detection.pb")


# Utility Functions
def is_ambulance_text(text):
    text = text.upper()
    words = text.split()
    for word in words:
        word_clean = ''.join(filter(str.isalpha, word))
        if 5 <= len(word_clean) <= 10:
            # Check normal and reversed
            if difflib.get_close_matches(word_clean, ["AMBULANCE"], cutoff=0.7):
                return True
            if difflib.get_close_matches(word_clean[::-1], ["AMBULANCE"], cutoff=0.7):
                return True
    return False

def preprocess_for_ocr(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    return thresh

def ocr_pytesseract(image):
    return pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')

def ocr_easyocr(image, conf_threshold=0.3):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = easyocr_reader.readtext(rgb)
    filtered_texts = [res[1] for res in results if res[2] > conf_threshold]
    return " ".join(filtered_texts)

def detect_text_with_east_and_ocr(crop):
    detected_texts = []
    text_boxes = detect_text_regions_east(crop)
    print(f"[DEBUG] EAST detected {len(text_boxes)} text boxes.")
    
    for (startX, startY, endX, endY) in text_boxes:
        # Clip coordinates
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(crop.shape[1], endX), min(crop.shape[0], endY)
        roi = crop[startY:endY, startX:endX]

        processed = preprocess_for_ocr(roi)
        text_pt = ocr_pytesseract(processed).strip()
        text_eo = ocr_easyocr(roi).strip()

        combined = (text_pt + " " + text_eo).strip()
        detected_texts.append(combined)

        print(f"[OCR DEBUG] pytesseract: '{text_pt}', easyocr: '{text_eo}', combined: '{combined}'")

        if is_ambulance_text(combined):
            return combined

    return " ".join(detected_texts)


st.title("🚦Smart Traffic Flow Regulation Using Canny Edge Detection ")
st.sidebar.title("📋 Console Logs")

# Session state
for key in ['ambulance_detected', 'canny_output', 'traffic_density', 'uploaded_image', 'uploaded_file_raw']:
    if key not in st.session_state:
        st.session_state[key] = None

# File uploader
uploaded_file = st.file_uploader("📤 Upload a Traffic Image", type=["jpg", "jpeg", "png"])
if uploaded_file != st.session_state['uploaded_file_raw']:
    st.session_state.update({
        'ambulance_detected': None,
        'canny_output': None,
        'traffic_density': None,
        'uploaded_image': Image.open(uploaded_file) if uploaded_file else None,
        'uploaded_file_raw': uploaded_file
    })

# Core functions
def apply_canny(image):
    gray_img = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
    edge = CannyEdgeDetector([gray_img], sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.20, weak_pixel=100)
    processed = edge.detect()[0]
    return (processed / processed.max() * 255).astype(np.uint8)

def count_white_pixels(img):
    return np.sum(img == 255)
def decode_predictions(scores, geometry, min_confidence):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < min_confidence:
                continue

            offsetX = x * 4.0
            offsetY = y * 4.0

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + cos * xData1[x] + sin * xData2[x])
            endY = int(offsetY - sin * xData1[x] + cos * xData2[x])
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    return (rects, confidences)


def detect_text_regions_east(image, width=320, height=320, min_confidence=0.5):
    orig = image.copy()
    (H, W) = image.shape[:2]
    rW = W / float(width)
    rH = H / float(height)

    # Resize and prepare blob
    resized = cv2.resize(image, (width, height))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    east_net.setInput(blob)
    (scores, geometry) = east_net.forward([
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
    ])

    (boxes, confidences) = decode_predictions(scores, geometry, min_confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            startX = int(x * rW)
            startY = int(y * rH)
            endX = int((x + w) * rW)
            endY = int((y + h) * rH)
            final_boxes.append((startX, startY, endX, endY))

    return final_boxes



def detect_vehicles_with_ocr(image):
    img_rgb = np.array(image.convert("RGB"))
    img_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    with torch.no_grad():
        prediction = model(img_tensor)

    found = False
    ambulance_text_detected = ""
    for i in range(len(prediction[0]['labels'])):
        label = prediction[0]['labels'][i].item()
        score = prediction[0]['scores'][i].item()
        if label in [3, 6, 8] and score > 0.3:
            box = prediction[0]['boxes'][i].cpu().numpy().astype(int)
            crop = img_rgb[box[1]:box[3], box[0]:box[2]]
            text = detect_text_with_east_and_ocr(crop)

            if text.strip().lower() != "":

                found = True
                ambulance_text_detected = text
                cv2.rectangle(img_rgb, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(img_rgb, "AMBULANCE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    return img_rgb, found, ambulance_text_detected

# Main interaction
if st.session_state['uploaded_image']:
    image = st.session_state['uploaded_image']
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    if st.button("🚑 Detect Ambulance"):
        with st.spinner("Analyzing vehicles and text..."):
            result_img, found, ambulance_text = detect_vehicles_with_ocr(image)
            st.image(result_img, caption="🚓 Detection Output", use_column_width=True)
            st.sidebar.write("✅ Detection complete.")

        if found:
            st.success("🚨 **Ambulance Detected!** Green signal time: 60 seconds or until it passess.")
         
            st.session_state['ambulance_detected'] = True
        else:
            st.warning("❌ No ambulance found.")
            st.session_state['ambulance_detected'] = False

    if st.session_state['ambulance_detected'] is True:
        st.info("🚑 Ambulance detected! Skipping Canny Edge Detection and traffic analysis.")
    else:
        if st.button("⚙️ Run Canny Edge Detection"):
            progress_bar = st.progress(0, text="Canny Edge Detection in progress...")

            with st.spinner("Detecting edges..."):
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1, text="Canny Edge Detection in progress...")

                canny_img = apply_canny(image)
                progress_bar.empty()

                st.image(canny_img, caption="🖼️ Canny Edge Output", use_column_width=True)
                st.session_state['canny_output'] = canny_img

        if st.button("📊 Analyze Traffic & Allocate Green Time"):
            if st.session_state['canny_output'] is None:
                st.warning("⚠️ Please run Canny Edge Detection first.")
            else:
                with st.spinner("Calculating white pixel density..."):
                    img = st.session_state['canny_output']
                    white_pixels = count_white_pixels(img)
                    img_area = img.shape[0] * img.shape[1]
                    density = white_pixels / img_area
                    st.session_state['traffic_density'] = density

                    # Visual feedback
                    if density >= 0.18:
                        level, time_sec, color = "🚗 Very High", 60, "#ffebee"
                    elif density >= 0.14:
                        level, time_sec, color = "🚙 High", 50, "#fff3e0"
                    elif density >= 0.10:
                        level, time_sec, color = "🚕 Moderate", 40, "#f0f4c3"
                    elif density >= 0.06:
                        level, time_sec, color = "🚘 Low", 30, "#e0f7fa"
                    else:
                        level, time_sec, color = "🛵 Very Low", 20, "#f3e5f5"

                    st.metric(label="📈 Traffic Density", value=f"{density:.4f}", delta=level)
                    st.metric(label="⏱️ Green Light Time", value=f"{time_sec} seconds")

                    st.markdown(f"""
                        <div style='padding:12px; background-color:{color}; border-left:6px solid #1976d2;'>
                            <b>📌 White Pixels:</b> {white_pixels}<br>
                            <b>📐 Image Area:</b> {img_area}<br>
                            <b>🚦 Density:</b> {density:.4f} ({level})<br>
                            <b>🟢 Green Time:</b> {time_sec} seconds
                        </div>
                    """, unsafe_allow_html=True)

# Sidebar # Sidebar Status Summary
st.sidebar.title("Process Status")

if st.session_state['uploaded_image'] is None:
    st.sidebar.write("❌ No image uploaded.")
else:
    st.sidebar.write("✅ Image uploaded.")

if st.session_state['ambulance_detected'] is None:
    st.sidebar.write("🚑 Ambulance Detection: Not done yet.")
elif st.session_state['ambulance_detected'] is True:
    st.sidebar.success("🚑 Ambulance detected!")
else:
    st.sidebar.info("🚑 No ambulance detected.")

if st.session_state['canny_output'] is None:
    st.sidebar.write("⚙️ Canny Edge Detection: Not done yet.")
else:
    st.sidebar.success("⚙️ Canny Edge Detection completed.")

if st.session_state['traffic_density'] is None:
    st.sidebar.write("📊 Traffic Analysis: Not done yet.")
else:
    dens = st.session_state['traffic_density']
    st.sidebar.success(f"📊 Traffic Density: {dens:.4f}")

# --- Add How To Use Info Panel ---
with st.sidebar.expander("ℹ️ How to Use This App"):
    st.markdown("""
    - **Upload Image:** Upload a traffic image to analyze.
    - **Detect Ambulance:** Detect if any ambulance is present in the image using OCR and object detection.
    - **Canny Edge Detection:** Highlights edges to estimate traffic density.
    - **Traffic Density:** Based on white pixel density, shows traffic load and green light timing.

    **Traffic Density Meaning:**
    - Very High (≥ 0.18): Heavy traffic — longer green light.
    - High (≥ 0.14): High traffic — moderate green light.
    - Moderate (≥ 0.10): Normal traffic.
    - Low (≥ 0.06): Light traffic.
    - Very Low (< 0.06): Very light traffic — shortest green light.

    **Ambulance Detected:** Overrides traffic density logic to allow faster passage.
    """)

# Reset
if st.button("🔄 Reset All"):
    st.session_state.clear()
    st.rerun()
