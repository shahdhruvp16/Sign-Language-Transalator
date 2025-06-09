import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image, ImageDraw, ImageFont
from googletrans import Translator
import tempfile

# Load model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
classes = model_dict['classes']

# Load fonts
gujarati_font_path = "NotoSansGujarati-Regular.ttf"
hindi_font_path = "NotoSansDevanagari-Regular.ttf"

# Translator
translator = Translator()

# Function to overlay multilingual text on image using PIL
def add_text(image, text, position, font_path, font_size=24, color=(255, 255, 255)):
    try:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error adding text: {e}")
        return image

# Streamlit App
st.set_page_config(page_title="Sign Language Translator", layout="centered")
st.title("🤟 Sign Language Translator")
st.write("This app detects sign language gestures and shows translation in English, Hindi, and Gujarati.")

# Video Capture
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

prediction_label = st.empty()

# Run until stopped
if st.button("Start Detection"):
    st.write("Press **Stop** to end.")
    stop_button = st.button("Stop")

    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()
        if not ret:
            break

        # Region of interest
        x1, y1, x2, y2 = 100, 100, 324, 324
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]
        roi_resized = cv2.resize(roi, (64, 64)).astype(np.float32) / 255.0
        roi_flat = roi_resized.flatten().reshape(1, -1)

        # Prediction
        prediction = model.predict(roi_flat)[0]
        label = classes[int(prediction)]

        # Translations
        translated_hi = translator.translate(label, src='en', dest='hi').text
        translated_gu = translator.translate(label, src='en', dest='gu').text

        # Overlay text
        frame = add_text(frame, f"English: {label}", (10, 40), font_path=hindi_font_path)
        frame = add_text(frame, f"Hindi: {translated_hi}", (10, 70), font_path=hindi_font_path)
        frame = add_text(frame, f"Gujarati: {translated_gu}", (10, 100), font_path=gujarati_font_path)

        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()
    cv2.destroyAllWindows()
