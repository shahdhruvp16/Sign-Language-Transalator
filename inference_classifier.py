import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
from PIL import ImageFont, ImageDraw, Image
import googletrans
from googletrans import Translator

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
expected_features = model_dict.get('num_features', 42)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Label dictionary
labels_dict = {
    0: 'Hello', 1: 'Rock', 2: 'You', 3: 'Me', 4: 'I Agree', 5: 'I Disagree',
    6: 'Goodbye', 7: 'Thank you', 8: 'Sorry', 9: 'I Love You'
}

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)
previous_prediction = None

# Translator
translator = Translator()

# Font paths (make sure these files are present)
gujarati_font_path = 'Noto_Sans_Gujarati/NotoSansGujarati-VariableFont_wdth,wght.ttf'
hindi_font_path = 'Noto_Sans_Gujarati/NotoSansDevanagari-Regular.ttf'

def draw_text(img, text, position, font_path, font_size, color=(0, 0, 0)):
    try:
        # Convert to PIL image
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        return np.array(img_pil)
    except Exception as e:
        print("Font draw error:", e)
        return img

while True:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to capture image.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        data_aux = np.asarray(data_aux).reshape(1, -1)
        if data_aux.shape[1] != expected_features:
            print(f"Feature count mismatch: expected {expected_features}, got {data_aux.shape[1]}")
            continue

        prediction = model.predict(data_aux)
        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) + 10, int(max(y_) * H) + 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        # Translation
        try:
            gujarati_text = translator.translate(predicted_character, dest='gu').text
            hindi_text = translator.translate(predicted_character, dest='hi').text
        except Exception as e:
            gujarati_text = "?"
            hindi_text = "?"
            print("Translation error:", e)

        # Display English
        cv2.putText(frame, predicted_character, (W // 2 - 100, H - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)

        # Display Gujarati using PIL
        frame = draw_text(frame, f"Gujarati: {gujarati_text}", (50, 50), gujarati_font_path, 30, (0, 100, 0))

        # Display Hindi using PIL
        frame = draw_text(frame, f"Hindi: {hindi_text}", (50, 90), hindi_font_path, 30, (0, 0, 200))

        if predicted_character != previous_prediction:
            previous_prediction = predicted_character
            engine.say(predicted_character)
            engine.runAndWait()

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()