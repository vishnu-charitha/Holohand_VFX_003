import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")
st.write("AI-powered holographic hand tracking using OpenCV + MediaPipe")

camera = st.camera_input("Open Camera")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

if camera:
    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                h, w, _ = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)

                cv2.circle(image, (x, y), 8, (0, 255, 0), -1)

    st.image(image, channels="BGR")