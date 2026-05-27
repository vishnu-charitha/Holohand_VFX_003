import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        st.error("Camera not working")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            x = int(hand_landmarks.landmark[9].x * w)
            y = int(hand_landmarks.landmark[9].y * h)

            # HOLO RINGS
            cv2.circle(frame, (x, y), 80, (255,255,0), 3)
            cv2.circle(frame, (x, y), 50, (255,255,0), 2)

            # glow dots
            for angle in range(0, 360, 20):
                px = int(x + 60 * np.cos(np.radians(angle)))
                py = int(y + 60 * np.sin(np.radians(angle)))
                cv2.circle(frame, (px, py), 6, (255,255,0), -1)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame, use_container_width=True)

cap.release()