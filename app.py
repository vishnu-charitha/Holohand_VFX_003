import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = st.camera_input("Open Camera")

if camera:

    file_bytes = np.asarray(
        bytearray(camera.read()),
        dtype=np.uint8
    )

    frame = cv2.imdecode(file_bytes, 1)

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape

            x = int(hand_landmarks.landmark[9].x * w)
            y = int(hand_landmarks.landmark[9].y * h)

            # glowing holo rings
            overlay = frame.copy()

            cv2.circle(overlay, (x, y), 90, (255, 255, 0), 5)
            cv2.circle(overlay, (x, y), 60, (255, 255, 0), 3)

            # dots around ring
            for angle in range(0, 360, 20):
                px = int(x + 75 * np.cos(np.radians(angle)))
                py = int(y + 75 * np.sin(np.radians(angle)))
                cv2.circle(
                    overlay,
                    (px, py),
                    5,
                    (255, 255, 0),
                    -1
                )

            frame = cv2.addWeighted(
                overlay,
                0.7,
                frame,
                0.3,
                0
            )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame, use_container_width=True)