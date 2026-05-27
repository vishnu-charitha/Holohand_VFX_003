import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("HoloHand VFX")
st.write("AI-powered holographic hand tracking")

camera = st.camera_input("Open Camera")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2
)

if camera:

    image = Image.open(camera).convert("RGB")
    frame = np.array(image)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # palm center landmark
            palm = hand_landmarks.landmark[9]

            cx = int(palm.x * w)
            cy = int(palm.y * h)

            # glowing hologram rings
            for radius in [60, 90, 120]:
                cv2.circle(
                    frame,
                    (cx, cy),
                    radius,
                    (255, 255, 0),
                    3
                )

            # energy beams
            for angle in range(0, 360, 30):
                x2 = cx + int(np.cos(np.radians(angle)) * 140)
                y2 = cy + int(np.sin(np.radians(angle)) * 140)

                cv2.line(
                    frame,
                    (cx, cy),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

            # glowing landmarks
            for lm in hand_landmarks.landmark:
                px = int(lm.x * w)
                py = int(lm.y * h)

                cv2.circle(
                    frame,
                    (px, py),
                    8,
                    (0, 255, 255),
                    -1
                )

st.image(frame, use_container_width=True)