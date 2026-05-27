import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")

camera = st.camera_input("Open Camera")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

if camera:
    image = Image.open(camera)
    image_np = np.array(image)

    results = hands.process(image_np)

    draw = ImageDraw.Draw(image)

    if results.multi_hand_landmarks:
        w, h = image.size

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                draw.ellipse(
                    (x - 5, y - 5, x + 5, y + 5),
                    fill="lime"
                )

    st.image(image)