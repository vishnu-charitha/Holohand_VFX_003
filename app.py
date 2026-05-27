import streamlit as st
from PIL import Image, ImageDraw
import mediapipe as mp
import numpy as np

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")

camera = st.camera_input("Open Camera")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2
)

if camera:

    image = Image.open(camera).convert("RGB")
    img = np.array(image)

    results = hands.process(img)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    h, w, _ = img.shape

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Palm center
            x = int(hand_landmarks.landmark[9].x * w)
            y = int(hand_landmarks.landmark[9].y * h)

            # Neon hologram rings
            for r in [60, 90, 120]:

                draw.ellipse(
                    (
                        x-r,
                        y-r,
                        x+r,
                        y+r
                    ),
                    outline=(0,255,255),
                    width=8
                )

            # Energy lines
            for angle in range(0, 360, 30):

                x2 = x + int(np.cos(np.radians(angle))*140)
                y2 = y + int(np.sin(np.radians(angle))*140)

                draw.line(
                    [(x,y),(x2,y2)],
                    fill=(0,255,255),
                    width=5
                )

    st.image(pil_img, use_container_width=True)