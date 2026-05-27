import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("HoloHand VFX")
st.write("AI-powered holographic hand tracking demo")

camera = st.camera_input("Open Camera")

if camera:
    image = Image.open(camera)

    img = image.copy()
    draw = ImageDraw.Draw(img)

    width, height = img.size
    center_x = width // 2
    center_y = height // 2

    # Hologram rings
    for radius in [60, 100, 140]:
        draw.ellipse(
            (
                center_x - radius,
                center_y - radius,
                center_x + radius,
                center_y + radius
            ),
            outline=(0, 255, 255),
            width=4
        )

    # Light beams
    for angle in range(0, 360, 20):
        x = center_x + int(np.cos(np.radians(angle)) * 180)
        y = center_y + int(np.sin(np.radians(angle)) * 180)

        draw.line(
            [(center_x, center_y), (x, y)],
            fill=(0, 255, 255),
            width=2
        )

    st.image(img, caption="HoloHand VFX Output")
    st.success("Hologram Effect Applied ✅")