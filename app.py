import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("HoloHand VFX")
st.write("AI-powered holographic VFX demo")

camera = st.camera_input("Open Camera")

if camera:
    image = Image.open(camera).convert("RGB")

    img = image.copy()
    draw = ImageDraw.Draw(img)

    width, height = img.size
    cx = width // 2
    cy = height // 2

    # HUGE neon hologram rings
    for radius in [100, 180, 260]:

        # Glow effect
        for glow in range(25, 0, -5):
            draw.ellipse(
                (
                    cx - radius - glow,
                    cy - radius - glow,
                    cx + radius + glow,
                    cy + radius + glow
                ),
                outline=(0, 255, 255),
                width=8
            )

    # Strong light beams
    for angle in range(0, 360, 10):
        x = cx + int(np.cos(np.radians(angle)) * 320)
        y = cy + int(np.sin(np.radians(angle)) * 320)

        draw.line(
            [(cx, cy), (x, y)],
            fill=(0, 255, 255),
            width=6
        )

    # Bright center hologram orb
    for r in range(70, 5, -5):
        draw.ellipse(
            (
                cx - r,
                cy - r,
                cx + r,
                cy + r
            ),
            fill=(0, 255, 255)
        )

    st.image(img, use_container_width=True)
    st.success("HOLO VFX ACTIVE ✅")