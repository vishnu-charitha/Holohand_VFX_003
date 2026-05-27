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

    # BIG glowing hologram rings
    ring_colors = [
        (0, 255, 255),
        (0, 220, 255),
        (120, 255, 255)
    ]

    for radius in [80, 140, 200]:
        for glow in range(8):
            draw.ellipse(
                (
                    cx - radius - glow,
                    cy - radius - glow,
                    cx + radius + glow,
                    cy + radius + glow
                ),
                outline=ring_colors[glow % len(ring_colors)],
                width=5
            )

    # Bright hologram beams
    for angle in range(0, 360, 15):
        x = cx + int(np.cos(np.radians(angle)) * 250)
        y = cy + int(np.sin(np.radians(angle)) * 250)

        draw.line(
            [(cx, cy), (x, y)],
            fill=(0, 255, 255),
            width=4
        )

    # Center glowing orb
    for r in range(50, 5, -5):
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
    st.success("Holo VFX Applied Successfully ✅")