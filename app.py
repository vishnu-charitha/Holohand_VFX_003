import streamlit as st
from PIL import Image, ImageDraw

st.title("HoloHand VFX Test")

camera = st.camera_input("Open Camera")

if camera:
    image = Image.open(camera).convert("RGB")

    draw = ImageDraw.Draw(image)

    w, h = image.size
    cx, cy = w // 2, h // 2

    # giant red ring (easy to see)
    draw.ellipse(
        (cx - 150, cy - 150, cx + 150, cy + 150),
        outline="red",
        width=20
    )

    draw.text((50, 50), "HOLO VFX ACTIVE", fill="yellow")

    st.image(image)