import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("HoloHand VFX Demo")

camera = st.camera_input("Open Camera")

if camera is not None:

    image = Image.open(camera).convert("RGB")

    width, height = image.size
    cx = width // 2
    cy = height // 2

    draw = ImageDraw.Draw(image)

    # HUGE visible red hologram circle
    draw.ellipse(
        (
            cx - 200,
            cy - 200,
            cx + 200,
            cy + 200
        ),
        outline="red",
        width=25
    )

    # second ring
    draw.ellipse(
        (
            cx - 120,
            cy - 120,
            cx + 120,
            cy + 120
        ),
        outline="cyan",
        width=20
    )

    # cross beams
    draw.line((0, cy, width, cy), fill="yellow", width=10)
    draw.line((cx, 0, cx, height), fill="yellow", width=10)

    st.subheader("HOLO VFX OUTPUT")

    # FORCE processed image display
    st.image(image, use_container_width=True)

    st.success("HOLO EFFECT APPLIED ✅")