import streamlit as st

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")
st.write("AI-powered holographic hand tracking using OpenCV + MediaPipe")

camera = st.camera_input("Open Camera")
if camera:
    st.success("Camera connected successfully!")