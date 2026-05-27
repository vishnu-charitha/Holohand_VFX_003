import streamlit as st

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("HoloHand VFX")
st.success("Deployment Successful ✅")

st.write("Real-time hologram hand tracking demo project")

camera = st.camera_input("Test Camera")

if camera:
    st.image(camera)
    st.success("Camera working successfully!")