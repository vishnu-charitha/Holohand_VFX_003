import cv2
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")
st.write("AI-powered hand tracking hologram effect using OpenCV + MediaPipe")

mp_hands = mp.solutions.hands


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

    def draw_glow_line(self, img, pt1, pt2, color=(255, 255, 0)):
        for thickness in range(12, 2, -2):
            overlay = img.copy()
            cv2.line(overlay, pt1, pt2, color, thickness)
            cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)
        cv2.line(img, pt1, pt2, color, 2)

    def draw_light_point(self, img, center, color=(220, 255, 255)):
        for radius, alpha in [(16, 0.06), (10, 0.10), (5, 0.18)]:
            overlay = img.copy()
            cv2.circle(overlay, center, radius, color, -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.circle(img, center, 3, (255, 255, 255), -1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        h, w, _ = img.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = []

                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append((px, py))

                fingers = [4, 8, 12, 16, 20]

                for i in range(len(fingers) - 1):
                    self.draw_glow_line(
                        img,
                        points[fingers[i]],
                        points[fingers[i + 1]]
                    )

                for point in points:
                    self.draw_light_point(img, point)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="holohand",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)