import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

st.set_page_config(
    page_title="HoloHand VFX",
    layout="wide"
)

st.title("HoloHand VFX ⚡")
st.write("Real-Time Hand Tracking Hologram System")

mp_hands = mp.solutions.hands


class HoloProcessor(VideoProcessorBase):

    def __init__(self):

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def draw_holo(self, frame, x, y):

        overlay = frame.copy()

        # Main circles
        cv2.circle(overlay, (x, y), 110, (255, 255, 0), 3)
        cv2.circle(overlay, (x, y), 85, (255, 255, 0), 2)
        cv2.circle(overlay, (x, y), 60, (255, 255, 255), 2)

        # More rays
        for angle in range(0, 360, 3):

            px = int(
                x + 180 * np.cos(np.radians(angle))
            )

            py = int(
                y + 180 * np.sin(np.radians(angle))
            )

            cv2.line(
                overlay,
                (x, y),
                (px, py),
                (255, 255, 0),
                1,
            )

        # Orbit dots
        for angle in range(0, 360, 6):

            px = int(
                x + 95 * np.cos(np.radians(angle))
            )

            py = int(
                y + 95 * np.sin(np.radians(angle))
            )

            cv2.circle(
                overlay,
                (px, py),
                5,
                (255, 255, 0),
                -1,
            )

        return cv2.addWeighted(
            overlay,
            0.8,
            frame,
            0.2,
            0,
        )

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        result = self.hands.process(rgb)

        h, w, _ = img.shape
        hand_positions = []

        if result.multi_hand_landmarks:

            for hand_landmarks in (
                result.multi_hand_landmarks
            ):

                palm = hand_landmarks.landmark[9]

                x = int(palm.x * w)
                y = int(palm.y * h)

                hand_positions.append(
                    (x, y)
                )

                img = self.draw_holo(
                    img,
                    x,
                    y
                )

        # Energy between hands
        if len(hand_positions) == 2:

            (x1, y1), (
                x2,
                y2,
            ) = hand_positions

            overlay = img.copy()

            for thickness in [
                40, 32, 24,
                18, 12, 8,
                5, 3
            ]:

                cv2.line(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    thickness,
                )

            # More rays between hands
            for i in range(200):

                t = i / 200

                px = int(
                    x1 +
                    t * (x2 - x1)
                )

                py = int(
                    y1 +
                    t * (y2 - y1)
                )

                jitter_x = np.random.randint(
                    -30, 30
                )

                jitter_y = np.random.randint(
                    -30, 30
                )

                cv2.line(
                    overlay,
                    (px, py),
                    (
                        px + jitter_x,
                        py + jitter_y,
                    ),
                    (255, 255, 255),
                    2,
                )

            img = cv2.addWeighted(
                overlay,
                0.9,
                img,
                0.1,
                0,
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="holo",
    mode=WebRtcMode.SENDRECV,

    rtc_configuration={
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    },

    media_stream_constraints={
        "video": {
            "width": 640,
            "height": 480,
            "frameRate": 30,
        },
        "audio": False,
    },

    async_processing=True,

    video_processor_factory=HoloProcessor,
)