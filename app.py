import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase,WebRtcMode

from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
)
from mediapipe.tasks.python.core.base_options import (
    BaseOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

st.set_page_config(
    page_title="HoloHand VFX",
    layout="wide"
)

st.title("HoloHand VFX ⚡")
st.write("Real-Time Hand Tracking Hologram System")

MODEL_PATH = "hand_landmarker.task"


class HoloProcessor(VideoProcessorBase):

    def __init__(self):

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=MODEL_PATH
            ),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=2
        )

        self.detector = (
            HandLandmarker
            .create_from_options(options)
        )

        self.timestamp = 0

    def draw_holo(self, frame, x, y):

        overlay = frame.copy()

        # Hologram circles
        cv2.circle(
            overlay,
            (x, y),
            110,
            (255, 255, 0),
            3
        )

        cv2.circle(
            overlay,
            (x, y),
            85,
            (255, 255, 0),
            2
        )

        cv2.circle(
            overlay,
            (x, y),
            60,
            (255, 255, 255),
            2
        )

        # Rays around hand
        for angle in range(0, 360, 5):

            px = int(
                x + 170 *
                np.cos(np.radians(angle))
            )

            py = int(
                y + 170 *
                np.sin(np.radians(angle))
            )

            cv2.line(
                overlay,
                (x, y),
                (px, py),
                (255, 255, 0),
                1
            )

        # Orbit dots
        for angle in range(0, 360, 8):

            px = int(
                x + 95 *
                np.cos(np.radians(angle))
            )

            py = int(
                y + 95 *
                np.sin(np.radians(angle))
            )

            cv2.circle(
                overlay,
                (px, py),
                5,
                (255, 255, 0),
                -1
            )

        return cv2.addWeighted(
            overlay,
            0.8,
            frame,
            0.2,
            0
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

        h, w, _ = img.shape

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        self.timestamp += 33

        result = (
            self.detector
            .detect_for_video(
                mp_image,
                self.timestamp
            )
        )

        hand_positions = []

        if result.hand_landmarks:

            for hand in (
                result.hand_landmarks
            ):

                palm = hand[9]

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
                y2
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
                    thickness
                )

            for i in range(120):

                t = i / 120

                px = int(
                    x1 +
                    t * (x2 - x1)
                )

                py = int(
                    y1 +
                    t * (y2 - y1)
                )

                jitter_x = (
                    np.random.randint(
                        -20, 20
                    )
                )

                jitter_y = (
                    np.random.randint(
                        -20, 20
                    )
                )

                cv2.circle(
                    overlay,
                    (
                        px + jitter_x,
                        py + jitter_y
                    ),
                    np.random.randint(
                        2, 6
                    ),
                    (255, 255, 255),
                    -1
                )

            img = cv2.addWeighted(
                overlay,
                0.9,
                img,
                0.1,
                0
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


from streamlit_webrtc import webrtc_streamer, WebRtcMode

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
        "video": True,
        "audio": False,
    },
    video_processor_factory=HoloProcessor
)