import cv2
import av
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="HoloHand VFX", layout="wide")

st.title("Real-Time Hand Tracking Hologram System")
st.write("AI-powered holographic hand tracking using OpenCV + MediaPipe")


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        self.hands = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path="hand_landmarker.task"
                ),
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                min_hand_presence_confidence=0.5,
            )
        )

    def draw_glow_line(self, img, pt1, pt2, color=(255, 255, 0)):
        for thickness in range(18, 2, -2):
            overlay = img.copy()
            cv2.line(overlay, pt1, pt2, color, thickness)
            cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)

        cv2.line(img, pt1, pt2, color, 3)

    def draw_light_point(self, img, center):
        glow_colors = [
            (255, 255, 255),
            (255, 200, 255),
            (180, 255, 255),
        ]

        for radius in [22, 15, 8]:
            overlay = img.copy()

            cv2.circle(
                overlay,
                center,
                radius,
                glow_colors[radius % len(glow_colors)],
                -1
            )

            cv2.addWeighted(
                overlay,
                0.15,
                img,
                0.85,
                0,
                img
            )

        cv2.circle(img, center, 4, (255, 255, 255), -1)

    def draw_hologram(self, img, center):
        x, y = center

        rings = [
            (180, (120, 255, 255)),
            (130, (255, 120, 255)),
            (90, (255, 255, 200)),
        ]

        for radius, color in rings:
            overlay = img.copy()

            cv2.circle(
                overlay,
                (x, y),
                radius,
                color,
                2
            )

            cv2.addWeighted(
                overlay,
                0.14,
                img,
                0.86,
                0,
                img
            )

        beam_colors = [
            (255, 180, 120),
            (120, 255, 255),
            (190, 120, 255),
        ]

        for i in range(20):
            angle = i * 18
            length = 150

            dx = int(np.cos(np.radians(angle)) * length)
            dy = int(np.sin(np.radians(angle)) * length)

            end_point = (x + dx, y + dy)

            overlay = img.copy()

            cv2.line(
                overlay,
                (x, y),
                end_point,
                beam_colors[i % 3],
                2
            )

            cv2.addWeighted(
                overlay,
                0.08,
                img,
                0.92,
                0,
                img
            )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Mirror webcam
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = self.hands.detect(mp_image)

        h, w, _ = img.shape
        hand_centers = []

        if results.hand_landmarks:

            for hand_landmarks in results.hand_landmarks:

                points = []

                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append((px, py))

                if len(points) < 21:
                    continue

                fingers = [4, 8, 12, 16, 20]

                # Finger beams
                for i in range(len(fingers) - 1):
                    self.draw_glow_line(
                        img,
                        points[fingers[i]],
                        points[fingers[i + 1]],
                        color=(255, 255, 0)
                    )

                # Hand joints
                for point in points:
                    self.draw_light_point(img, point)

                # Hand center
                cx = int(np.mean([p[0] for p in points]))
                cy = int(np.mean([p[1] for p in points]))

                hand_centers.append((cx, cy))

                # Hologram around hand
                self.draw_hologram(
                    img,
                    (cx, cy)
                )

        # Two-hand hologram mode
        if len(hand_centers) >= 2:

            center_x = (
                hand_centers[0][0] +
                hand_centers[1][0]
            ) // 2

            center_y = (
                hand_centers[0][1] +
                hand_centers[1][1]
            ) // 2

            middle_center = (center_x, center_y)

            # Big center hologram
            self.draw_hologram(
                img,
                middle_center
            )

            # Connect both hands
            beam_colors = [
                (0, 255, 255),
                (255, 0, 255),
            ]

            for idx, center in enumerate(hand_centers[:2]):
                self.draw_glow_line(
                    img,
                    center,
                    middle_center,
                    color=beam_colors[idx]
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


webrtc_streamer(
    key="holohand",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)