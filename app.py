import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

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

st.title("Real-Time Hand Tracking Hologram System")

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionTaskRunningMode.VIDEO,
    num_hands=2
)

detector = HandLandmarker.create_from_options(
    options
)

cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()

start_time = time.time()


def draw_holo(frame, x, y):

    overlay = frame.copy()

    # Hologram rings
    cv2.circle(overlay, (x, y), 110, (255, 255, 0), 3)
    cv2.circle(overlay, (x, y), 85, (255, 255, 0), 2)
    cv2.circle(overlay, (x, y), 60, (255, 255, 255), 2)
    cv2.circle(overlay, (x, y), 35, (255, 255, 0), 2)

    # Dense rays around hand
    for angle in range(0, 360, 5):

        px = int(
            x + 170 * np.cos(
                np.radians(angle)
            )
        )

        py = int(
            y + 170 * np.sin(
                np.radians(angle)
            )
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
            x + 95 * np.cos(
                np.radians(angle)
            )
        )

        py = int(
            y + 95 * np.sin(
                np.radians(angle)
            )
        )

        cv2.circle(
            overlay,
            (px, py),
            5,
            (255, 255, 0),
            -1
        )

    # Center glow
    cv2.circle(
        overlay,
        (x, y),
        12,
        (255, 255, 255),
        -1
    )

    return cv2.addWeighted(
        overlay,
        0.8,
        frame,
        0.2,
        0
    )


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    hand_positions = []

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            palm = hand[9]

            x = int(palm.x * w)
            y = int(palm.y * h)

            hand_positions.append((x, y))

            frame = draw_holo(
                frame,
                x,
                y
            )

    # ENERGY BETWEEN TWO HANDS
    if len(hand_positions) == 2:

        (x1, y1), (x2, y2) = hand_positions

        overlay = frame.copy()

        # Big glowing beam
        for thickness in [
            40, 32, 24, 18,
            12, 8, 5, 3
        ]:

            cv2.line(
                overlay,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                thickness
            )

        # Dense particles
        for i in range(120):

            t = i / 120

            px = int(
                x1 + t * (x2 - x1)
            )

            py = int(
                y1 + t * (y2 - y1)
            )

            jitter_x = np.random.randint(
                -20, 20
            )

            jitter_y = np.random.randint(
                -20, 20
            )

            radius = np.random.randint(
                2, 7
            )

            cv2.circle(
                overlay,
                (
                    px + jitter_x,
                    py + jitter_y
                ),
                radius,
                (255, 255, 255),
                -1
            )

        # Extra sci-fi rays
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        for angle in range(0, 360, 12):

            ray_x = int(
                mid_x +
                150 * np.cos(
                    np.radians(angle)
                )
            )

            ray_y = int(
                mid_y +
                150 * np.sin(
                    np.radians(angle)
                )
            )

            cv2.line(
                overlay,
                (mid_x, mid_y),
                (ray_x, ray_y),
                (255, 255, 0),
                2
            )

        frame = cv2.addWeighted(
            overlay,
            0.9,
            frame,
            0.1,
            0
        )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_placeholder.image(
        frame,
        use_container_width=True
    )

cap.release()