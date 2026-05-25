import os
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"


def _get_model_path() -> str:
    script_dir = Path(__file__).resolve().parent
    model_path = script_dir / MODEL_FILENAME

    if model_path.exists():
        return str(model_path)

    print(f"Downloading MediaPipe hand landmarker model to {model_path}...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return str(model_path)


use_tasks_api = False
try:
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        raise AttributeError("mediapipe.solutions is not available")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils
except (AttributeError, ImportError):
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.core import base_options as mp_base_options
    from mediapipe.tasks.python.vision.core import vision_task_running_mode

    model_path = _get_model_path()
    options = HandLandmarkerOptions(
        base_options=mp_base_options.BaseOptions(model_asset_path=model_path),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    hands = HandLandmarker.create_from_options(options)
    use_tasks_api = True
    video_start_time = time.perf_counter()

# Webcam
cap = cv2.VideoCapture(0)

# Function to draw glowing line
def draw_glow_line(img, pt1, pt2, color=(255, 255, 0)):
    for thickness in range(12, 2, -2):
        overlay = img.copy()
        cv2.line(overlay, pt1, pt2, color, thickness)
        alpha = 0.08
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.line(img, pt1, pt2, color, 2)


# Function to draw glowing landmark points
def draw_light_point(img, center, color=(220, 255, 255)):
    for radius, alpha in [(16, 0.06), (10, 0.10), (5, 0.18)]:
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.circle(img, center, 3, (255, 255, 255), -1)


# Function to draw a holographic 3D lighting effect
def draw_hologram(img, center):
    x, y = center

    for r, thickness, alpha, color in [
        (180, 2, 0.04, (120, 230, 255)),
        (135, 2, 0.06, (180, 255, 240)),
        (90, 2, 0.08, (255, 255, 220)),
    ]:
        overlay = img.copy()
        cv2.circle(overlay, (x, y), r, color, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    beam_colors = [
        (255, 180, 120),
        (120, 255, 255),
        (190, 120, 255),
        (255, 255, 120),
        (120, 255, 180),
    ]

    for i in range(20):
        angle = i * 18
        length = 120 + (i % 5) * 10
        dx = int(np.cos(np.deg2rad(angle)) * length)
        dy = int(np.sin(np.deg2rad(angle)) * length)
        end_point = (x + dx, y + dy)
        color = beam_colors[i % len(beam_colors)]
        overlay = img.copy()
        cv2.line(overlay, (x, y), end_point, color, 1)
        alpha = 0.04 + (i % 3) * 0.01
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for i in range(12):
        angle = i * 30 + 15
        inner = 40 + (i % 3) * 10
        outer = 140 + (i % 2) * 30
        start = (
            x + int(np.cos(np.deg2rad(angle)) * inner),
            y + int(np.sin(np.deg2rad(angle)) * inner),
        )
        end = (
            x + int(np.cos(np.deg2rad(angle)) * outer),
            y + int(np.sin(np.deg2rad(angle)) * outer),
        )
        overlay = img.copy()
        cv2.line(overlay, start, end, beam_colors[(i + 2) % len(beam_colors)], 1)
        cv2.addWeighted(overlay, 0.03, img, 0.97, 0, img)

    for r, alpha, color in [
        (110, 0.18, (255, 255, 255)),
        (70, 0.22, (180, 255, 255)),
        (40, 0.16, (255, 180, 255)),
    ]:
        overlay = img.copy()
        cv2.circle(overlay, (x, y), r, color, 2)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    for i in range(8):
        angle = i * 45
        radius = 55 + (i % 2) * 12
        light_x = x + int(np.cos(np.deg2rad(angle)) * radius)
        light_y = y + int(np.sin(np.deg2rad(angle)) * radius)
        draw_light_point(img, (light_x, light_y), color=beam_colors[i % len(beam_colors)])

    cv2.circle(img, (x, y), 35, (255, 255, 255), 2)


while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if use_tasks_api:
        image = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
        timestamp_ms = int((time.perf_counter() - video_start_time) * 1000)
        result = hands.detect_for_video(image, timestamp_ms)
        hand_landmarks_list = result.hand_landmarks
    else:
        result = hands.process(rgb)
        hand_landmarks_list = result.multi_hand_landmarks

    hand_centers = []

    if hand_landmarks_list:
        for hand_landmarks in hand_landmarks_list:

            points = []

            # Get hand landmark points
            if use_tasks_api:
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append((px, py))
            else:
                for lm in hand_landmarks.landmark:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    points.append((px, py))

            # Draw glowing finger connections
            fingers = [4, 8, 12, 16, 20]

            for i in range(len(fingers) - 1):
                p1 = points[fingers[i]]
                p2 = points[fingers[i + 1]]
                draw_glow_line(frame, p1, p2)

            # Hand center
            cx = int(np.mean([p[0] for p in points]))
            cy = int(np.mean([p[1] for p in points]))

            hand_centers.append((cx, cy))

            # Draw glowing hand joints
            for point in points:
                draw_light_point(frame, point)

            # Draw a subtle 3D ring around each hand center
            draw_hologram(frame, (cx, cy))

    # If 2 hands detected → create hologram between hands
    if len(hand_centers) == 2:

        center_x = (
            hand_centers[0][0] +
            hand_centers[1][0]
        ) // 2

        center_y = (
            hand_centers[0][1] +
            hand_centers[1][1]
        ) // 2

        draw_hologram(
            frame,
            (center_x, center_y)
        )

        # Connect hands to hologram with colored beams
        beam_colors = [(255, 180, 120), (120, 255, 255), (190, 120, 255)]
        for idx, center in enumerate(hand_centers):
            color = beam_colors[idx % len(beam_colors)]
            draw_glow_line(frame, center, (center_x, center_y), color=color)
            for offset in [(-10, -10), (10, -10), (-10, 10), (10, 10)]:
                jitter_end = (center_x + offset[0], center_y + offset[1])
                draw_glow_line(frame, center, jitter_end, color=color)

    cv2.imshow(
        "Hand Tracking Hologram",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

if use_tasks_api:
    hands.close()

cap.release()
cv2.destroyAllWindows()