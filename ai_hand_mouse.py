import cv2
import mediapipe as mp
import pyautogui
import time
import subprocess
import math
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False

# =====================================
# MEDIAPIPE SETUP
# =====================================
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = vision.HandLandmarker.create_from_options(options)

# =====================================
# CAMERA + SCREEN
# =====================================
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()
time.sleep(1)

# =====================================
# GLOBAL VARIABLES
# =====================================
prev_x, prev_y = 0, 0
prev_angle = 0
trail_points = []

dragging = False
lock_mode = False

last_click_time = 0
last_right_click_time = 0
last_volume_time = 0

click_threshold = 0.03
click_cooldown = 0.3
volume_cooldown = 0.5

margin = 60

# =====================================
# DRAW HAND GRID + ANGLE
# =====================================
def draw_hand_grid(frame, hand_landmarks):
    h, w, _ = frame.shape

    wrist = hand_landmarks[0]
    index_base = hand_landmarks[5]

    x1, y1 = int(wrist.x * w), int(wrist.y * h)
    x2, y2 = int(index_base.x * w), int(index_base.y * h)

    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

    xs = [int(lm.x * w) for lm in hand_landmarks]
    ys = [int(lm.y * h) for lm in hand_landmarks]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # box
    cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 255), 2)

    # grid
    for i in range(1, 3):
        x = int(min_x + (max_x - min_x) * i / 3)
        cv2.line(frame, (x, min_y), (x, max_y), (255, 0, 255), 1)

    for j in range(1, 3):
        y = int(min_y + (max_y - min_y) * j / 3)
        cv2.line(frame, (min_x, y), (max_x, y), (255, 0, 255), 1)

    cv2.putText(frame, f"Angle: {int(angle)}",
                (min_x, min_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 255), 2)

    return angle

# =====================================
# DRAW TRAIL
# =====================================
def draw_trail(frame):
    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i-1], trail_points[i], (0,255,255), 2)

print("🚀 Vision-Pro Hand Mouse Running — press q to quit")

prev_time = time.time()

# =====================================
# MAIN LOOP
# =====================================
while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = landmarker.detect(mp_image)

    # Active control area
    cv2.rectangle(frame,
                  (margin, margin),
                  (frame_w-margin, frame_h-margin),
                  (0,255,0), 2)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            angle = draw_hand_grid(frame, hand_landmarks)

            wrist = hand_landmarks[0]
            index_tip = hand_landmarks[8]
            thumb_tip = hand_landmarks[4]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]

            # =====================================
            # LOCK MODE (OPEN PALM DETECTION)
            # =====================================
            palm_open = (
                hand_landmarks[8].y < hand_landmarks[6].y and
                hand_landmarks[12].y < hand_landmarks[10].y and
                hand_landmarks[16].y < hand_landmarks[14].y and
                hand_landmarks[20].y < hand_landmarks[18].y
            )

            lock_mode = palm_open

            # =====================================
            # MAP TO SCREEN
            # =====================================
            cam_x = np.interp(index_tip.x * frame_w,
                              (margin, frame_w - margin),
                              (0, screen_width))

            cam_y = np.interp(index_tip.y * frame_h,
                              (margin, frame_h - margin),
                              (0, screen_height))

            x, y = int(cam_x), int(cam_y)

            # =====================================
            # ULTRA STABLE SMOOTHING
            # =====================================
            angle_diff = abs(angle - prev_angle)

            if angle_diff > 25:
                alpha = 0.12
            else:
                speed = math.hypot(x-prev_x, y-prev_y)
                if speed < 40:
                    alpha = 0.06
                elif speed < 120:
                    alpha = 0.18
                else:
                    alpha = 0.35

            prev_angle = angle

            smoothed_x = prev_x + (x-prev_x)*alpha
            smoothed_y = prev_y + (y-prev_y)*alpha
            prev_x, prev_y = smoothed_x, smoothed_y

            # MOVE CURSOR
            if not lock_mode:
                pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
            else:
                prev_x, prev_y = x, y

            # =====================================
            # TRAIL VISUAL
            # =====================================
            trail_points.append((int(index_tip.x*frame_w),
                                 int(index_tip.y*frame_h)))
            if len(trail_points) > 20:
                trail_points.pop(0)

            # =====================================
            # LEFT CLICK (Thumb + Index)
            # =====================================
            distance = math.hypot(thumb_tip.x-index_tip.x,
                                  thumb_tip.y-index_tip.y)

            if distance < click_threshold and time.time()-last_click_time > click_cooldown:
                pyautogui.click()
                last_click_time = time.time()

            # =====================================
            # RIGHT CLICK (Index + Ring)
            # =====================================
            distance_right = math.hypot(ring_tip.x-index_tip.x,
                                        ring_tip.y-index_tip.y)

            if distance_right < click_threshold and time.time()-last_right_click_time > click_cooldown:
                pyautogui.rightClick()
                last_right_click_time = time.time()

            # =====================================
            # DRAG & DROP
            # =====================================
            if distance < click_threshold and not dragging:
                pyautogui.mouseDown()
                dragging = True
            elif distance >= click_threshold and dragging:
                pyautogui.mouseUp()
                dragging = False

            # =====================================
            # SCROLL (INDEX + MIDDLE CLOSE)
            # =====================================
            scroll_distance = abs(index_tip.y - middle_tip.y)
            if scroll_distance < 0.02:
                pyautogui.scroll(int((prev_y - smoothed_y) * 0.5))

            # =====================================
            # MAC VOLUME CONTROL
            # =====================================
            vol_up = math.hypot(thumb_tip.x-pinky_tip.x,
                                thumb_tip.y-pinky_tip.y)

            if vol_up < click_threshold and time.time()-last_volume_time > volume_cooldown:
                subprocess.run(['osascript','-e',
                'set volume output volume (output volume of (get volume settings) + 10)'])
                last_volume_time = time.time()

            vol_down = math.hypot(index_tip.x-pinky_tip.x,
                                  index_tip.y-pinky_tip.y)

            if vol_down < click_threshold and time.time()-last_volume_time > volume_cooldown:
                subprocess.run(['osascript','-e',
                'set volume output volume (output volume of (get volume settings) - 10)'])
                last_volume_time = time.time()

    draw_trail(frame)

    # FPS
    curr_time = time.time()
    fps = 1/(curr_time-prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.putText(frame,"Open Palm = LOCK MODE",
                (10,frame_h-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    cv2.imshow("VISION PRO HAND MOUSE", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
landmarker.close()
