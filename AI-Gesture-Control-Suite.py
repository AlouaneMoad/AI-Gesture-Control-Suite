import cv2
import mediapipe as mp
from pynput.mouse import Button, Controller
import math
import numpy as np
import time
from screeninfo import get_monitors
import platform
import subprocess

# --- Platform-Specific Volume Control (No changes needed here) ---
OS_PLATFORM = platform.system()
if OS_PLATFORM == "Windows":
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL

    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_controller = interface.QueryInterface(IAudioEndpointVolume)
        VOL_RANGE = volume_controller.GetVolumeRange()
        MIN_VOL, MAX_VOL = VOL_RANGE[0], VOL_RANGE[1]
    except Exception as e:
        print(f"Could not initialize Windows volume control: {e}")
        volume_controller = None
else:
    volume_controller = None

# --- Configuration Constants ---
WEBCAM_ID = 0
CAM_WIDTH, CAM_HEIGHT = 640, 480
SMOOTHING = 5
FRAME_REDUCTION = 100

# Gesture thresholds
PINCH_THRESHOLD = 35
DOUBLE_PINCH_CLUSTER_RADIUS = 40
CLICK_TAP_DURATION = 0.2
DRAG_START_DURATION = 0.25

# --- Global Variables ---
prev_loc_x, prev_loc_y = 0, 0
curr_loc_x, curr_loc_y = 0, 0
is_dragging = False
is_pinching = False
pinch_start_time = 0
last_action_time = 0
ACTION_COOLDOWN = 0.6

# Screen dimensions
try:
    monitor = get_monitors()[0]
    SCREEN_WIDTH, SCREEN_HEIGHT = monitor.width, monitor.height
except Exception:
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

mouse = Controller()


def set_system_volume(level_percent):
    """Sets system volume cross-platform."""
    try:
        if OS_PLATFORM == "Windows" and volume_controller:
            vol_level = np.interp(level_percent, [0, 100], [MIN_VOL, MAX_VOL])
            volume_controller.SetMasterVolumeLevel(vol_level, None)
        elif OS_PLATFORM == "Darwin":  # macOS
            subprocess.run(["osascript", "-e", f"set volume output volume {level_percent}"], check=True)
        elif OS_PLATFORM == "Linux":
            subprocess.run(["amixer", "-D", "pulse", "sset", "Master", f"{level_percent}%"], check=True)
    except Exception as e:
        print(f"Error setting volume: {e}")


def main():
    """Main function for the redesigned gesture control suite."""
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
    ) as hands:
        mp_draw = mp.solutions.drawing_utils

        while True:
            success, img = cap.read()
            if not success: continue

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION),
                          (CAM_WIDTH - FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION),
                          (255, 0, 255), 2)

            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)

                # --- TWO HAND GESTURES ---
                if num_hands == 2:
                    hand1_lm = get_landmark_list(results.multi_hand_landmarks[0], img)
                    hand2_lm = get_landmark_list(results.multi_hand_landmarks[1], img)

                    hand1_handedness = results.multi_handedness[0].classification[0].label
                    hand2_handedness = results.multi_handedness[1].classification[0].label

                    fingers1 = count_fingers_up(hand1_lm, hand1_handedness)
                    fingers2 = count_fingers_up(hand2_lm, hand2_handedness)

                    if fingers1 == [0, 1, 0, 0, 0] and fingers2 == [0, 1, 0, 0, 0]:
                        handle_volume_control_two_hands(img, hand1_lm, hand2_lm)
                    else:
                        reset_all_actions()

                # --- ONE HAND GESTURES ---
                elif num_hands == 1:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    handedness = results.multi_handedness[0].classification[0].label
                    lm_list = get_landmark_list(hand_landmarks, img)
                    fingers = count_fingers_up(lm_list, handedness)

                    index_tip = lm_list[8][1:]
                    thumb_tip = lm_list[4][1:]
                    middle_tip = lm_list[12][1:]

                    pinch_dist = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])

                    if is_double_pinch(thumb_tip, index_tip, middle_tip):
                        handle_double_click(img, middle_tip)
                    elif pinch_dist < PINCH_THRESHOLD:
                        handle_pinch_click_drag(img, index_tip)
                    elif fingers == [0, 1, 0, 0, 0]:
                        handle_mouse_movement(img, index_tip)
                    else:
                        reset_all_actions()

                for hand_lms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            else:
                reset_all_actions()

            cv2.imshow("AI Gesture Control Suite - Press 'q' to quit", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def get_landmark_list(hand_landmarks, img):
    return [[lm_id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])] for lm_id, lm in
            enumerate(hand_landmarks.landmark)]


def count_fingers_up(lm_list, handedness):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]

    if handedness == 'Right':
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # Left hand
        if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    for i in range(1, 5):
        if lm_list[tip_ids[i]][2] < lm_list[tip_ids[i] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def is_double_pinch(thumb, index, middle):
    centroid_x = (thumb[0] + index[0] + middle[0]) / 3
    centroid_y = (thumb[1] + index[1] + middle[1]) / 3

    dist_thumb = math.hypot(thumb[0] - centroid_x, thumb[1] - centroid_y)
    dist_index = math.hypot(index[0] - centroid_x, index[1] - centroid_y)
    dist_middle = math.hypot(middle[0] - centroid_x, middle[1] - centroid_y)

    return max(dist_thumb, dist_index, dist_middle) < DOUBLE_PINCH_CLUSTER_RADIUS


def handle_volume_control_two_hands(img, lm_list1, lm_list2):
    reset_all_actions()
    index1, index2 = lm_list1[8][1:], lm_list2[8][1:]
    distance = math.hypot(index1[0] - index2[0], index1[1] - index2[1])
    vol_percent = np.interp(distance, [30, 300], [0, 100])
    set_system_volume(vol_percent)

    cv2.line(img, (index1[0], index1[1]), (index2[0], index2[1]), (0, 255, 255), 3)
    cv2.circle(img, (index1[0], index1[1]), 10, (0, 255, 255), cv2.FILLED)
    cv2.circle(img, (index2[0], index2[1]), 10, (0, 255, 255), cv2.FILLED)
    bar_height = int(np.interp(vol_percent, [0, 100], [400, 150]))
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, bar_height), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol_percent)} %', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


def handle_double_click(img, middle_tip):
    global last_action_time
    reset_all_actions()
    if (time.time() - last_action_time) > ACTION_COOLDOWN:
        mouse.click(Button.left, 2)
        last_action_time = time.time()
        cv2.circle(img, (middle_tip[0], middle_tip[1]), 15, (255, 255, 0), cv2.FILLED)


def handle_pinch_click_drag(img, index_tip):
    global is_pinching, pinch_start_time, is_dragging
    if not is_pinching:
        is_pinching = True
        pinch_start_time = time.time()

    if (time.time() - pinch_start_time) > DRAG_START_DURATION and not is_dragging:
        is_dragging = True
        mouse.press(Button.left)

    if is_dragging:
        handle_mouse_movement(img, index_tip, dragging=True)
        cv2.circle(img, (index_tip[0], index_tip[1]), 15, (0, 255, 0), cv2.FILLED)
    else:
        cv2.circle(img, (index_tip[0], index_tip[1]), 15, (0, 255, 255), cv2.FILLED)


def handle_mouse_movement(img, index_tip, dragging=False):
    global prev_loc_x, prev_loc_y, curr_loc_x, curr_loc_y
    if not dragging: reset_all_actions(keep_drag=True)

    x_mapped = np.interp(index_tip[0], (FRAME_REDUCTION, CAM_WIDTH - FRAME_REDUCTION), (0, SCREEN_WIDTH))
    y_mapped = np.interp(index_tip[1], (FRAME_REDUCTION, CAM_HEIGHT - FRAME_REDUCTION), (0, SCREEN_HEIGHT))
    curr_loc_x = prev_loc_x + (x_mapped - prev_loc_x) / SMOOTHING
    curr_loc_y = prev_loc_y + (y_mapped - prev_loc_y) / SMOOTHING
    mouse.position = (int(curr_loc_x), int(curr_loc_y))
    prev_loc_x, prev_loc_y = curr_loc_x, curr_loc_y

    if not is_dragging:
        cv2.circle(img, (index_tip[0], index_tip[1]), 15, (255, 0, 255), cv2.FILLED)


def reset_all_actions(keep_drag=False):
    global is_pinching, pinch_start_time, is_dragging, last_action_time

    if is_pinching:
        pinch_duration = time.time() - pinch_start_time
        if not is_dragging and pinch_duration < CLICK_TAP_DURATION:
            if (time.time() - last_action_time) > ACTION_COOLDOWN:
                mouse.click(Button.left, 1)
                last_action_time = time.time()

    if not keep_drag and is_dragging:
        mouse.release(Button.left)

    if not keep_drag: is_dragging = False

    is_pinching = False


if __name__ == "__main__":
    main()

