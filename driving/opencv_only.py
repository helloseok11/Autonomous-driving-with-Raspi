# -*- coding: utf-8 -*-

import afb1
import cv2
import numpy as np
import time

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FPS = 15

MOTOR_SPEED = 65
CURVE_SPEED = 52
SHARP_TURN_SPEED = 42
SLOW_SPEED = 38

SERVO_LEFT = 30
SERVO_CENTER = 90
SERVO_RIGHT = 141

STEERING_GAIN = 0.55
MAX_SERVO_CHANGE = 16

SMOOTHING = 0.55

MAX_CENTER_STEP = 60

DEFAULT_LANE_WIDTH = 260

LOST_LANE_LIMIT = 25

WHITE_S_MAX = 65
WHITE_V_MIN = 165

LOCAL_CONTRAST_MIN = 27
LOCAL_BLUR_SIZE = 21


MIN_ABS_LANE_SLOPE = 0.35

KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_CLOSE = np.ones((7, 7), np.uint8)

ROI_Y_START = int(CAMERA_HEIGHT * 0.42)
ROI_HEIGHT = CAMERA_HEIGHT - ROI_Y_START
ROI_Y_BOTTOM = ROI_HEIGHT - 1
ROI_Y_TOP = int(ROI_HEIGHT * 0.10)
ROI_Y_TARGET = int(ROI_HEIGHT * 0.24)

WHITE_LOWER = np.array([0, 0, WHITE_V_MIN], dtype=np.uint8)
WHITE_UPPER = np.array([180, WHITE_S_MAX, 255], dtype=np.uint8)

ROAD_MASK = np.zeros((ROI_HEIGHT, CAMERA_WIDTH), dtype=np.uint8)
ROAD_POLYGON = np.array([[
    (0, ROI_Y_BOTTOM),
    (CAMERA_WIDTH - 1, ROI_Y_BOTTOM),
    (int(CAMERA_WIDTH * 0.82), ROI_Y_TOP),
    (int(CAMERA_WIDTH * 0.18), ROI_Y_TOP)
]], dtype=np.int32)
cv2.fillPoly(ROAD_MASK, ROAD_POLYGON, 255)

LOG_INTERVAL = 15

current_servo = SERVO_CENTER

smoothed_lane_center = None
last_valid_lane_center = None

lane_width = DEFAULT_LANE_WIDTH
lost_lane_count = 0

frame_count = 0

turn_hold_direction = 0
turn_hold_frames = 0
straight_confirm_count = 0

TURN_START_ERROR = 65
TURN_HOLD_TIME = 28
TURN_HOLD_MIN_ANGLE = 30
STRAIGHT_CONFIRM_FRAMES = 6

def get_x_at_y(line, target_y):
    """Return the x-coordinate of a Hough line at target_y."""
    x1, y1, x2, y2 = line

    if x2 == x1:
        return None

    slope = (y2 - y1) / (x2 - x1)

    if abs(slope) < MIN_ABS_LANE_SLOPE:
        return None

    intercept = y1 - slope * x1

    x = (target_y - intercept) / slope

    return x

def detect_lane(frame_bgr):
    global smoothed_lane_center
    global last_valid_lane_center
    global lane_width

    roi = frame_bgr[ROI_Y_START:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    color_white_mask = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)

    local_bg = cv2.GaussianBlur(
        gray,
        (LOCAL_BLUR_SIZE, LOCAL_BLUR_SIZE),
        0
    )
    local_contrast = cv2.subtract(gray, local_bg)
    contrast_mask = cv2.inRange(
        local_contrast,
        LOCAL_CONTRAST_MIN,
        255
    )

    white_mask = cv2.bitwise_and(
        color_white_mask,
        contrast_mask
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_OPEN,
        KERNEL_OPEN
    )

    white_mask = cv2.morphologyEx(
        white_mask,
        cv2.MORPH_CLOSE,
        KERNEL_CLOSE
    )

    lane_mask = cv2.bitwise_and(white_mask, ROAD_MASK)

    edges = cv2.Canny(lane_mask, 30, 100)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=12,
        minLineLength=15,
        maxLineGap=80
    )

    left_x_values = []
    right_x_values = []

    if lines is not None:
        for line in lines:
            values = np.asarray(line).reshape(-1)

            if len(values) != 4:
                continue

            x1, y1, x2, y2 = values

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < MIN_ABS_LANE_SLOPE:
                continue

            x_target = get_x_at_y(
                (x1, y1, x2, y2),
                ROI_Y_TARGET
            )

            if x_target is None:
                continue

            if slope < 0 and x_target < CAMERA_WIDTH * 0.75:
                left_x_values.append(x_target)

            elif slope > 0 and x_target > CAMERA_WIDTH * 0.25:
                right_x_values.append(x_target)

    left_x = None
    right_x = None

    if left_x_values:
        left_x = int(np.median(left_x_values))

    if right_x_values:
        right_x = int(np.median(right_x_values))

    if left_x is not None and right_x is not None:
        measured_width = right_x - left_x

        if 100 < measured_width < 500:
            lane_width = int(
                lane_width * 0.80
                + measured_width * 0.20
            )
        else:
            right_x = left_x + lane_width

    elif left_x is not None:
        right_x = left_x + lane_width

    elif right_x is not None:
        left_x = right_x - lane_width

    lane_center = None

    if left_x is not None and right_x is not None:
        raw_center = (left_x + right_x) / 2

        if 0 < raw_center < CAMERA_WIDTH:
            if last_valid_lane_center is None:
                filtered_center = raw_center
            else:
                center_change = raw_center - last_valid_lane_center

                center_change = np.clip(
                    center_change,
                    -MAX_CENTER_STEP,
                    MAX_CENTER_STEP
                )

                filtered_center = (
                    last_valid_lane_center + center_change
                )

            if smoothed_lane_center is None:
                smoothed_lane_center = filtered_center
            else:
                smoothed_lane_center = (
                    smoothed_lane_center * (1 - SMOOTHING)
                    + filtered_center * SMOOTHING
                )

            last_valid_lane_center = smoothed_lane_center
            lane_center = int(smoothed_lane_center)

    return lane_center, left_x, right_x, None

def calculate_servo(lane_center):
    global current_servo
    global lost_lane_count
    global turn_hold_direction
    global turn_hold_frames
    global straight_confirm_count

    if lane_center is not None:
        lost_lane_count = 0

        camera_center = CAMERA_WIDTH / 2
        error = lane_center - camera_center

        if abs(error) > TURN_START_ERROR:
            turn_hold_direction = 1 if error > 0 else -1
            turn_hold_frames = TURN_HOLD_TIME
            straight_confirm_count = 0

        if turn_hold_frames > 0:
            if abs(error) < 30:
                straight_confirm_count += 1
            else:
                straight_confirm_count = 0

            if straight_confirm_count >= STRAIGHT_CONFIRM_FRAMES:
                turn_hold_frames = 0
                turn_hold_direction = 0
                straight_confirm_count = 0
            else:
                turn_hold_frames -= 1

        target_servo = SERVO_CENTER + error * STEERING_GAIN

    else:
        lost_lane_count += 1
        straight_confirm_count = 0

        if turn_hold_frames > 0 and turn_hold_direction != 0:
            turn_hold_frames -= 1

            if turn_hold_direction < 0:
                target_servo = SERVO_LEFT
            else:
                target_servo = SERVO_RIGHT

        elif lost_lane_count <= LOST_LANE_LIMIT:
            target_servo = current_servo

        else:
            target_servo = SERVO_CENTER

    if turn_hold_frames > 0 and turn_hold_direction != 0:
        if turn_hold_direction < 0:

            target_servo = min(
                target_servo,
                SERVO_CENTER - TURN_HOLD_MIN_ANGLE
            )
        else:

            target_servo = max(
                target_servo,
                SERVO_CENTER + TURN_HOLD_MIN_ANGLE
            )

    target_servo = int(np.clip(
        target_servo,
        SERVO_LEFT,
        SERVO_RIGHT
    ))

    servo_change = target_servo - current_servo

    servo_change = int(np.clip(
        servo_change,
        -MAX_SERVO_CHANGE,
        MAX_SERVO_CHANGE
    ))

    current_servo += servo_change

    return int(np.clip(
        current_servo,
        SERVO_LEFT,
        SERVO_RIGHT
    ))

STARTUP_SPEED = 100
STARTUP_SEC = 5.0


def camera_to_bgr(frame):
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main():
    global frame_count

    afb1.camera.init(CAMERA_WIDTH, CAMERA_HEIGHT, FPS)
    afb1.gpio.init()
    startup_started = time.monotonic()

    try:
        afb1.gpio.servo(SERVO_CENTER)

        while True:
            frame_bgr = camera_to_bgr(afb1.camera.get_image())
            if frame_bgr is None:
                afb1.gpio.motor(0)
                continue

            now = time.monotonic()

            if now - startup_started < STARTUP_SEC:
                speed = STARTUP_SPEED
                servo = SERVO_CENTER
                status = "STARTUP"
            else:
                lane_center, _, _, _ = detect_lane(frame_bgr)
                servo = calculate_servo(lane_center)

                if lane_center is None:
                    speed = SLOW_SPEED
                    status = "LANE LOST"
                else:
                    error_x = lane_center - CAMERA_WIDTH // 2

                    if turn_hold_frames > 0 or abs(error_x) > 120:
                        speed = SHARP_TURN_SPEED
                    elif abs(error_x) > 70:
                        speed = CURVE_SPEED
                    elif abs(error_x) > 40:
                        speed = 58
                    else:
                        speed = MOTOR_SPEED

                    status = (
                        "LEFT" if error_x < -45
                        else "RIGHT" if error_x > 45
                        else "GO"
                    )

            afb1.gpio.servo(servo)
            afb1.gpio.motor(speed)

            if frame_count % LOG_INTERVAL == 0:
                print(
                    f"[{frame_count}] {status} "
                    f"servo={servo} speed={speed}"
                )
            frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        afb1.gpio.stop_all()
        try:
            afb1.camera.release_camera()
        except (AttributeError, TypeError):
            pass


if __name__ == "__main__":
    main()
