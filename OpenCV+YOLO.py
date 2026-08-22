# -*- coding: utf-8 -*-

import afb1
import cv2
import numpy as np
import threading
import time
from pathlib import Path
from ultralytics import YOLO

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
TURN_LEFT_SERVO = 15
TURN_RIGHT_SERVO = 165

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

FRAME_AREA = CAMERA_WIDTH * CAMERA_HEIGHT
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

YOLO_PATH = Path(__file__).resolve().parent / "best.pt"
YOLO_SIZE, YOLO_CONF = 320, 0.08
YOLO_INTERVAL = 0.10
YOLO_APPROACH_SPEED = 40

AREA_TRIGGER = {
    "circle": 0.0034,
    "person": 0.0011,
    "car1": 0.0060,
    "car2": 0.0010,
    "red light": 0.001,
    "left": 0.0008,
    "right": 0.0008,
}

CAR1_AREA_MAX = 0.0300

RELEVANT_LABELS = frozenset(AREA_TRIGGER)

MISSION_SPEED = 55
BETWEEN_CARS_SPEED = 55
STARTUP_SPEED = 100
STARTUP_SEC = 5.0
TURN_LEFT_SEC = 5.0
TURN_RIGHT_SEC = 5.0
CAR1_RIGHT_SEC = 2.0
CAR1_LEFT_SEC = 0.5
CAR2_LEFT_SEC = 2.0
CAR2_RIGHT_SEC = 0.5
MISSION_COOLDOWN = 3.0
PERSON_CLEAR_COUNT = 1
OBJECT_CLEAR_COUNT = 2


class YoloWorker:
    """YOLO runs independently; pending old frames are replaced by the newest."""

    def __init__(self, model):
        self.model = model
        self.names = {
            int(class_id): str(name).lower().strip().replace("_", " ")
            for class_id, name in model.names.items()
        }
        self.condition = threading.Condition()
        self.lock = threading.Lock()
        self.pending = None
        self.result = (0, {}, None)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.condition:
            self.pending = frame
            self.condition.notify()

    def snapshot(self):
        with self.lock:
            return self.result

    def _run(self):
        while True:
            with self.condition:
                while self.running and self.pending is None:
                    self.condition.wait()
                if not self.running:
                    return
                frame, self.pending = self.pending, None
            try:
                output = self.model(
                    frame, imgsz=YOLO_SIZE, conf=YOLO_CONF,
                    iou=0.30, max_det=8, agnostic_nms=True, verbose=False
                )[0]
                # Keep only the strongest box for each class. This removes
                # repeated/overlapping boxes that survived model NMS.
                detections = {}
                for box in output.boxes:
                    class_id = int(box.cls[0])
                    label = self.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                    confidence = float(box.conf[0])

                    item = {
                        "label": label,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                        "area_ratio": box_area / FRAME_AREA,
                    }

                    previous = detections.get(label)
                    if previous is None or confidence > previous["confidence"]:
                        detections[label] = item

                result = (time.monotonic(), detections, None)
            except Exception as exc:
                result = (time.monotonic(), {}, repr(exc))
            with self.lock:
                self.result = result

    def stop(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
        self.thread.join(timeout=2)


class Mission:
    def __init__(self):
        self.mode = "OPENCV"
        self.started = 0.0
        self.last_stamp = 0.0
        self.last_trigger = {}
        self.person_stop = False
        self.person_clear = 0
        self.red_stop = False
        self.traffic_signal_done = False
        self.active_label = None
        self.active_visible = False
        self.active_clear = 0
        self.suppressed_labels = set()
        self.between_car1_car2 = False

    def seen(self, detections, label):
        item = detections.get(label)
        if item is None:
            return False

        area_ratio = item["area_ratio"]

        if label == "car1":
            return AREA_TRIGGER[label] <= area_ratio <= CAR1_AREA_MAX

        return area_ratio >= AREA_TRIGGER[label]

    def start(self, mode, label, now):
        if now - self.last_trigger.get(mode, -999) < MISSION_COOLDOWN:
            return False
        self.mode, self.started = mode, now
        self.active_label, self.active_visible, self.active_clear = label, True, 0

        if label == "car1":
            self.between_car1_car2 = True
        elif label == "car2":
            self.between_car1_car2 = False

        self.last_trigger[mode] = now
        print(f"[MISSION] {mode}")
        return True

    def update(self, stamp, detections, now):
        if stamp <= self.last_stamp:
            return
        self.last_stamp = stamp

        for label in tuple(self.suppressed_labels):
            if label not in detections:
                self.suppressed_labels.discard(label)

        if self.seen(detections, "person"):
            self.person_stop, self.person_clear = True, 0
        elif self.person_stop:
            self.person_clear += 1
            if self.person_clear >= PERSON_CLEAR_COUNT:
                self.person_stop = False

        if self.active_label is not None:
            if self.seen(detections, self.active_label):
                self.active_visible, self.active_clear = True, 0
            else:
                self.active_clear += 1
                if self.active_clear >= OBJECT_CLEAR_COUNT:
                    self.active_visible = False

        if not self.traffic_signal_done and self.seen(detections, "red light"):
            self.red_stop = True

        # Direction arrows are acted on only after stopping at the red light.
        # Once a left/right signal is executed, red light is ignored thereafter.
        if self.red_stop:
            if self.seen(detections, "left"):
                self.red_stop = False
                self.traffic_signal_done = True
                self.start("TURN_LEFT", "left", now)
            elif self.seen(detections, "right"):
                self.red_stop = False
                self.traffic_signal_done = True
                self.start("TURN_RIGHT", "right", now)
            return

        if self.mode != "OPENCV":
            return
        for label, mode in (
            ("car1", "LANE_RIGHT"), ("car2", "LANE_LEFT"),
            ("circle", "TURN_RIGHT"),
        ):
            if label in self.suppressed_labels:
                continue
            if self.seen(detections, label) and self.start(mode, label, now):
                return

    def opencv_allowed(self):
        return not self.person_stop and not self.red_stop and self.mode == "OPENCV"

    def yolo_box_blocks_opencv(self, detections):
        for label in RELEVANT_LABELS:
            if label in self.suppressed_labels:
                continue
            if label == "red light" and self.traffic_signal_done:
                continue
            if self.seen(detections, label):
                return True
        return False

    def finish_if_clear(self):
        if not self.active_visible:
            self.mode = "OPENCV"
            self.active_label = None
            self.active_clear = 0
            return True
        return False

    def command(self, road_speed, road_servo, now):
        if self.person_stop:
            return 0, SERVO_CENTER, "STOP_PERSON"
        if self.red_stop:
            return 0, SERVO_CENTER, "STOP_RED_LIGHT"

        elapsed = now - self.started
        if self.mode in ("TURN_LEFT", "TURN_RIGHT"):
            turn_duration = (
                TURN_LEFT_SEC
                if self.mode == "TURN_LEFT"
                else TURN_RIGHT_SEC
            )

            if elapsed < turn_duration:
                angle = TURN_LEFT_SERVO if self.mode == "TURN_LEFT" else TURN_RIGHT_SERVO
                return MISSION_SPEED, angle, self.mode

            if not self.finish_if_clear():
                return MISSION_SPEED, SERVO_CENTER, self.mode

        if self.mode == "LANE_RIGHT":
            if elapsed < CAR1_RIGHT_SEC:
                return MISSION_SPEED, 145, self.mode

            if elapsed < CAR1_RIGHT_SEC + CAR1_LEFT_SEC:
                return MISSION_SPEED, SERVO_LEFT, self.mode

            self.mode = "OPENCV"
            self.active_label = None
            self.active_visible = False
            self.active_clear = 0
            self.suppressed_labels.add("car1")

            return MOTOR_SPEED, SERVO_CENTER, "OPENCV"

        if self.mode == "LANE_LEFT":
            if elapsed < CAR2_LEFT_SEC:
                return MISSION_SPEED, 35, self.mode
            if elapsed < CAR2_LEFT_SEC + CAR2_RIGHT_SEC:
                return MISSION_SPEED, 91, self.mode
            if not self.finish_if_clear():
                return MISSION_SPEED, SERVO_CENTER, self.mode

        return road_speed, road_servo, "OPENCV"



def has_relevant_yolo_box(detections):
    return bool(RELEVANT_LABELS.intersection(detections))

def camera_to_bgr(frame):
    if frame is None:
        return None
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main():
    global frame_count
    if not YOLO_PATH.is_file():
        raise FileNotFoundError(f"YOLO model not found: {YOLO_PATH}")

    yolo = YOLO(str(YOLO_PATH))
    print("YOLO classes:", yolo.names)
    required = set(AREA_TRIGGER)
    actual = {str(v).lower().strip().replace("_", " ")
              for v in yolo.names.values()}
    if missing := required - actual:
        print("[WARNING] Model class-name mismatch:", sorted(missing))

    afb1.camera.init(CAMERA_WIDTH, CAMERA_HEIGHT, FPS)
    afb1.gpio.init()
    worker = YoloWorker(yolo)
    mission = Mission()
    startup_started = time.monotonic()
    last_submit = 0.0
    detections = {}
    road_speed, road_servo, road_status = MOTOR_SPEED, SERVO_CENTER, "GO"

    try:
        afb1.gpio.servo(SERVO_CENTER)
        while True:
            frame_bgr = camera_to_bgr(afb1.camera.get_image())
            if frame_bgr is None:
                afb1.gpio.motor(0)
                continue
            now = time.monotonic()

            if now - startup_started < STARTUP_SEC:
                afb1.gpio.servo(SERVO_CENTER)
                afb1.gpio.motor(STARTUP_SPEED)

                if frame_count % LOG_INTERVAL == 0:
                    print(
                        f"[{frame_count}] STARTUP "
                        f"servo={SERVO_CENTER} speed={STARTUP_SPEED}"
                    )

                frame_count += 1
                continue

            if now - last_submit >= YOLO_INTERVAL:
                worker.submit(frame_bgr)
                last_submit = now
            stamp, latest, error = worker.snapshot()
            if error:
                afb1.gpio.motor(0)
                raise RuntimeError(f"YOLO error: {error}")
            if stamp and now - stamp < 0.6:
                detections = latest
                mission.update(stamp, detections, now)
            elif stamp and now - stamp >= 0.6:
                detections = {}

            yolo_box_active = mission.yolo_box_blocks_opencv(detections)
            opencv_enabled = mission.opencv_allowed() and not yolo_box_active

            if opencv_enabled:
                lane_center, left_x, right_x, _ = detect_lane(frame_bgr)
                road_servo = calculate_servo(lane_center)

                if lane_center is None:
                    road_speed, road_status = SLOW_SPEED, "LANE LOST"
                else:
                    error_x = lane_center - CAMERA_WIDTH // 2

                    if turn_hold_frames > 0 or abs(error_x) > 120:
                        road_speed = SHARP_TURN_SPEED
                    elif abs(error_x) > 70:
                        road_speed = CURVE_SPEED
                    elif abs(error_x) > 40:
                        road_speed = 58
                    else:
                        road_speed = MOTOR_SPEED

                    road_status = (
                        "LEFT" if error_x < -45
                        else "RIGHT" if error_x > 45
                        else "GO"
                    )

            elif yolo_box_active and mission.mode == "OPENCV":
                road_speed = min(road_speed, YOLO_APPROACH_SPEED)
                road_status = "YOLO DETECT"

            if mission.between_car1_car2 and mission.mode == "OPENCV":
                road_speed = min(road_speed, BETWEEN_CARS_SPEED)

            speed, servo, mode = mission.command(
                road_speed,
                road_servo,
                now
            )
            afb1.gpio.servo(servo)
            afb1.gpio.motor(speed)

            if frame_count % LOG_INTERVAL == 0:
                print(f"[{frame_count}] {mode} {road_status} "
                      f"servo={servo} speed={speed}")
            frame_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        afb1.gpio.stop_all()
        try:
            afb1.camera.release_camera()
        except (AttributeError, TypeError):
            pass


if __name__ == "__main__":
    main()