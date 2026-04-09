import cv2
import numpy as np
from collections import deque
from pathlib import Path
import json

# ==========================
# CONFIG
# ==========================
ROI_CONFIG_PATH = Path("roi_config.json")

PRESENCE_WINDOW_WIDTH = 16      # narrow window immediately right of housing
DARK_PIXEL_THRESHOLD = 90       # grayscale value considered "dark"
DARK_RATIO_FAIL = 0.10          # 10% dark pixels = FAIL

STABILITY_WINDOW = 7            # frames
STABILITY_REQUIRED = 5          # required to flip state

# ==========================
# STATE
# ==========================
decision_history = deque(maxlen=STABILITY_WINDOW)
stable_result = "PASS"

# ==========================
# ROI LOAD / SAVE
# ==========================
def load_roi():
    if ROI_CONFIG_PATH.exists():
        with open(ROI_CONFIG_PATH, "r") as f:
            return json.load(f)
    return {"x": 1200, "y": 717, "w": 184, "h": 740}

def save_roi(roi):
    with open(ROI_CONFIG_PATH, "w") as f:
        json.dump(roi, f, indent=2)

# ==========================
# CORE INSPECTION
# ==========================
def inspect_presence(gray, roi):
    x, y, w, h = roi.values()
    roi_img = gray[y:y+h, x:x+w]

    if roi_img.size == 0:
        return "FAIL", None, None, 0.0

    # --- Smooth to remove texture noise
    roi_blur = cv2.GaussianBlur(roi_img, (5, 5), 0)

    # --- Find rightmost bright housing face
    col_mean = roi_blur.mean(axis=0)

    housing_edge_x = None
    for i in range(len(col_mean) - 1, -1, -1):
        if col_mean[i] > 140:   # housing is bright
            housing_edge_x = i
            break

    if housing_edge_x is None:
        return "FAIL", None, None, 0.0

    # --- Presence window (immediately right)
    pw_x = housing_edge_x + 1
    pw_w = min(PRESENCE_WINDOW_WIDTH, w - pw_x)

    if pw_w <= 0:
        return "PASS", housing_edge_x, None, 0.0

    presence_window = roi_blur[:, pw_x:pw_x + pw_w]

    # --- Dark pixel ratio
    dark_pixels = presence_window < DARK_PIXEL_THRESHOLD
    dark_ratio = dark_pixels.sum() / dark_pixels.size

    instant_result = "FAIL" if dark_ratio >= DARK_RATIO_FAIL else "PASS"

    return instant_result, housing_edge_x, (pw_x, 0, pw_w, h), dark_ratio

# ==========================
# CAMERA OPEN (ROBUST)
# ==========================
def open_camera():
    for backend in [cv2.CAP_DSHOW, cv2.CAP_ANY]:
        for idx in range(6):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Using camera index {idx}")
                return cap
    raise RuntimeError("Camera not found")

# ==========================
# MAIN
# ==========================
def main():
    global stable_result

    roi = load_roi()
    cap = open_camera()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        instant, housing_x, pw, dark_ratio = inspect_presence(gray, roi)
        decision_history.append(instant)

        if decision_history.count("FAIL") >= STABILITY_REQUIRED:
            stable_result = "FAIL"
        elif decision_history.count("PASS") >= STABILITY_REQUIRED:
            stable_result = "PASS"

        # ==========================
        # DRAW OVERLAY
        # ==========================
        display = frame.copy()
        x, y, w, h = roi.values()

        # ROI
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Housing edge
        if housing_x is not None:
            hx = x + housing_x
            cv2.line(display, (hx, y), (hx, y+h), (255, 0, 0), 1)

        # Presence window
        if pw is not None:
            pw_x, _, pw_w, _ = pw
            cv2.rectangle(
                display,
                (x + pw_x, y),
                (x + pw_x + pw_w, y + h),
                (0, 255, 255),
                1
            )

        # Result text
        color = (0, 255, 0) if stable_result == "PASS" else (0, 0, 255)
        cv2.putText(display, stable_result, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

        cv2.putText(display, f"Dark ratio: {dark_ratio:.2f}",
                    (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Gap Inspection (R edit ROI | S save ROI | ESC quit)", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('s'):
            save_roi(roi)
            print("ROI saved")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



