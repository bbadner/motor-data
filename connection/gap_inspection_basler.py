"""
Gap Inspection – Basler / OpenCV fallback
PASS = no connector edge visible to the right of housing edge
FAIL = connector edge visible past housing edge
"""

import cv2
import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass

# =========================
# Config
# =========================
ROI_CONFIG_PATH = Path("roi_config.json")

EDGE_THRESH = 40
MIN_VERTICAL_EDGE_PX = 20
PRESENCE_WINDOW_W = 12
PRESENCE_WINDOW_OFFSET = 4
STABILITY_FRAMES = 4

# =========================
# ROI Model
# =========================
@dataclass
class InspectionConfig:
    roi_x: int = 1200
    roi_y: int = 700
    roi_w: int = 180
    roi_h: int = 260

# =========================
# ROI Load / Save (BACKWARD SAFE)
# =========================
def load_roi():
    if not ROI_CONFIG_PATH.exists():
        return InspectionConfig()

    with open(ROI_CONFIG_PATH, "r") as f:
        data = json.load(f)

    # 🔁 Backward compatibility
    if "x" in data:
        return InspectionConfig(
            roi_x=data["x"],
            roi_y=data["y"],
            roi_w=data["w"],
            roi_h=data["h"],
        )

    return InspectionConfig(**data)

def save_roi(cfg):
    with open(ROI_CONFIG_PATH, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

# =========================
# Camera Handling
# =========================
def open_camera():
    # Basler first
    try:
        from pypylon import pylon
        cam = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )
        cam.Open()
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print("Using Basler camera")
        return ("basler", cam)
    except Exception:
        pass

    # OpenCV fallback
    for idx in range(3):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Using OpenCV camera {idx}")
            return ("opencv", cap)

    raise RuntimeError("Camera not found")

def grab_frame(cam):
    kind, dev = cam
    if kind == "basler":
        grab = dev.RetrieveResult(2000)
        if grab.GrabSucceeded():
            img = grab.Array
            grab.Release()
            return img
        return None
    else:
        ret, frame = dev.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# =========================
# Vision Logic
# =========================
def detect_housing_edge(gray):
    sobel = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)
    return int(np.argmax(sobel.sum(axis=0)))

def connector_edge_present(gray, housing_x):
    x0 = housing_x + PRESENCE_WINDOW_OFFSET
    x1 = min(x0 + PRESENCE_WINDOW_W, gray.shape[1])

    window = gray[:, x0:x1]
    sobel = cv2.Sobel(window, cv2.CV_16S, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)

    return np.sum(sobel > EDGE_THRESH) > MIN_VERTICAL_EDGE_PX

# =========================
# Main Loop
# =========================
def main():
    cfg = load_roi()
    cam = open_camera()

    state = "PASS"
    stable = 0

    while True:
        frame = grab_frame(cam)
        if frame is None:
            continue

        roi = frame[
            cfg.roi_y:cfg.roi_y+cfg.roi_h,
            cfg.roi_x:cfg.roi_x+cfg.roi_w
        ]

        housing_x = detect_housing_edge(roi)
        fail = connector_edge_present(roi, housing_x)

        new_state = "FAIL" if fail else "PASS"
        if new_state == state:
            stable += 1
        else:
            stable = 0

        if stable >= STABILITY_FRAMES:
            state = new_state

        vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        cv2.line(vis, (housing_x, 0), (housing_x, vis.shape[0]), (255, 0, 0), 1)

        px0 = housing_x + PRESENCE_WINDOW_OFFSET
        cv2.rectangle(
            vis,
            (px0, 0),
            (px0 + PRESENCE_WINDOW_W, vis.shape[0]),
            (0, 255, 255),
            1
        )

        color = (0, 255, 0) if state == "PASS" else (0, 0, 255)
        cv2.putText(vis, state, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        cv2.imshow("Gap Inspection", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord("r"):
            cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h = cv2.selectROI(
                "Gap Inspection", frame, False
            )
        elif key == ord("s"):
            save_roi(cfg)
            print("ROI saved")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
