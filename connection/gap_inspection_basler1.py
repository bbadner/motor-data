import json
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from pypylon import pylon
except Exception as exc:
    raise SystemExit(
        "pypylon is required for Basler cameras.\n"
        "Install with: pip install pypylon\n"
        f"Import error: {exc}"
    )

# ==========================
# CONFIG FILE
# ==========================
ROI_CONFIG_PATH = Path("roi_config.json")

@dataclass
class ROIConfig:
    x: int = 1200
    y: int = 717
    w: int = 184
    h: int = 740

def load_roi() -> ROIConfig:
    if ROI_CONFIG_PATH.exists():
        try:
            data = json.loads(ROI_CONFIG_PATH.read_text())
            # Accept either {"x":..,"y":..,"w":..,"h":..} or legacy dicts with those keys
            return ROIConfig(
                x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"])
            )
        except Exception:
            pass
    return ROIConfig()

def save_roi(cfg: ROIConfig) -> None:
    ROI_CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2))

# ==========================
# INSPECTION TUNABLES
# ==========================
# Your decision rule:
GAP_GOOD_MAX_PX = 5  # PASS if gap < 5, FAIL if gap >= 5

# Vertical band inside ROI (fraction of ROI height) to ignore background above/below connector
BAND_Y0_FRAC = 0.45
BAND_Y1_FRAC = 0.85

# Search width to the right of housing edge for connector pixels
CONNECTOR_SEARCH_W = 80

# Preprocessing
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
MORPH_KERNEL = 3  # small cleanup

# Stability (prevents bouncing)
HISTORY_N = 7
REQUIRED_TO_FLIP = 5  # require 5 of last 7 frames to flip state

# Display
ZOOM_SCALE = 3  # zoomed ROI window
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ==========================
# BASLER CAMERA
# ==========================
def open_basler_camera() -> pylon.InstantCamera:
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if not devices:
        raise RuntimeError(
            "No Basler camera found by pypylon.\n"
            "Check: camera powered, USB3 cable, drivers installed, and no other app is using it."
        )

    cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    cam.Open()

    # Try to make streaming stable (safe defaults)
    try:
        cam.PixelFormat.SetValue("Mono8")
    except Exception:
        pass

    try:
        cam.AcquisitionMode.SetValue("Continuous")
    except Exception:
        pass

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return cam

def grab_gray(cam: pylon.InstantCamera) -> Optional[np.ndarray]:
    if not cam.IsGrabbing():
        return None
    grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_Return)
    if not grab.GrabSucceeded():
        grab.Release()
        return None

    img = grab.Array  # already Mono8 if PixelFormat set, otherwise could be Bayer/etc.
    grab.Release()

    if img is None:
        return None
    if img.ndim == 2:
        return img
    # Fallback if camera gives color
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ==========================
# CORE GAP MEASUREMENT
# ==========================
def measure_gap_px(gray: np.ndarray, roi: ROIConfig) -> Tuple[Optional[int], Optional[int], Optional[int], np.ndarray, Tuple[int,int,int,int]]:
    """
    Returns:
      housing_edge_x (ROI-relative),
      connector_edge_x (ROI-relative),
      gap_px,
      roi_gray (cropped),
      band_rect (x0,y0,w,h) in ROI-relative coords
    """
    H, W = gray.shape[:2]
    x, y, w, h = roi.x, roi.y, roi.w, roi.h
    if x < 0 or y < 0 or x + w > W or y + h > H or w <= 2 or h <= 2:
        return None, None, None, np.zeros((1,1), np.uint8), (0,0,0,0)

    roi_gray = gray[y:y+h, x:x+w].copy()

    # Contrast normalize (makes thresholding more stable)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    roi_eq = clahe.apply(roi_gray)

    # Otsu threshold to separate bright plastic/connector from darker background
    _, bw = cv2.threshold(roi_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small cleanup
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL, MORPH_KERNEL))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)

    # Vertical band
    by0 = int(h * BAND_Y0_FRAC)
    by1 = int(h * BAND_Y1_FRAC)
    by0 = max(0, min(h-1, by0))
    by1 = max(by0+1, min(h, by1))
    band = bw[by0:by1, :]

    # For each row in band, find rightmost bright pixel = housing edge candidate
    # But only accept rows where there is a decent amount of bright pixels on left side.
    housing_candidates = []
    for r in range(band.shape[0]):
        row = band[r]
        white = np.where(row > 0)[0]
        if white.size < 10:
            continue
        housing_candidates.append(int(white.max()))
    if len(housing_candidates) < 10:
        return None, None, None, roi_gray, (0, by0, w, by1-by0)

    housing_edge = int(np.median(housing_candidates))

    # Search to the right for the connector bright pixels (leftmost bright pixel beyond housing edge)
    search_x0 = min(w-1, housing_edge + 1)
    search_x1 = min(w, search_x0 + CONNECTOR_SEARCH_W)
    if search_x1 <= search_x0 + 1:
        return housing_edge, None, None, roi_gray, (0, by0, w, by1-by0)

    band_search = band[:, search_x0:search_x1]

    connector_candidates = []
    for r in range(band_search.shape[0]):
        row = band_search[r]
        white = np.where(row > 0)[0]
        if white.size == 0:
            continue
        connector_candidates.append(int(white.min()) + search_x0)

    if len(connector_candidates) < 5:
        # No connector pixels found (could be missing connector or too dark)
        return housing_edge, None, None, roi_gray, (0, by0, w, by1-by0)

    connector_edge = int(np.median(connector_candidates))

    gap_px = connector_edge - housing_edge - 1
    if gap_px < 0:
        gap_px = 0

    return housing_edge, connector_edge, gap_px, roi_gray, (0, by0, w, by1-by0)

# ==========================
# UI / MAIN LOOP
# ==========================
def main():
    roi = load_roi()

    cam = open_basler_camera()

    history = deque(maxlen=HISTORY_N)
    stable = "PASS"  # start optimistic

    print("Controls: R = reset ROI (not implemented drag in this version), S = save ROI, ESC = quit")

    while True:
        gray = grab_gray(cam)
        if gray is None:
            continue

        # Measure
        hx, cx, gap, roi_gray, band_rect = measure_gap_px(gray, roi)

        # Decision
        if gap is None:
            instant = "FAIL"  # if we cannot measure reliably, treat as fail
        else:
            instant = "PASS" if gap < GAP_GOOD_MAX_PX else "FAIL"

        history.append(instant)

        # Hysteresis / stability
        if history.count("FAIL") >= REQUIRED_TO_FLIP:
            stable = "FAIL"
        elif history.count("PASS") >= REQUIRED_TO_FLIP:
            stable = "PASS"

        # Display frame (convert gray to BGR for colored overlays)
        frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # ROI rectangle
        cv2.rectangle(frame_bgr, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), (0, 255, 255), 2)

        # Band rectangle inside ROI
        bx, by, bw, bh = band_rect
        cv2.rectangle(frame_bgr,
                      (roi.x + bx, roi.y + by),
                      (roi.x + bx + bw, roi.y + by + bh),
                      (0, 255, 255), 1)

        # Housing edge line (blue)
        if hx is not None:
            xh = roi.x + hx
            cv2.line(frame_bgr, (xh, roi.y), (xh, roi.y + roi.h), (255, 0, 0), 1)

        # Connector edge line (red)
        if cx is not None:
            xc = roi.x + cx
            cv2.line(frame_bgr, (xc, roi.y), (xc, roi.y + roi.h), (0, 0, 255), 1)

        # Text
        color = (0, 255, 0) if stable == "PASS" else (0, 0, 255)
        cv2.putText(frame_bgr, stable, (30, 60), FONT, 2.2, color, 4)

        if gap is None:
            cv2.putText(frame_bgr, "Gap: N/A", (30, 110), FONT, 1.2, (255, 255, 255), 2)
        else:
            cv2.putText(frame_bgr, f"Gap: {gap}px", (30, 110), FONT, 1.2, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f"Rule: PASS if < {GAP_GOOD_MAX_PX}px", (30, 150), FONT, 0.9, (200, 200, 200), 2)

        cv2.imshow("Gap Inspection (Basler)", frame_bgr)

        # Zoomed ROI window so you can actually see the important area
        roi_view = frame_bgr[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
        roi_zoom = cv2.resize(roi_view, (roi_view.shape[1]*ZOOM_SCALE, roi_view.shape[0]*ZOOM_SCALE), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("ROI Zoom", roi_zoom)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            save_roi(roi)
            print("ROI saved:", asdict(roi))
        # (Optional) You can add ROI drag editing back in later.

    cam.StopGrabbing()
    cam.Close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
cv2
import numpy as np

# =========================
# CONFIGURATION
# =========================
CAMERA_INDEX = 0        # USB camera index (Basler via pypylon not required here)
ROI_X = 420
ROI_Y = 180
ROI_W = 180
ROI_H = 260

DARK_THRESH = 60        # Pixel intensity considered "dark"
MIN_EDGE_STRENGTH = 12 # Edge strength to qualify as housing edge
GAP_LIMIT_PX = 5        # PASS < 5, FAIL >= 5

# =========================
# GAP MEASUREMENT LOGIC
# =========================
def measure_gap(gray_roi):
    edges = cv2.Canny(gray_roi, 60, 160)

    # Sum edges vertically → column strength
    col_strength = np.sum(edges > 0, axis=0)

    housing_x = None
    for x in range(len(col_strength)):
        if col_strength[x] > MIN_EDGE_STRENGTH:
            housing_x = x  # keep rightmost

    if housing_x is None or housing_x >= gray_roi.shape[1] - 2:
        return None, None

    # Scan right of housing edge
    scan = gray_roi[:, housing_x + 1:]
    mean_cols = np.mean(scan, axis=0)

    gap_px = 0
    for val in mean_cols:
        if val < DARK_THRESH:
            gap_px += 1
        else:
            break

    return gap_px, housing_x

# =========================
# MAIN
# =========================
def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]
        gap_px, hx = measure_gap(roi)

        status = "NO EDGE"
        color = (0, 255, 255)

        if gap_px is not None:
            if gap_px >= GAP_LIMIT_PX:
                status = "FAIL"
                color = (0, 0, 255)
            else:
                status = "PASS"
                color = (0, 255, 0)

        # =========================
        # OVERLAYS
        # =========================
        vis = frame.copy()

        # ROI
        cv2.rectangle(
            vis,
            (ROI_X, ROI_Y),
            (ROI_X + ROI_W, ROI_Y + ROI_H),
            (0, 255, 255),
            1
        )

        if hx is not None:
            # Housing edge (BLUE)
            cv2.line(
                vis,
                (ROI_X + hx, ROI_Y),
                (ROI_X + hx, ROI_Y + ROI_H),
                (255, 0, 0),
                1
            )

            # Gap region (RED)
            cv2.rectangle(
                vis,
                (ROI_X + hx + 1, ROI_Y),
                (ROI_X + hx + 1 + gap_px, ROI_Y + ROI_H),
                (0, 0, 255),
                2
            )

        # Text
        cv2.putText(
            vis,
            status,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            color,
            3
        )

        if gap_px is not None:
            cv2.putText(
                vis,
                f"Gap: {gap_px}px",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2
            )

        cv2.imshow("Gap Inspection", vis)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

