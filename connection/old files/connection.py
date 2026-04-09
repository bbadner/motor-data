import cv2
import numpy as np
import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


# =========================
# Config
# =========================

CONFIG_PATH = "connection_config.json"

@dataclass
class Config:
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x,y,w,h
    good_max_gap_px: int = 18                       # tune after you see gap values
    min_edge_strength: int = 40                     # vertical edge threshold (Sobel)
    search_max_px: int = 28                         # max px to search for connector edge after controller edge
    stable_frames_required: int = 3                 # debounce
    presence_delta_thresh: float = 8.0              # part present/absent threshold
    presence_frames_required: int = 3               # debounce presence

def load_config(path: str) -> Config:
    if not os.path.exists(path):
        return Config()
    with open(path, "r") as f:
        d = json.load(f)
    cfg = Config()
    cfg.roi = tuple(d.get("roi", cfg.roi))
    cfg.good_max_gap_px = int(d.get("good_max_gap_px", cfg.good_max_gap_px))
    cfg.min_edge_strength = int(d.get("min_edge_strength", cfg.min_edge_strength))
    cfg.search_max_px = int(d.get("search_max_px", cfg.search_max_px))
    cfg.stable_frames_required = int(d.get("stable_frames_required", cfg.stable_frames_required))
    cfg.presence_delta_thresh = float(d.get("presence_delta_thresh", cfg.presence_delta_thresh))
    cfg.presence_frames_required = int(d.get("presence_frames_required", cfg.presence_frames_required))
    return cfg

def save_config(cfg: Config, path: str) -> None:
    d = {
        "roi": list(cfg.roi),
        "good_max_gap_px": cfg.good_max_gap_px,
        "min_edge_strength": cfg.min_edge_strength,
        "search_max_px": cfg.search_max_px,
        "stable_frames_required": cfg.stable_frames_required,
        "presence_delta_thresh": cfg.presence_delta_thresh,
        "presence_frames_required": cfg.presence_frames_required,
    }
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"[INFO] Saved config to {path}")


# =========================
# Output (stub)
# =========================

class OutputControl:
    """Replace these with your PLC/relay control later."""
    def __init__(self):
        self.state = "OFF"

    def green(self):
        if self.state != "GREEN":
            print("[OUTPUT] GREEN")
            self.state = "GREEN"

    def red_latched(self):
        if self.state != "RED":
            print("[OUTPUT] RED (LATCHED)")
            self.state = "RED"

    def reset(self):
        if self.state != "OFF":
            print("[OUTPUT] RESET/OFF")
            self.state = "OFF"


# =========================
# Presence detection
# =========================

class PresenceDetector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.empty_baseline: Optional[float] = None
        self.present = False
        self.present_count = 0
        self.absent_count = 0

    def update(self, gray_roi: np.ndarray) -> bool:
        mean_val = float(np.mean(gray_roi))
        if self.empty_baseline is None:
            self.empty_baseline = mean_val

        delta = abs(mean_val - self.empty_baseline)
        is_present_now = delta >= self.cfg.presence_delta_thresh

        if is_present_now:
            self.present_count += 1
            self.absent_count = 0
        else:
            self.absent_count += 1
            self.present_count = 0

        if not self.present and self.present_count >= self.cfg.presence_frames_required:
            self.present = True
        if self.present and self.absent_count >= self.cfg.presence_frames_required:
            self.present = False

        return self.present

    def rebalance_empty(self, gray_roi: np.ndarray):
        self.empty_baseline = float(np.mean(gray_roi))
        self.present = False
        self.present_count = 0
        self.absent_count = 0
        print("[INFO] Empty baseline re-learned.")


# =========================
# Gap measurement (edge-to-edge)
# =========================

def measure_gap_px(gray_roi: np.ndarray, cfg: Config) -> Optional[int]:
    """
    Measures gap between:
      - first strong vertical edge from left (controller right face)
      - next strong vertical edge within search window (connector face)
    """
    # vertical edges only
    sobel = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel = np.abs(sobel).astype(np.uint8)

    h, w = sobel.shape
    white_edges = []
    conn_edges = []

    y0 = int(h * 0.35)
    y1 = int(h * 0.65)

    for y in range(y0, y1):
        row = sobel[y, :]

        left_candidates = np.where(row > cfg.min_edge_strength)[0]
        if left_candidates.size == 0:
            continue

        white_edge = int(left_candidates[0])

        search_start = white_edge + 3
        search_end = min(white_edge + cfg.search_max_px, w)
        if search_start >= search_end:
            continue

        search_region = row[search_start:search_end]
        right_candidates = np.where(search_region > cfg.min_edge_strength)[0]
        if right_candidates.size == 0:
            continue

        conn_edge = int(search_start + right_candidates[0])

        white_edges.append(white_edge)
        conn_edges.append(conn_edge)

    if len(white_edges) < 8:
        return None

    white_edge = int(np.median(white_edges))
    conn_edge = int(np.median(conn_edges))
    return int(conn_edge - white_edge)


# =========================
# ROI selection (calibration)
# =========================

_roi_start = None
_roi_current = None
_roi_done = False

def _roi_mouse(event, x, y, flags, param):
    global _roi_start, _roi_current, _roi_done
    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_start = (x, y)
        _roi_current = (x, y)
        _roi_done = False
    elif event == cv2.EVENT_MOUSEMOVE and _roi_start is not None:
        _roi_current = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and _roi_start is not None:
        _roi_current = (x, y)
        _roi_done = True

def calibrate_roi(frame: np.ndarray, cfg: Config) -> Config:
    global _roi_start, _roi_current, _roi_done
    _roi_start, _roi_current, _roi_done = None, None, False

    win = "CALIBRATE ROI (drag box over connector gap area) - press S to save, Q to quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, _roi_mouse)

    while True:
        disp = frame.copy()

        if _roi_start and _roi_current:
            x0, y0 = _roi_start
            x1, y1 = _roi_current
            x = min(x0, x1)
            y = min(y0, y1)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(disp, f"ROI: x={x} y={y} w={w} h={h}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(10) & 0xFF

        if key in (ord('q'), 27):
            cv2.destroyWindow(win)
            return cfg

        if key == ord('s'):
            if _roi_start and _roi_current:
                x0, y0 = _roi_start
                x1, y1 = _roi_current
                x = min(x0, x1)
                y = min(y0, y1)
                w = abs(x1 - x0)
                h = abs(y1 - y0)
                if w >= 10 and h >= 10:
                    cfg.roi = (x, y, w, h)
                    cv2.destroyWindow(win)
                    return cfg


# =========================
# Sources
# =========================

def frames_from_webcam(index: int = 0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()

def frames_from_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()

def frames_from_folder(folder: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = [os.path.join(folder, p) for p in sorted(os.listdir(folder)) if p.lower().endswith(exts)]
    if not paths:
        raise RuntimeError("No images found in folder.")
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        yield frame


# =========================
# Run inspection
# =========================

def run_inspection(frame_iter, cfg: Config):
    if cfg.roi == (0,0,0,0):
        raise RuntimeError("ROI not set. Run --mode calibrate first.")

    output = OutputControl()
    presence = PresenceDetector(cfg)

    red_latched = False
    stable_count = 0
    last_class = None

    for frame in frame_iter:
        H, W = frame.shape[:2]
        x, y, w, h = cfg.roi
        x = max(0, min(W-1, x))
        y = max(0, min(H-1, y))
        w = max(10, min(W-x, w))
        h = max(10, min(H-y, h))

        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        present_now = presence.update(gray)

        if not present_now:
            # reset for next unit
            red_latched = False
            stable_count = 0
            last_class = None
            output.reset()

            disp = frame.copy()
            cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,255), 2)
            cv2.putText(disp, "NO PART (move part in, or press B to baseline empty)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Inspection", disp)

        else:
            gap = measure_gap_px(gray, cfg)

            if gap is None:
                current = "BAD"   # conservative
            else:
                current = "GOOD" if gap <= cfg.good_max_gap_px else "BAD"

            if last_class == current:
                stable_count += 1
            else:
                stable_count = 1
                last_class = current

            if stable_count >= cfg.stable_frames_required:
                if current == "BAD":
                    red_latched = True

            # output
            if red_latched:
                output.red_latched()
            else:
                output.green()

            # display
            disp = frame.copy()
            cv2.rectangle(disp, (x,y), (x+w,y+h), (0,255,255), 2)
            label = "FAIL" if red_latched else "PASS"
            color = (0,0,255) if red_latched else (0,255,0)
            cv2.putText(disp, f"{label}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            if gap is not None:
                cv2.putText(disp, f"Gap(px): {gap}  Thresh: {cfg.good_max_gap_px}",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            else:
                cv2.putText(disp, "Gap(px): N/A (measure failed => BAD)",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Inspection", disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('b'):
            # re-baseline empty background (do this with no part in view)
            presence.rebalance_empty(gray)

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calibrate", "run"], required=True)
    ap.add_argument("--source", choices=["webcam", "video", "folder"], default="folder")
    ap.add_argument("--video", default="", help="Path to video file when --source video")
    ap.add_argument("--folder", default="", help="Path to folder when --source folder")
    ap.add_argument("--webcam_index", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(CONFIG_PATH)

    # get a frame iterator
    if args.source == "webcam":
        iterator = frames_from_webcam(args.webcam_index)
    elif args.source == "video":
        iterator = frames_from_video(args.video)
    else:
        iterator = frames_from_folder(args.folder)

    # Grab first frame for calibration
    first_frame = next(iterator, None)
    if first_frame is None:
        raise RuntimeError("No frames available from source.")

    if args.mode == "calibrate":
        cfg = calibrate_roi(first_frame, cfg)
        save_config(cfg, CONFIG_PATH)
        print("[INFO] Calibration complete. Now run with --mode run")
        return

    # Rebuild iterator (since we consumed one frame)
    if args.source == "webcam":
        iterator = frames_from_webcam(args.webcam_index)
    elif args.source == "video":
        iterator = frames_from_video(args.video)
    else:
        iterator = frames_from_folder(args.folder)

    run_inspection(iterator, cfg)


if __name__ == "__main__":
    main()
