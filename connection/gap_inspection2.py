#!/usr/bin/env python3
"""
Gap Inspection (Basler / pypylon) – robust "dark-gap" detection.

What it does
------------
1) You define an ROI (press R, drag a box, press S to save).
2) In that ROI, it finds the *right-most strong vertical edge* in the left portion
   (this is the housing's right edge).
3) Immediately to the right of that housing edge, it measures how many consecutive
   columns are "dark" (background) before the image becomes "non-dark" again.
   That consecutive dark run length is reported as Gap (px).

Decision (per your latest requirement)
--------------------------------------
PASS (good):  gap_px <= 5
FAIL (bad):   gap_px >  5

UI
--
- Window 1: "Gap Inspection" shows the full frame with ROI box.
- Window 2: "ROI Zoom" shows a zoomed view of the ROI with:
    * Blue line   = housing edge
    * Yellow box  = gap measurement window (to the right of housing edge)
    * Red line    = end of detected dark gap (only if gap_px > 0)

Keys
----
R : enter ROI edit mode (drag with left mouse)
S : save ROI to roi_config.json
ESC / Q : quit

Notes
-----
- Uses Basler via pypylon. If pypylon can't open a camera, it will exit with guidance.
- Uses a median-of-last-N filter + hysteresis to reduce PASS/FAIL bouncing.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections import deque
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from pypylon import pylon
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pypylon is required. Install with: pip install pypylon") from exc


ROI_CONFIG_PATH = Path("roi_config.json")


@dataclass
class Config:
    # ROI in full-frame coordinates
    roi_x: int = 1200
    roi_y: int = 717
    roi_w: int = 184
    roi_h: int = 740

    # Gap decision thresholds
    gap_good_max_px: int = 5  # PASS if <= this

    # How far right of the housing edge we consider for a "gap"
    gap_search_max_px: int = 40  # window width in px to look for dark gap

    # Housing edge search region (fraction of ROI width)
    housing_edge_search_frac: float = 0.70  # search only left 70% for housing edge

    # Edge detection settings
    blur_ksize: int = 5
    sobel_ksize: int = 3
    edge_peak_min_rel: float = 0.25  # min peak relative to max gradient

    # "Dark" definition
    # We compute Otsu threshold on ROI; a column is "dark" if its mean < otsu * dark_scale
    dark_scale: float = 0.90

    # Debounce / smoothing
    gap_history: int = 7          # keep last N gaps
    hysteresis_px: int = 1        # extra margin to reduce bouncing


class RoiEditor:
    def __init__(self):
        self.edit_mode = False
        self.dragging = False
        self.start: Optional[Tuple[int, int]] = None
        self.end: Optional[Tuple[int, int]] = None
        self.frame_w = 0
        self.frame_h = 0
        self.candidate: Optional[Tuple[int, int, int, int]] = None

    def set_frame_size(self, w: int, h: int):
        self.frame_w, self.frame_h = w, h

    def begin(self):
        self.edit_mode = True
        self.dragging = False
        self.start = None
        self.end = None
        self.candidate = None
        print("ROI edit mode: drag a box, press S to save, R to cancel.")

    def cancel(self):
        self.edit_mode = False
        self.dragging = False
        self.start = None
        self.end = None
        self.candidate = None
        print("ROI edit mode cancelled.")

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param=None):
        if not self.edit_mode:
            return
        x = int(np.clip(x, 0, max(0, self.frame_w - 1)))
        y = int(np.clip(y, 0, max(0, self.frame_h - 1)))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)
            self._update_candidate()

    def _update_candidate(self):
        if not self.start or not self.end:
            self.candidate = None
            return
        x0, y0 = self.start
        x1, y1 = self.end
        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)
        if w < 5 or h < 5:
            self.candidate = None
            return
        self.candidate = (x, y, w, h)


def load_config() -> Config:
    cfg = Config()
    if not ROI_CONFIG_PATH.exists():
        return cfg

    try:
        data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        # Allow legacy "4 ints in a text file" (rare but it happened in your earlier attempts)
        try:
            arr = np.loadtxt(str(ROI_CONFIG_PATH), dtype=int).reshape(-1)
            if arr.size >= 4:
                cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h = map(int, arr[:4])
            return cfg
        except Exception:
            return cfg

    # Accept both formats:
    # 1) {"roi_x":..., "roi_y":..., "roi_w":..., "roi_h":...}
    # 2) {"x":..., "y":..., "w":..., "h":...}
    if isinstance(data, dict):
        if {"roi_x", "roi_y", "roi_w", "roi_h"}.issubset(data.keys()):
            cfg.roi_x = int(data["roi_x"])
            cfg.roi_y = int(data["roi_y"])
            cfg.roi_w = int(data["roi_w"])
            cfg.roi_h = int(data["roi_h"])
        elif {"x", "y", "w", "h"}.issubset(data.keys()):
            cfg.roi_x = int(data["x"])
            cfg.roi_y = int(data["y"])
            cfg.roi_w = int(data["w"])
            cfg.roi_h = int(data["h"])
    return cfg


def save_config(cfg: Config):
    ROI_CONFIG_PATH.write_text(
        json.dumps(
            {"roi_x": cfg.roi_x, "roi_y": cfg.roi_y, "roi_w": cfg.roi_w, "roi_h": cfg.roi_h},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved ROI to {ROI_CONFIG_PATH.resolve()}")


def open_basler_camera() -> pylon.InstantCamera:
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()
    if len(devices) == 0:
        raise RuntimeError(
            "No Basler camera detected by pypylon.\n"
            "Checklist:\n"
            "  1) Camera powered + USB3 cable connected\n"
            "  2) Basler pylon runtime installed\n"
            "  3) Another program isn't already using the camera\n"
            "  4) Try unplug/replug and reboot\n"
        )
    cam = pylon.InstantCamera(factory.CreateDevice(devices[0]))
    cam.Open()

    # Conservative defaults; you can tune later.
    # Some models allow these nodes; ignore if not present.
    try:
        cam.PixelFormat.SetValue("Mono8")
    except Exception:
        pass
    try:
        cam.GainAuto.SetValue("Off")
    except Exception:
        pass
    try:
        cam.ExposureAuto.SetValue("Off")
    except Exception:
        pass

    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    return cam


def grab_frame(cam: pylon.InstantCamera) -> np.ndarray:
    grab = cam.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)
    try:
        if not grab.GrabSucceeded():
            raise RuntimeError("Grab failed")
        img = grab.Array  # ndarray
        # Ensure uint8 2D
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.asarray(img)
        if img.dtype != np.uint8:
            img = cv2.convertScaleAbs(img)
        return img
    finally:
        grab.Release()


def find_housing_edge_x(roi_gray: np.ndarray, cfg: Config) -> Optional[int]:
    """
    Returns ROI-local x coordinate of the housing right edge.

    Approach:
      - Compute |dI/dx| via Sobel.
      - Collapse to 1D profile by summing over y.
      - Consider only left fraction (where housing edge should be).
      - Find peaks above a relative threshold; pick the RIGHTMOST such peak.
    """
    if cfg.blur_ksize >= 3 and cfg.blur_ksize % 2 == 1:
        roi_blur = cv2.GaussianBlur(roi_gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    else:
        roi_blur = roi_gray

    gx = cv2.Sobel(roi_blur, cv2.CV_32F, 1, 0, ksize=cfg.sobel_ksize)
    prof = np.sum(np.abs(gx), axis=0)

    w = roi_gray.shape[1]
    search_w = max(5, int(w * cfg.housing_edge_search_frac))
    prof_search = prof[:search_w]
    if prof_search.size < 5:
        return None

    maxv = float(np.max(prof_search))
    if maxv <= 1e-6:
        return None

    # Candidate indices where profile is "strong"
    strong = np.where(prof_search >= (cfg.edge_peak_min_rel * maxv))[0]
    if strong.size == 0:
        return None

    # Rightmost strong index is robust here (housing edge is the right boundary of the housing area)
    return int(strong[-1])


def measure_gap_px(roi_gray: np.ndarray, housing_edge_x: int, cfg: Config) -> Tuple[int, int, int, float]:
    """
    Measures the dark-gap run length in pixels immediately to the right of the housing edge.

    Returns:
        gap_px, win_x1, win_x2, dark_thresh
    """
    h, w = roi_gray.shape[:2]
    x1 = int(np.clip(housing_edge_x + 1, 0, w))
    x2 = int(np.clip(housing_edge_x + 1 + cfg.gap_search_max_px, 0, w))
    if x2 <= x1 + 1:
        return 0, x1, x2, 0.0

    # Otsu threshold on ROI -> split bright/dark; then slightly bias toward "dark"
    _t, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_thresh = float(_t) * float(cfg.dark_scale)

    window = roi_gray[:, x1:x2]
    col_mean = window.mean(axis=0)

    # Count consecutive "dark" columns from the start of the window
    gap = 0
    for v in col_mean:
        if float(v) < dark_thresh:
            gap += 1
        else:
            break

    return int(gap), x1, x2, float(dark_thresh)


def draw_overlays_full(frame_bgr: np.ndarray, cfg: Config, status: str, gap_px: Optional[int]) -> np.ndarray:
    out = frame_bgr.copy()
    x, y, w, h = cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Big status text
    color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
    cv2.putText(out, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6, cv2.LINE_AA)
    if gap_px is not None:
        cv2.putText(out, f"Gap: {gap_px}px", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    return out


def draw_overlays_zoom(
    roi_gray: np.ndarray,
    housing_edge_x: Optional[int],
    gap_px: Optional[int],
    win_x1: Optional[int],
    win_x2: Optional[int],
    cfg: Config,
    zoom: int = 4,
) -> np.ndarray:
    roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    h, w = roi_gray.shape[:2]

    if housing_edge_x is not None:
        cv2.line(roi_bgr, (housing_edge_x, 0), (housing_edge_x, h - 1), (255, 0, 0), 1)  # blue

    if win_x1 is not None and win_x2 is not None:
        cv2.rectangle(roi_bgr, (win_x1, 0), (win_x2 - 1, h - 1), (0, 255, 255), 1)  # yellow

    if housing_edge_x is not None and gap_px is not None and gap_px > 0:
        end_x = int(np.clip(housing_edge_x + 1 + gap_px, 0, w - 1))
        cv2.line(roi_bgr, (end_x, 0), (end_x, h - 1), (0, 0, 255), 1)  # red

    zoomed = cv2.resize(roi_bgr, (w * zoom, h * zoom), interpolation=cv2.INTER_NEAREST)

    # Re-draw lines with thickness in zoomed coordinates (so they're visible)
    if housing_edge_x is not None:
        x = housing_edge_x * zoom
        cv2.line(zoomed, (x, 0), (x, zoomed.shape[0] - 1), (255, 0, 0), 2)

    if win_x1 is not None and win_x2 is not None:
        x1 = win_x1 * zoom
        x2 = (win_x2 - 1) * zoom
        cv2.rectangle(zoomed, (x1, 0), (x2, zoomed.shape[0] - 1), (0, 255, 255), 2)

    if housing_edge_x is not None and gap_px is not None and gap_px > 0:
        end_x = int(np.clip((housing_edge_x + 1 + gap_px) * zoom, 0, zoomed.shape[1] - 1))
        cv2.line(zoomed, (end_x, 0), (end_x, zoomed.shape[0] - 1), (0, 0, 255), 2)

    # Text overlay in zoom window
    y0 = 30
    cv2.putText(zoomed, "Blue: housing edge", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(zoomed, "Yellow: gap window", (10, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(zoomed, "Red: gap end", (10, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    if gap_px is not None:
        cv2.putText(
            zoomed,
            f"Gap = {gap_px}px (PASS <= {cfg.gap_good_max_px})",
            (10, y0 + 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return zoomed


def clamp_roi(cfg: Config, frame_w: int, frame_h: int) -> None:
    cfg.roi_x = int(np.clip(cfg.roi_x, 0, max(0, frame_w - 1)))
    cfg.roi_y = int(np.clip(cfg.roi_y, 0, max(0, frame_h - 1)))
    cfg.roi_w = int(np.clip(cfg.roi_w, 10, frame_w - cfg.roi_x))
    cfg.roi_h = int(np.clip(cfg.roi_h, 10, frame_h - cfg.roi_y))


def main():
    cfg = load_config()

    cam = open_basler_camera()
    print("Basler camera opened.")

    roi_editor = RoiEditor()

    # Median filter + hysteresis state
    gap_hist: deque[int] = deque(maxlen=cfg.gap_history)
    status = "PASS"  # initial guess

    # Windows
    cv2.namedWindow("Gap Inspection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI Zoom", cv2.WINDOW_NORMAL)

    def _mouse(event, x, y, flags, param):
        roi_editor.on_mouse(event, x, y, flags, param)

    cv2.setMouseCallback("Gap Inspection", _mouse)

    while True:
        frame = grab_frame(cam)
        frame_h, frame_w = frame.shape[:2]
        roi_editor.set_frame_size(frame_w, frame_h)

        clamp_roi(cfg, frame_w, frame_h)

        roi = frame[cfg.roi_y : cfg.roi_y + cfg.roi_h, cfg.roi_x : cfg.roi_x + cfg.roi_w].copy()
        housing_edge_x = find_housing_edge_x(roi, cfg)

        gap_px: Optional[int] = None
        win_x1 = win_x2 = None

        if housing_edge_x is not None:
            g, x1, x2, _dark_t = measure_gap_px(roi, housing_edge_x, cfg)
            gap_px = g
            win_x1, win_x2 = x1, x2

            # Update smoothing history
            gap_hist.append(gap_px)
            gap_med = int(np.median(list(gap_hist))) if len(gap_hist) > 0 else gap_px

            # Hysteresis decision to reduce bouncing
            if status == "PASS":
                # Switch to FAIL only if clearly above threshold + hysteresis
                if gap_med > cfg.gap_good_max_px + cfg.hysteresis_px:
                    status = "FAIL"
            else:
                # Switch to PASS only if clearly below (or equal) threshold - hysteresis
                if gap_med <= max(0, cfg.gap_good_max_px - cfg.hysteresis_px):
                    status = "PASS"

            gap_px_display = gap_med
        else:
            status = "FAIL"
            gap_px_display = None

        # Draw full frame overlay
        full_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        full_view = draw_overlays_full(full_bgr, cfg, status, gap_px_display)

        # If in ROI edit mode, draw drag rectangle / candidate
        if roi_editor.edit_mode:
            if roi_editor.start and roi_editor.end:
                x0, y0 = roi_editor.start
                x1, y1 = roi_editor.end
                cv2.rectangle(full_view, (x0, y0), (x1, y1), (255, 255, 0), 2)
            if roi_editor.candidate:
                x, y, w, h = roi_editor.candidate
                cv2.putText(full_view, f"Candidate ROI: x={x}, y={y}, w={w}, h={h}",
                            (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)

        # Draw ROI zoom window
        zoom_view = draw_overlays_zoom(roi, housing_edge_x, gap_px_display, win_x1, win_x2, cfg, zoom=4)

        cv2.imshow("Gap Inspection", full_view)
        cv2.imshow("ROI Zoom", zoom_view)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("r"), ord("R")):
            if roi_editor.edit_mode:
                roi_editor.cancel()
            else:
                roi_editor.begin()
        if key in (ord("s"), ord("S")):
            if roi_editor.edit_mode and roi_editor.candidate:
                cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h = roi_editor.candidate
                save_config(cfg)
                roi_editor.cancel()

    try:
        cam.StopGrabbing()
    except Exception:
        pass
    try:
        cam.Close()
    except Exception:
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
