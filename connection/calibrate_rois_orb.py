#!/usr/bin/env python3
"""calibrate_rois_orb.py

Purpose
-------
Interactive calibration tool to define:
  1) ORB TEMPLATE ROI (anchor region)  -> saved as template.png
  2) SEAM ROI (measurement region)     -> saved as a 4-corner quad in template coordinates

This keeps the same "drag rectangles in an OpenCV window" workflow used in baseline_script.py,
but stores seam ROI as a quad relative to the TEMPLATE ROI origin so it can be transformed later
with ORB+Homography.

Outputs
-------
  template.png
  vision_config.json  (contains seam_quad_in_template + optional warp size + baseline placeholder)

Keys
----
  T : select TEMPLATE ROI (drag with mouse)
  S : select SEAM ROI (drag with mouse)
  W : write template + config
  ESC : exit without saving

Notes
-----
- Close Basler pylon Viewer before running (exclusive camera access).
- Seam quad is stored in template coordinate system with origin at TEMPLATE ROI top-left.
  The quad may extend outside the template image bounds; that's OK and expected.
"""

import os
import json
from datetime import datetime
import cv2
import numpy as np
from pypylon import pylon

CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"

# Optional: a fixed warp size for the seam patch in the later ORB+homography runtime.
# You can tweak later.
DEFAULT_WARP_W = 420
DEFAULT_WARP_H = 200

# Visual / UI
WINDOW_CAL = "CALIBRATION (ORB Anchor + Seam)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_ESC = 27


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class ROISelector:
    """Mouse-drag rectangle selector."""
    def __init__(self, win):
        self.win = win
        self.dragging = False
        self.x0 = self.y0 = 0
        self.x1 = self.y1 = 0
        self.last_roi = None  # (x,y,w,h)

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.x0, self.y0 = x, y
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.x1, self.y1 = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            x_min, x_max = sorted([self.x0, x])
            y_min, y_max = sorted([self.y0, y])
            w = x_max - x_min
            h = y_max - y_min
            if w > 10 and h > 10:
                self.last_roi = (x_min, y_min, w, h)


def start_camera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    try:
        camera.ExposureAuto.SetValue("Off")
        camera.GainAuto.SetValue("Off")
    except Exception:
        pass
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return camera, converter


def rect_to_quad(x, y, w, h):
    # Order: top-left, top-right, bottom-right, bottom-left
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def main():
    print("NOTE: Close Basler pylon Viewer before running (exclusive camera access).")

    camera = None
    try:
        camera, converter = start_camera()

        cv2.namedWindow(WINDOW_CAL, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_CAL, 1400, 900)
        cv2.moveWindow(WINDOW_CAL, 20, 20)

        selector = ROISelector(WINDOW_CAL)
        cv2.setMouseCallback(WINDOW_CAL, selector.callback)

        mode = "TEMPLATE"  # or "SEAM"
        template_roi = None
        seam_roi = None

        print("\n--- CALIBRATION ---")
        print("Click the calibration window so it has focus.")
        print("T = TEMPLATE ROI (ORB anchor)  |  S = SEAM ROI  |  W = Write  |  ESC = Exit")

        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                grab.Release()
                continue

            frame = converter.Convert(grab).GetArray()  # grayscale
            grab.Release()

            disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Accept last drag into the current mode
            if selector.last_roi is not None:
                if mode == "TEMPLATE":
                    template_roi = selector.last_roi
                else:
                    seam_roi = selector.last_roi
                selector.last_roi = None

            # Draw ROIs
            if template_roi is not None:
                tx, ty, tw, th = template_roi
                cv2.rectangle(disp, (tx, ty), (tx + tw, ty + th), (255, 0, 0), 2)
                cv2.putText(disp, "TEMPLATE ROI (ORB anchor)", (tx, max(20, ty - 10)), FONT, 0.8, (255, 0, 0), 2)

            if seam_roi is not None:
                sx, sy, sw, sh = seam_roi
                cv2.rectangle(disp, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
                cv2.putText(disp, "SEAM ROI (measurement)", (sx, max(20, sy - 10)), FONT, 0.8, (0, 255, 255), 2)

            # On-screen instructions
            cv2.putText(disp, f"Mode: {mode}   (T=Template, S=Seam, W=Write, ESC=Exit)", (20, 40), FONT, 0.9, (255, 255, 255), 2)
            cv2.putText(disp, "Drag a rectangle with the mouse. Release to accept.", (20, 80), FONT, 0.8, (255, 255, 255), 2)

            if template_roi is None or seam_roi is None:
                cv2.putText(disp, "Define BOTH TEMPLATE and SEAM ROI before writing.", (20, 120), FONT, 0.8, (0, 0, 255), 2)

            cv2.imshow(WINDOW_CAL, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == KEY_ESC:
                print("Calibration cancelled.")
                break
            if k in (ord('t'), ord('T')):
                mode = "TEMPLATE"
            if k in (ord('s'), ord('S')):
                mode = "SEAM"
            if k in (ord('w'), ord('W')):
                if template_roi is None or seam_roi is None:
                    print("[CAL] Need BOTH TEMPLATE ROI and SEAM ROI before writing.")
                    continue

                tx, ty, tw, th = template_roi
                sx, sy, sw, sh = seam_roi

                # Save template image crop
                template_img = frame[ty:ty + th, tx:tx + tw].copy()
                cv2.imwrite(TEMPLATE_PATH, template_img)
                print("[CAL] Saved", TEMPLATE_PATH)

                # Seam quad in TEMPLATE coordinates (origin = template top-left)
                seam_quad_frame = rect_to_quad(sx, sy, sw, sh)
                seam_quad_template = [(int(x - tx), int(y - ty)) for (x, y) in seam_quad_frame]

                cfg = {
                    "seam_quad_in_template": seam_quad_template,
                    "warp_size": {"w": int(DEFAULT_WARP_W), "h": int(DEFAULT_WARP_H)},
                    "baseline_boundary_x": None,
                    "notes": {
                        "template_roi_in_frame": [int(tx), int(ty), int(tw), int(th)],
                        "seam_roi_in_frame": [int(sx), int(sy), int(sw), int(sh)],
                        "created": stamp()
                    }
                }

                with open(CONFIG_PATH, "w") as f:
                    json.dump(cfg, f, indent=2)
                print("[CFG] Saved", CONFIG_PATH)
                print("[CFG] seam_quad_in_template:", seam_quad_template)
                print("Done. You can now move to the ORB+homography runtime script.")
                break

    except Exception as e:
        print("\nERROR:", e)
        print("If you see 'Device is exclusively opened by another client', close pylon Viewer and retry.\n")
    finally:
        try:
            if camera is not None:
                camera.StopGrabbing()
                camera.Close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
