#!/usr/bin/env python3
"""orb_homography_inspection_v26_gap_darkness_otsu_band_tuning_keysafe.py

Otsu-based dark-gap inspection with keyboard-layout-safe band tuning.

Why this exists
---------------
Some systems do not deliver '[' and ']' reliably to OpenCV (keyboard layout,
window focus, etc.). This version supports BOTH:
  - i / k : move band up / down   (recommended)
  - j / l : shrink / grow band    (recommended)
And also supports:
  - [ / ] : move band up / down
  - - / = : shrink / grow band

What it does
------------
- Uses ORB + RANSAC homography to map a seam quad from template space into the live frame.
- Warps that seam quad into a normalized seam patch.
- Computes a "dark gap" metric on a horizontal band within the seam patch:
    * blur
    * Otsu threshold (dark pixels become 1)
    * dark_ratio = fraction of dark pixels in band
    * max_run = longest contiguous run of "gap columns" across the width
- Declares BAD (latched) when dark metrics exceed thresholds.
- Declares UNSTABLE only when tracking is lost / seam out of view / empty patch.

Hotkeys (click the Trial Inspection window first!)
--------------------------------------------------
Band tuning:
  i / [   : move band UP
  k / ]   : move band DOWN
  j / -   : shrink band height
  l / =   : grow band height
  s       : save band fractions to vision_config.json
  o       : reset band to defaults

Other:
  r       : reset BAD latch
  m       : toggle per-part latch
  d       : toggle debug window
  p       : pause processing
  ESC     : quit

Required files in the same folder
---------------------------------
- template.png
- vision_config.json (must include seam_quad_in_template and warp_size)

"""

import os
import json
import traceback
from collections import deque

import cv2
import numpy as np
from pypylon import pylon

CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"

# -------------------------- ORB / Homography --------------------------
ORB_NFEATURES = 2500
ORB_KEEP_BEST = 120
RANSAC_REPROJ_THRESH = 4.0
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.25
HOLD_LAST_GOOD_FRAMES = 10

# -------------------------- Part present state machine --------------------------
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# -------------------------- Band defaults & tuning --------------------------
DEFAULT_Y0_FRAC = 0.40
DEFAULT_Y1_FRAC = 0.65
SHIFT_STEP_FRAC = 0.02
SIZE_STEP_FRAC = 0.02
MIN_BAND_HEIGHT = 0.04

# -------------------------- Otsu parameters --------------------------
BAND_BLUR_K = 5  # odd
COL_DARK_RATIO_AT_OR_ABOVE = 0.55  # stricter to avoid false full-width runs

# -------------------------- Decision thresholds --------------------------
DARK_RATIO_FAIL = 0.35
MAX_RUN_FAIL = 30
DARK_RATIO_HARD = 0.30
MAX_RUN_HARD = 60

# -------------------------- Smoothing --------------------------
METRIC_HISTORY = 7

# -------------------------- Latch behavior --------------------------
PER_PART_LATCH_DEFAULT = True

# -------------------------- UI --------------------------
WINDOW_MAIN = "Trial Inspection"
WINDOW_DEBUG = "DEBUG VIEW"
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_ESC = 27


def to_gray(frame: np.ndarray) -> np.ndarray | None:
    """Ensure frame is 2D grayscale."""
    if frame is None:
        return None
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return None


def load_config() -> dict | None:
    if not (os.path.exists(CONFIG_PATH) and os.path.exists(TEMPLATE_PATH)):
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("[CFG] Saved", CONFIG_PATH)


def start_camera():
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    cam.Open()
    try:
        cam.ExposureAuto.SetValue("Off")
        cam.GainAuto.SetValue("Off")
    except Exception:
        pass
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    conv = pylon.ImageFormatConverter()
    conv.OutputPixelFormat = pylon.PixelType_Mono8
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return cam, conv


def estimate_homography_orb(template_gray: np.ndarray, frame_gray: np.ndarray):
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(frame_gray, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None, {"reason": "not_enough_features", "kp1": len(kp1), "kp2": len(kp2)}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return None, {"reason": "no_matches"}

    matches = sorted(matches, key=lambda m: m.distance)[: min(len(matches), ORB_KEEP_BEST)]
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_REPROJ_THRESH)
    if H is None or inliers is None:
        return None, {"reason": "homography_failed", "matches": len(matches)}

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(1, len(matches))
    return H, {"matches": len(matches), "inliers": inlier_count, "inlier_ratio": float(inlier_ratio)}


def perspective_transform_points(H: np.ndarray, pts):
    pts_np = np.float32(pts).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts_np, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def order_quad_tltrbrbl(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return [
        (float(tl[0]), float(tl[1])),
        (float(tr[0]), float(tr[1])),
        (float(br[0]), float(br[1])),
        (float(bl[0]), float(bl[1])),
    ]


def quad_in_frame(quad, w: int, h: int, margin: int = 2) -> bool:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs) >= -margin) and (min(ys) >= -margin) and (max(xs) <= (w - 1 + margin)) and (max(ys) <= (h - 1 + margin))


def warp_quad_to_rect(gray: np.ndarray, quad_pts, out_w: int, out_h: int) -> np.ndarray | None:
    if out_w <= 1 or out_h <= 1:
        return None
    src = np.float32(quad_pts)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    patch = cv2.warpPerspective(gray, M, (out_w, out_h))
    if patch is None or patch.size == 0:
        return None
    return patch


def longest_run_1d(mask_1d: np.ndarray) -> int:
    if mask_1d is None:
        return 0
    mask_1d = np.asarray(mask_1d).astype(np.uint8).reshape(-1)
    if mask_1d.size == 0 or mask_1d.max() == 0:
        return 0
    p = np.concatenate(([0], mask_1d, [0]))
    d = np.diff(p)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return 0
    n = min(starts.size, ends.size)
    if n <= 0:
        return 0
    runs = ends[:n] - starts[:n]
    return int(np.max(runs)) if runs.size else 0

def compute_gap_metrics_otsu(seam_patch: np.ndarray, y0_frac: float, y1_frac: float):

    if seam_patch is None or seam_patch.size == 0:
        return None, None, None, None

    h, w = seam_patch.shape[:2]

    # ---------- AUTO SEAM DETECTION ----------
    # detect horizontal edges
    sobel = cv2.Sobel(seam_patch, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.abs(sobel)

    # average edge strength per row
    row_energy = sobel.mean(axis=1)

    # strongest horizontal edge
    seam_row = int(np.argmax(row_energy))

    # band height = ~12% of seam patch
    band_half = int(h * 0.06)

    y0 = int(np.clip(seam_row - band_half, 0, h - 2))
    y1 = int(np.clip(seam_row + band_half, y0 + 1, h))

    band = seam_patch[y0:y1, :]

    if band is None or band.size == 0:
        return None, None, None, (y0, y1)

    # ---------- BLUR ----------
    k = BAND_BLUR_K if BAND_BLUR_K % 2 == 1 else BAND_BLUR_K + 1
    band_blur = cv2.GaussianBlur(band, (k, k), 0)

    # ---------- OTSU THRESHOLD ----------
    thr, _ = cv2.threshold(
        band_blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # clamp threshold
    thr = min(thr, 120)

    # ---------- DARK PIXEL MASK ----------
    mask = (band_blur < thr).astype(np.uint8) * 255

    dark_ratio = float(mask.mean() / 255.0)

    col_dark_ratio = mask.mean(axis=0) / 255.0
    col_mask = col_dark_ratio >= COL_DARK_RATIO_AT_OR_ABOVE

    max_run = longest_run_1d(col_mask)

    return dark_ratio, max_run, float(thr), (y0, y1)


def clamp_band(y0: float, y1: float):
    y0 = float(np.clip(y0, 0.0, 0.98))
    y1 = float(np.clip(y1, 0.02, 1.0))
    if y1 - y0 < MIN_BAND_HEIGHT:
        mid = (y0 + y1) / 2.0
        y0 = max(0.0, mid - MIN_BAND_HEIGHT / 2.0)
        y1 = min(1.0, mid + MIN_BAND_HEIGHT / 2.0)
    if y0 >= y1:
        y0 = max(0.0, y1 - MIN_BAND_HEIGHT)
    return y0, y1


def main():
    cfg = load_config()
    if cfg is None:
        print("Missing vision_config.json or template.png.")
        return

    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("template.png missing or unreadable.")
        return

    seam_quad_t = cfg.get("seam_quad_in_template")
    if seam_quad_t is None or len(seam_quad_t) != 4:
        print("vision_config.json missing seam_quad_in_template.")
        return
    seam_quad_t = [(float(p[0]), float(p[1])) for p in seam_quad_t]

    warp = cfg.get("warp_size", {"w": 420, "h": 200})
    warp_w = int(warp.get("w", 420))
    warp_h = int(warp.get("h", 200))

    y0_frac = float(cfg.get("gap_band_y0_frac", DEFAULT_Y0_FRAC))
    y1_frac = float(cfg.get("gap_band_y1_frac", DEFAULT_Y1_FRAC))
    y0_frac, y1_frac = clamp_band(y0_frac, y1_frac)

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, 950, 650)
    cv2.moveWindow(WINDOW_MAIN, 20, 20)

    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_DEBUG, 1400, 900)
    cv2.moveWindow(WINDOW_DEBUG, 1000, 20)

    state = "WAIT_FOR_PART"
    present_ct = 0
    leave_ct = 0

    bad = False
    part_failed = False
    per_part_latch = PER_PART_LATCH_DEFAULT

    dark_hist = deque(maxlen=METRIC_HISTORY)
    run_hist = deque(maxlen=METRIC_HISTORY)

    last_good_H = None
    hold_left = 0
    last_orb = {}

    paused = False
    show_debug = True

    cam = None

    try:
        cam, conv = start_camera()

        while cam.IsGrabbing():
            grab = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                grab.Release()
                continue

            frame = conv.Convert(grab).GetArray()
            grab.Release()
            frame = to_gray(frame)
            if frame is None:
                raise RuntimeError("Could not convert grabbed frame to grayscale")

            k = cv2.waitKey(1) & 0xFF
            if k == KEY_ESC:
                break

            # --- Hotkeys ---
            if k in (ord('d'), ord('D')):
                show_debug = not show_debug
                if not show_debug:
                    try:
                        cv2.destroyWindow(WINDOW_DEBUG)
                    except Exception:
                        pass
                else:
                    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_DEBUG, 1400, 900)
                    cv2.moveWindow(WINDOW_DEBUG, 1000, 20)

            if k in (ord('p'), ord('P')):
                paused = not paused

            if k in (ord('m'), ord('M')):
                per_part_latch = not per_part_latch

            if k in (ord('r'), ord('R')):
                bad = False
                part_failed = False
                dark_hist.clear(); run_hist.clear()

            if k in (ord('o'), ord('O')):
                y0_frac, y1_frac = clamp_band(DEFAULT_Y0_FRAC, DEFAULT_Y1_FRAC)

            # move band (i/k + [ ])
            if k in (ord('i'), ord('I'), ord('[')):
                y0_frac, y1_frac = clamp_band(y0_frac - SHIFT_STEP_FRAC, y1_frac - SHIFT_STEP_FRAC)
            if k in (ord('k'), ord('K'), ord(']')):
                y0_frac, y1_frac = clamp_band(y0_frac + SHIFT_STEP_FRAC, y1_frac + SHIFT_STEP_FRAC)

            # resize band (j/l + -/=)
            if k in (ord('j'), ord('J'), ord('-')):
                mid = (y0_frac + y1_frac) / 2.0
                half = (y1_frac - y0_frac) / 2.0
                half = max(MIN_BAND_HEIGHT / 2.0, half - SIZE_STEP_FRAC / 2.0)
                y0_frac, y1_frac = clamp_band(mid - half, mid + half)
            if k in (ord('l'), ord('L'), ord('=')):
                mid = (y0_frac + y1_frac) / 2.0
                half = (y1_frac - y0_frac) / 2.0
                half = min(0.45, half + SIZE_STEP_FRAC / 2.0)
                y0_frac, y1_frac = clamp_band(mid - half, mid + half)

            # save band
            if k in (ord('s'), ord('S')):
                cfg["gap_band_y0_frac"] = float(y0_frac)
                cfg["gap_band_y1_frac"] = float(y1_frac)
                save_config(cfg)

            # --- ORB homography ---
            H, orb = estimate_homography_orb(template, frame)
            last_orb = orb

            use_H = None
            if H is not None and int(orb.get("inliers", 0)) >= MIN_INLIERS and float(orb.get("inlier_ratio", 0.0)) >= MIN_INLIER_RATIO:
                use_H = H
                last_good_H = H
                hold_left = HOLD_LAST_GOOD_FRAMES
            elif last_good_H is not None and hold_left > 0:
                use_H = last_good_H
                hold_left -= 1

            part_present = (use_H is not None)

            # --- State machine ---
            if not paused:
                if state == "WAIT_FOR_PART":
                    present_ct = present_ct + 1 if part_present else 0
                    if present_ct >= ENTER_PRESENT_FRAMES:
                        state = "INSPECT"
                        present_ct = leave_ct = 0
                        bad = False
                        part_failed = False
                        dark_hist.clear(); run_hist.clear()
                else:
                    leave_ct = leave_ct + 1 if not part_present else 0
                    if leave_ct >= LEAVE_FRAMES:
                        state = "WAIT_FOR_PART"
                        leave_ct = present_ct = 0
                        bad = False
                        part_failed = False

            # --- Seam patch & metrics ---
            seam_patch = None
            seam_quad_frame = None
            dark_ratio = None
            max_run = None
            otsu_thr = None
            band_y = None

            unstable = False
            unstable_reason = ""

            if state == "INSPECT":
                if use_H is None:
                    unstable = True
                    unstable_reason = "TRACKING LOST"
                else:
                    seam_quad_frame = order_quad_tltrbrbl(perspective_transform_points(use_H, seam_quad_t))
                    fh, fw = frame.shape[:2]
                    if not quad_in_frame(seam_quad_frame, fw, fh, margin=5):
                        unstable = True
                        unstable_reason = "SEAM OUT OF VIEW"
                    else:
                        seam_patch = warp_quad_to_rect(frame, seam_quad_frame, warp_w, warp_h)
                        if seam_patch is None or seam_patch.size == 0:
                            unstable = True
                            unstable_reason = "EMPTY SEAM PATCH"

            if (not paused) and (not unstable) and seam_patch is not None:
                dark_ratio, max_run, otsu_thr, band_y = compute_gap_metrics_otsu(seam_patch, y0_frac, y1_frac)
                if dark_ratio is None or max_run is None:
                    unstable = True
                    unstable_reason = "EMPTY GAP BAND"
                else:
                    dark_hist.append(dark_ratio)
                    run_hist.append(max_run)
                    dark_med = float(np.median(np.array(dark_hist, dtype=np.float32)))
                    run_med = int(np.median(np.array(run_hist, dtype=np.int32)))

                    if (dark_med >= DARK_RATIO_HARD) or (run_med >= MAX_RUN_HARD):
                        bad = True
                    elif (dark_med >= DARK_RATIO_FAIL) or (run_med >= MAX_RUN_FAIL):
                        bad = True

                    if per_part_latch and ((dark_med >= DARK_RATIO_FAIL) or (run_med >= MAX_RUN_FAIL)):
                        part_failed = True
                    if part_failed:
                        bad = True

            # --- UI ---
            bg = np.zeros((650, 950, 3), dtype=np.uint8)
            txt = (0, 0, 0)

            if state != "INSPECT" or unstable:
                bg[:] = (0, 165, 255)
                label = "UNSTABLE - " + (unstable_reason if unstable_reason else "POSITION PART")
            elif bad:
                bg[:] = (0, 0, 255)
                label = "BAD"
                txt = (255, 255, 255)
            else:
                bg[:] = (0, 255, 0)
                label = "GOOD"

            cv2.putText(bg, label, (30, 110), FONT, 1.6, txt, 5)
            cv2.putText(bg, f"State: {state}", (30, 185), FONT, 1.0, txt, 2)
            cv2.putText(bg, f"Band y0={y0_frac:.3f} y1={y1_frac:.3f}  (i/k move, j/l size, s save, o reset)", (30, 225), FONT, 0.75, txt, 2)
            cv2.putText(bg, f"dark_ratio={None if dark_ratio is None else f'{dark_ratio:.3f}'} (fail>={DARK_RATIO_FAIL})", (30, 260), FONT, 0.85, txt, 2)
            cv2.putText(bg, f"max_run={None if max_run is None else f'{max_run}'} px (fail>={MAX_RUN_FAIL})", (30, 295), FONT, 0.85, txt, 2)
            cv2.putText(bg, f"otsu_thr={None if otsu_thr is None else f'{otsu_thr:.0f}'}  inliers={last_orb.get('inliers',0)} ratio={last_orb.get('inlier_ratio',0):.2f}", (30, 335), FONT, 0.75, txt, 2)
            cv2.putText(bg, "Keys: r=Reset  m=PerPart  d=Debug  p=Pause  ESC=Quit", (30, 620), FONT, 0.7, txt, 2)

            cv2.imshow(WINDOW_MAIN, bg)

            if show_debug:
                dbg = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if seam_quad_frame is not None:
                    cv2.polylines(dbg, [np.int32(seam_quad_frame).reshape(-1, 1, 2)], True, (0, 255, 255), 2)

                if seam_patch is not None:
                    patch_vis = cv2.cvtColor(seam_patch, cv2.COLOR_GRAY2BGR)
                    if band_y is not None:
                        yy0, yy1 = band_y
                        cv2.rectangle(patch_vis, (0, yy0), (warp_w - 1, yy1), (0, 255, 255), 2)
                        if otsu_thr is not None:
                            cv2.putText(patch_vis, f"otsu_thr={otsu_thr:.0f}", (5, max(15, yy0+18)), FONT, 0.6, (0, 255, 255), 2)

                    scale = 1.6
                    patch_vis = cv2.resize(patch_vis, (int(warp_w * scale), int(warp_h * scale)))
                    ph, pw = patch_vis.shape[:2]
                    dbg[0:ph, 0:pw] = patch_vis
                    cv2.rectangle(dbg, (0, 0), (pw, ph), (255, 255, 255), 1)

                cv2.imshow(WINDOW_DEBUG, dbg)

    except Exception as e:
        print("\nERROR:", e)
        print(traceback.format_exc())

    finally:
        if cam is not None:
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
