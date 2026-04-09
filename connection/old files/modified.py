#!/usr/bin/env python3
"""orb_homography_inspection_v21_gap_darkness_robust.py

Production-tolerant connector seating inspection (dark-gap detection)
===================================================================

This version is a robust update of the dark-gap approach and addresses the
broadcasting/shape errors you hit when the seam patch/band becomes empty or
the frame arrives as 3-channel.

Core idea
---------
You confirmed the defect signature:
- When the connector is NOT seated, a dark gap appears between controller and connector.

So we:
1) Use ORB + RANSAC homography to locate the part.
2) Warp the seam ROI into a normalized patch.
3) Measure "dark gap" in a band where the gap appears.
4) Trip BAD (latched) when dark-gap metrics exceed thresholds.

UNSTABLE is shown only when:
- ORB tracking is lost (no homography), or
- seam quad falls out of the frame, or
- seam patch/band is empty.

Baseline capture
---------------
Baseline is used ONLY to compute robust band intensity stats (median & MAD)
from a seated connector. The script auto-derives a dark-pixel threshold:
    thresh = median - THRESH_K * MAD

Keys
----
b  capture baseline (seated ONLY)
u  clear baseline stats
r  reset BAD latch
m  toggle per-part latch
p  pause
d  debug
ESC quit

"""

import os
import json
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from pypylon import pylon

CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"
FAIL_DIR = "fails"

# -------------------------- ORB / Homography --------------------------
ORB_NFEATURES = 2500
ORB_KEEP_BEST = 120
RANSAC_REPROJ_THRESH = 4.0
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.25  # production tolerant
HOLD_LAST_GOOD_FRAMES = 10

# -------------------------- Part present state machine --------------------------
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# -------------------------- Baseline capture (timeboxed) --------------------------
BASELINE_CAPTURE_SAMPLES = 25
BASELINE_CAPTURE_DURATION_SEC = 90
BASELINE_CAPTURE_MIN_SAMPLES = 20

# -------------------------- Gap band settings (in seam patch coordinates) --------------------------
# Fractions of seam patch height. Adjust after checking the patch preview.
GAP_BAND_Y0_FRAC = 0.40
GAP_BAND_Y1_FRAC = 0.65

# Robust thresholding from baseline band statistics
THRESH_K = 2.5

# Decision thresholds (start values; tune with your data)
DARK_RATIO_FAIL = 0.18
MAX_RUN_FAIL = 35
DARK_RATIO_HARD = 0.30
MAX_RUN_HARD = 70

# History smoothing
METRIC_HISTORY = 7

# Latch behavior
PER_PART_LATCH_DEFAULT = True

# UI
WINDOW_MAIN = "Trial Inspection"
WINDOW_DEBUG = "DEBUG VIEW"
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_ESC = 27


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def save_fail_images(frame_gray: np.ndarray, annotated_bgr: np.ndarray, info_text: str = "") -> None:
    ensure_dir(FAIL_DIR)
    ts = stamp()
    raw_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_raw.png")
    ann_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_annotated.png")
    txt_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_info.txt")
    cv2.imwrite(raw_path, frame_gray)
    cv2.imwrite(ann_path, annotated_bgr)
    if info_text:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(info_text)


def to_gray(frame: np.ndarray) -> np.ndarray:
    """Ensure a 2D uint8 grayscale image."""
    if frame is None:
        return None
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        # Handle Mono8 accidentally wrapped, or BGR
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return None


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


def warp_quad_to_rect(gray: np.ndarray, quad_pts, out_w: int, out_h: int) -> np.ndarray:
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
    """Longest contiguous run of True values in a 1D boolean array."""
    if mask_1d is None or mask_1d.size == 0:
        return 0
    m = mask_1d.astype(np.uint8)
    padded = np.pad(m, (1, 1), mode='constant')
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def compute_gap_metrics(seam_patch: np.ndarray, band_stats: dict | None):
    """Return (dark_ratio, max_run, thresh, (y0,y1)). Robust to empty patches."""
    if seam_patch is None or seam_patch.size == 0:
        return None, None, None, None

    h, w = seam_patch.shape[:2]
    y0 = int(np.clip(GAP_BAND_Y0_FRAC * h, 0, h - 1))
    y1 = int(np.clip(GAP_BAND_Y1_FRAC * h, y0 + 1, h))
    band = seam_patch[y0:y1, :]

    if band is None or band.size == 0:
        return None, None, None, (y0, y1)

    # Determine threshold
    if band_stats and (band_stats.get('median') is not None) and (band_stats.get('mad') is not None):
        med = float(band_stats['median'])
        mad = float(band_stats['mad'])
        thresh = med - THRESH_K * max(mad, 1.0)
    else:
        thresh = float(np.percentile(band, 15))

    dark = (band < thresh)
    dark_ratio = float(dark.mean())

    col_dark_ratio = dark.mean(axis=0)
    col_mask = col_dark_ratio > 0.45
    max_run = longest_run_1d(col_mask)

    return dark_ratio, max_run, float(thresh), (y0, y1)


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
    # We still request Mono8; convert-to-gray guard handles unexpected output.
    conv.OutputPixelFormat = pylon.PixelType_Mono8
    conv.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return cam, conv


def load_config():
    if not (os.path.exists(CONFIG_PATH) and os.path.exists(TEMPLATE_PATH)):
        return None
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print("[CFG] Saved", CONFIG_PATH)


def main():
    print("NOTE: close Basler pylon Viewer before running (exclusive camera access).")

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

    if warp_w <= 1 or warp_h <= 1:
        raise RuntimeError(f"Invalid warp_size in vision_config.json: w={warp_w}, h={warp_h}")

    band_stats = cfg.get("baseline_gap_band_stats", None)

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

    # baseline capture state
    cap_active = False
    cap_start_t = 0.0
    cap_next_sample_t = 0.0
    cap_interval = max(0.05, BASELINE_CAPTURE_DURATION_SEC / max(1, BASELINE_CAPTURE_SAMPLES))
    cap_bands = []

    # ORB tracking
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

            # key handling
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

            if k in (ord('u'), ord('U')):
                band_stats = None
                cfg["baseline_gap_band_stats"] = None
                save_config(cfg)
                bad = False
                part_failed = False
                dark_hist.clear(); run_hist.clear()

            # ORB homography
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

            # state machine
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

            # seam patch + metrics
            seam_patch = None
            seam_quad_frame = None
            dark_ratio = None
            max_run = None
            thresh = None
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

            now = time.monotonic()

            # baseline capture start
            if k in (ord('b'), ord('B')) and (not cap_active):
                cap_active = True
                cap_start_t = now
                cap_interval = max(0.05, float(BASELINE_CAPTURE_DURATION_SEC) / max(1, int(BASELINE_CAPTURE_SAMPLES)))
                cap_next_sample_t = now
                cap_bands = []
                bad = False
                part_failed = False
                dark_hist.clear(); run_hist.clear()

            # baseline capture collection
            if cap_active:
                elapsed = now - cap_start_t
                if (seam_patch is not None) and (now >= cap_next_sample_t) and (len(cap_bands) < BASELINE_CAPTURE_SAMPLES):
                    h, w = seam_patch.shape[:2]
                    y0 = int(np.clip(GAP_BAND_Y0_FRAC * h, 0, h - 1))
                    y1 = int(np.clip(GAP_BAND_Y1_FRAC * h, y0 + 1, h))
                    band = seam_patch[y0:y1, :].astype(np.float32)
                    if band.size > 0:
                        cap_bands.append(band)
                    cap_next_sample_t = now + cap_interval

                if (len(cap_bands) >= BASELINE_CAPTURE_SAMPLES) or (elapsed >= BASELINE_CAPTURE_DURATION_SEC):
                    cap_active = False
                    if len(cap_bands) >= BASELINE_CAPTURE_MIN_SAMPLES:
                        all_pix = np.concatenate([b.reshape(-1) for b in cap_bands], axis=0)
                        med = float(np.median(all_pix))
                        mad = float(np.median(np.abs(all_pix - med)))
                        band_stats = {
                            "median": med,
                            "mad": mad,
                            "y0_frac": GAP_BAND_Y0_FRAC,
                            "y1_frac": GAP_BAND_Y1_FRAC,
                            "thresh_k": THRESH_K,
                        }
                        cfg["baseline_gap_band_stats"] = band_stats
                        save_config(cfg)
                        print(f"[BASELINE] Saved gap band stats: median={med:.1f} mad={mad:.1f}")
                    else:
                        print(f"[BASELINE] Not enough samples: {len(cap_bands)}")
                    cap_bands = []

            # compute gap metrics
            if (not cap_active) and (not unstable) and seam_patch is not None and (band_stats is not None):
                dark_ratio, max_run, thresh, band_y = compute_gap_metrics(seam_patch, band_stats)
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

            # main panel UI
            bg = np.zeros((650, 950, 3), dtype=np.uint8)
            txt = (0, 0, 0)

            if cap_active or (band_stats is None):
                bg[:] = (0, 165, 255)
                if cap_active:
                    rem_s = max(0.0, BASELINE_CAPTURE_DURATION_SEC - (now - cap_start_t))
                    rem_n = max(0, BASELINE_CAPTURE_SAMPLES - len(cap_bands))
                    label = f"CAPTURING BASELINE... ({rem_n} samples, {rem_s:.0f}s left)"
                else:
                    label = "SET BASELINE (press b)"
            else:
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

            if band_stats is not None:
                med = band_stats.get('median', None)
                mad = band_stats.get('mad', None)
                cv2.putText(bg, f"GapBand med={med:.1f} mad={mad:.1f} K={THRESH_K}", (30, 225), FONT, 0.8, txt, 2)

            cv2.putText(bg, f"dark_ratio={None if dark_ratio is None else f'{dark_ratio:.3f}'} (fail>={DARK_RATIO_FAIL})", (30, 260), FONT, 0.85, txt, 2)
            cv2.putText(bg, f"max_run={None if max_run is None else f'{max_run}'} px (fail>={MAX_RUN_FAIL})", (30, 295), FONT, 0.85, txt, 2)
            cv2.putText(bg, f"thr={None if thresh is None else f'{thresh:.0f}'}  inliers={last_orb.get('inliers',0)} ratio={last_orb.get('inlier_ratio',0):.2f}", (30, 335), FONT, 0.75, txt, 2)
            cv2.putText(bg, f"PerPartLatch={per_part_latch}", (30, 370), FONT, 0.75, txt, 2)
            cv2.putText(bg, "Keys: b=Baseline  u=Clear  r=Reset  m=PerPart  d=Debug  p=Pause  ESC=Quit", (30, 620), FONT, 0.7, txt, 2)

            cv2.imshow(WINDOW_MAIN, bg)

            # debug view
            if show_debug:
                dbg = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if seam_quad_frame is not None:
                    cv2.polylines(dbg, [np.int32(seam_quad_frame).reshape(-1, 1, 2)], True, (0, 255, 255), 2)

                if seam_patch is not None:
                    patch_vis = cv2.cvtColor(seam_patch, cv2.COLOR_GRAY2BGR)
                    if band_y is not None:
                        y0, y1 = band_y
                        cv2.rectangle(patch_vis, (0, y0), (warp_w - 1, y1), (0, 255, 255), 2)
                        if thresh is not None:
                            cv2.putText(patch_vis, f"thr={thresh:.0f}", (5, max(15, y0+18)), FONT, 0.6, (0, 255, 255), 2)
                    scale = 1.6
                    patch_vis = cv2.resize(patch_vis, (int(warp_w * scale), int(warp_h * scale)))
                    ph, pw = patch_vis.shape[:2]
                    dbg[0:ph, 0:pw] = patch_vis
                    cv2.rectangle(dbg, (0, 0), (pw, ph), (255, 255, 255), 1)

                cv2.imshow(WINDOW_DEBUG, dbg)

    except Exception as e:
        print("\nERROR:", e)
        print(traceback.format_exc())
        print("If you see 'Device is exclusively opened by another client', close pylon Viewer and retry.")

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
