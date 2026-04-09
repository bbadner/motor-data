#!/usr/bin/env python3
"""orb_homography_baseline_inspection_v9.py

v9: Stop the boundary from hopping to a different edge (fix seated gap bouncing up to ~17)
---------------------------------------------------------------------------------------
Your symptom: after u then b, it briefly shows GOOD then quickly goes BAD because the gap
estimate occasionally jumps to ~17 even though the connector is seated.

What this usually means
-----------------------
The boundary detector is sometimes picking a *different strong edge* inside the seam patch
(e.g., connector rib/cable edge/reflection) within the allowed search window.

Key change in v9
----------------
After baseline is set, we no longer pick the single strongest edge in the whole search window.
Instead, we use *tracking*: pick the best edge near the previous boundary (or baseline if
no previous), and only fall back to the global best if the local edge confidence is too low.
This dramatically reduces edge hopping.

We keep:
- ORB + RANSAC homography for pose invariance
- seam quad ordering + warp to normalized patch
- baseline capture session (median over N frames)
- median filtered gap (GapMed)
- soft vs hard fail logic

Keys
----
  b : baseline capture session (collect ~BASELINE_CAPTURE_FRAMES)
  u : clear baseline + reset
  r : reset current part state
  m : toggle PerPartLatch
  k : toggle ClearOnReseat
  l : toggle LatchMode (global latch)
  p : pause
  d : toggle debug
  c : exit
  ESC : quit
"""

import os
import json
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from pypylon import pylon

print("RUNNING FILE:", os.path.abspath(__file__))

CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"
FAIL_DIR = "fails"

# ----------------------------
# Pass / fail thresholds
# ----------------------------
GAP_FAIL_AT_OR_ABOVE_PX = 4
FAIL_CONSEC_TO_TRIP = 3
PASS_CONSEC_TO_CLEAR = 6

# HARD FAIL: obvious pull-out
HARD_FAIL_AT_OR_ABOVE_PX = 12
HARD_FAIL_CONSEC_TO_TRIP = 1

# Median filtering for decisions
GAP_HISTORY = 7

# Gap-stability gating for SOFT decisions
GAP_JITTER_MAX = 2
STABLE_GAP_REQUIRED_FOR_DECISION = 2

# ----------------------------
# Latch behavior
# ----------------------------
LATCH_MODE_DEFAULT = False
PER_PART_LATCH_DEFAULT = True
CLEAR_ON_RESEAT_DEFAULT = False

# ----------------------------
# Part present based on ORB confidence
# ----------------------------
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# ----------------------------
# Boundary detection
# ----------------------------
COLMEAN_SMOOTH_K = 11
GRAD_SMOOTH_K = 9
BOUNDARY_SEARCH_FRAC_PREBASELINE = (0.10, 0.90)
BASELINE_MARGIN_LEFT = 5
BASELINE_MARGIN_RIGHT = 60

# Confidence threshold for accepting an edge
BOUNDARY_CONF_MIN = 1.0

# Tracking parameters (NEW)
TRACK_NEAR_PX = 10              # search +/- this around prev/baseline
TRACK_MIN_CONF = 0.8            # if best near-edge conf >= this, use it; else fallback

# Boundary stability for max-gap tracking
BOUNDARY_HISTORY = 9
MIN_HISTORY_FOR_JITTER = 3
BOUNDARY_JITTER_PX_MAX = 6
STABLE_REQUIRED_FOR_MAXGAP = 3
JUMP_REJECT_PX = 40             # larger because tracking handles stability
FAIL_IF_BOUNDARY_MISSING = False

# ----------------------------
# Baseline capture session
# ----------------------------
BASELINE_CAPTURE_FRAMES = 18
BASELINE_CAPTURE_MIN_SAMPLES = 10

# ----------------------------
# ORB / Homography
# ----------------------------
ORB_NFEATURES = 2000
ORB_KEEP_BEST = 100
RANSAC_REPROJ_THRESH = 4.0
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.35
HOLD_LAST_GOOD_FRAMES = 10

# ----------------------------
# UI
# ----------------------------
WINDOW_MAIN = "Trial Inspection"
WINDOW_DEBUG = "DEBUG VIEW"
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_ESC = 27


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def smooth_1d(x, k):
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")


def save_fail_images(frame_gray, annotated_bgr, info_text=""):
    ensure_dir(FAIL_DIR)
    ts = stamp()
    raw_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_raw.png")
    ann_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_annotated.png")
    txt_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_info.txt")
    cv2.imwrite(raw_path, frame_gray)
    cv2.imwrite(ann_path, annotated_bgr)
    if info_text:
        with open(txt_path, "w") as f:
            f.write(info_text)
    print("[SAVE]", raw_path)
    print("[SAVE]", ann_path)
    if info_text:
        print("[SAVE]", txt_path)


def boundary_from_grad(grad_s, a, b, target=None):
    """Pick boundary index using tracking.

    grad_s: 1D gradient signal
    a,b: search window
    target: preferred x (prev boundary or baseline)

    Returns: idx or None, dbg dict
    """
    search = grad_s[a:b]
    if search.size < 5:
        return None, {"reason": "search_window_too_small", "a": a, "b": b}

    # global best
    g_rel = int(np.argmax(search))
    g_idx = g_rel + a
    g_peak = float(grad_s[g_idx])
    g_med = float(np.median(search))
    g_conf = g_peak - g_med

    if target is None:
        if g_conf < BOUNDARY_CONF_MIN:
            return None, {"reason": "low_conf", "conf": g_conf, "idx": g_idx, "a": a, "b": b}
        return g_idx, {"reason": "global", "conf": g_conf, "idx": g_idx, "a": a, "b": b}

    # near window around target
    t0 = int(clamp(target - TRACK_NEAR_PX, a, b-1))
    t1 = int(clamp(target + TRACK_NEAR_PX + 1, a+1, b))
    near = grad_s[t0:t1]
    if near.size >= 3:
        n_rel = int(np.argmax(near))
        n_idx = n_rel + t0
        n_peak = float(grad_s[n_idx])
        n_med = float(np.median(search))  # compare to same window median
        n_conf = n_peak - n_med
    else:
        n_idx = None
        n_conf = -1e9

    # prefer near edge if confident enough; else fallback to global
    if n_idx is not None and n_conf >= TRACK_MIN_CONF and n_conf >= BOUNDARY_CONF_MIN:
        return n_idx, {"reason": "near", "conf": n_conf, "idx": n_idx, "a": a, "b": b, "t0": t0, "t1": t1}

    if g_conf >= BOUNDARY_CONF_MIN:
        return g_idx, {"reason": "fallback_global", "conf": g_conf, "idx": g_idx, "a": a, "b": b, "t0": t0, "t1": t1, "near_conf": n_conf}

    return None, {"reason": "low_conf", "conf": g_conf, "idx": g_idx, "a": a, "b": b, "t0": t0, "t1": t1, "near_conf": n_conf}


def detect_boundary_x(seam_roi_gray, baseline=None, prev=None):
    """Detect boundary x with optional baseline and previous boundary tracking."""
    roi = cv2.GaussianBlur(seam_roi_gray, (5, 5), 0)
    col_mean = roi.mean(axis=0).astype(np.float32)
    col_mean_s = smooth_1d(col_mean, COLMEAN_SMOOTH_K)
    grad = -(np.diff(col_mean_s, prepend=col_mean_s[0]))
    grad_s = smooth_1d(grad, GRAD_SMOOTH_K)

    w = len(col_mean_s)
    if w < 10:
        return None, {"reason": "roi_too_small"}

    if baseline is None:
        a = int(BOUNDARY_SEARCH_FRAC_PREBASELINE[0] * w)
        b = int(BOUNDARY_SEARCH_FRAC_PREBASELINE[1] * w)
        a = clamp(a, 0, w - 2)
        b = clamp(b, a + 2, w)
        target = None
    else:
        a = int(clamp(baseline - BASELINE_MARGIN_LEFT, 0, w - 2))
        b = int(clamp(baseline + BASELINE_MARGIN_RIGHT, a + 2, w))
        target = prev if prev is not None else baseline

    idx, dbg = boundary_from_grad(grad_s, a, b, target=target)
    return idx, dbg


def estimate_homography_orb(template_gray, frame_gray,
                            orb_nfeatures=ORB_NFEATURES,
                            keep_best=ORB_KEEP_BEST,
                            ransac_thresh=RANSAC_REPROJ_THRESH):
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(frame_gray, None)

    if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
        return None, {"reason": "not_enough_features", "kp1": len(kp1) if kp1 is not None else 0, "kp2": len(kp2) if kp2 is not None else 0}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return None, {"reason": "no_matches"}

    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:min(len(matches), keep_best)]

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if H is None or inliers is None:
        return None, {"reason": "homography_failed", "matches": len(matches)}

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / max(1, len(matches))
    return H, {"matches": len(matches), "inliers": inlier_count, "inlier_ratio": float(inlier_ratio)}


def perspective_transform_points(H, pts):
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
    return [(float(tl[0]), float(tl[1])),
            (float(tr[0]), float(tr[1])),
            (float(br[0]), float(br[1])),
            (float(bl[0]), float(bl[1]))]


def warp_quad_to_rect(gray, quad_pts, out_w, out_h):
    src = np.float32(quad_pts)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    roi = cv2.warpPerspective(gray, M, (out_w, out_h))
    return roi


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


def load_config():
    if not (os.path.exists(CONFIG_PATH) and os.path.exists(TEMPLATE_PATH)):
        return None
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print("[CFG] Saved", CONFIG_PATH)


def main():
    print("NOTE: close Basler pylon Viewer before running (exclusive camera access).")
    cfg = load_config()
    if cfg is None:
        print("Missing vision_config.json or template.png. Run calibrate_rois_orb.py first.")
        return

    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("template.png missing.")
        return

    seam_quad_t = cfg.get("seam_quad_in_template", None)
    if seam_quad_t is None or len(seam_quad_t) != 4:
        print("vision_config.json missing seam_quad_in_template.")
        return

    warp = cfg.get("warp_size", {"w": 420, "h": 200})
    warp_w = int(warp.get("w", 420))
    warp_h = int(warp.get("h", 200))

    baseline = cfg.get("baseline_boundary_x", None)
    baseline = int(baseline) if baseline is not None else None

    th, tw = template.shape[:2]
    template_corners_t = [(0.0, 0.0), (float(tw), 0.0), (float(tw), float(th)), (0.0, float(th))]

    camera = None
    try:
        camera, converter = start_camera()

        cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_MAIN, 950, 650)
        cv2.moveWindow(WINDOW_MAIN, 20, 20)

        cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_DEBUG, 1400, 900)
        cv2.moveWindow(WINDOW_DEBUG, 1000, 20)

        latch_mode = LATCH_MODE_DEFAULT
        per_part_latch = PER_PART_LATCH_DEFAULT
        clear_on_reseat = CLEAR_ON_RESEAT_DEFAULT

        paused = False
        show_debug = True

        state = "WAIT_FOR_PART"
        present_ct = 0
        leave_ct = 0

        bad = False
        part_failed = False
        fail_consec = 0
        pass_consec = 0
        hard_fail_consec = 0

        max_gap_stable = 0
        saved_this_part = False

        boundary_hist = deque(maxlen=BOUNDARY_HISTORY)
        prev_boundary = None
        stable_boundary_count = 0

        gap_hist = deque(maxlen=GAP_HISTORY)
        stable_gap_count = 0

        last_good_H = None
        hold_frames_left = 0
        last_orb_dbg = {}
        last_bdbg = {}

        baseline_capture_active = False
        baseline_capture_left = 0
        baseline_samples = []

        toast_text = ""
        toast_until = 0

        def toast(msg, ms=2200):
            nonlocal toast_text, toast_until
            toast_text = msg
            toast_until = cv2.getTickCount() + int(ms * (cv2.getTickFrequency() / 1000.0))
            print("[UI]", msg)

        def reset_part_state(reason=""):
            nonlocal bad, part_failed, fail_consec, pass_consec, hard_fail_consec
            nonlocal max_gap_stable, saved_this_part
            nonlocal boundary_hist, prev_boundary, stable_boundary_count
            nonlocal gap_hist, stable_gap_count
            nonlocal baseline_capture_active, baseline_capture_left, baseline_samples
            bad = False
            part_failed = False
            fail_consec = 0
            pass_consec = 0
            hard_fail_consec = 0
            max_gap_stable = 0
            saved_this_part = False
            boundary_hist.clear()
            prev_boundary = None
            stable_boundary_count = 0
            gap_hist.clear()
            stable_gap_count = 0
            baseline_capture_active = False
            baseline_capture_left = 0
            baseline_samples.clear()
            if reason:
                toast(reason)

        toast("Press u to clear baseline, then b to capture baseline.")

        while camera.IsGrabbing():
            grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab.GrabSucceeded():
                grab.Release()
                continue
            frame = converter.Convert(grab).GetArray()
            grab.Release()

            k = cv2.waitKey(1) & 0xFF

            if k == KEY_ESC:
                break
            if k in (ord('c'), ord('C')):
                break
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
            if k in (ord('l'), ord('L')):
                latch_mode = not latch_mode
                toast(f"LatchMode={latch_mode}")
            if k in (ord('m'), ord('M')):
                per_part_latch = not per_part_latch
                toast(f"PerPartLatch={per_part_latch}")
            if k in (ord('k'), ord('K')):
                clear_on_reseat = not clear_on_reseat
                toast(f"ClearOnReseat={clear_on_reseat}")
            if k in (ord('r'), ord('R')):
                reset_part_state("Reset")
            if k in (ord('u'), ord('U')):
                baseline = None
                cfg["baseline_boundary_x"] = None
                save_config(cfg)
                reset_part_state("Baseline cleared")

            # ORB homography
            H_est, orb_dbg = estimate_homography_orb(template, frame)
            last_orb_dbg = orb_dbg
            use_H = None
            if H_est is not None:
                if int(orb_dbg.get("inliers", 0)) >= MIN_INLIERS and float(orb_dbg.get("inlier_ratio", 0.0)) >= MIN_INLIER_RATIO:
                    use_H = H_est
                    last_good_H = H_est
                    hold_frames_left = HOLD_LAST_GOOD_FRAMES
            if use_H is None and last_good_H is not None and hold_frames_left > 0:
                use_H = last_good_H
                hold_frames_left -= 1

            part_present = (use_H is not None)

            if not paused:
                if state == "WAIT_FOR_PART":
                    present_ct = present_ct + 1 if part_present else 0
                    if present_ct >= ENTER_PRESENT_FRAMES:
                        state = "INSPECT"
                        present_ct = leave_ct = 0
                        reset_part_state("INSPECT")
                else:
                    leave_ct = leave_ct + 1 if not part_present else 0
                    if leave_ct >= LEAVE_FRAMES:
                        state = "WAIT_FOR_PART"
                        leave_ct = present_ct = 0
                        if not latch_mode:
                            reset_part_state("WAIT")

            seam_patch = None
            seam_quad_frame = None
            tpl_quad_frame = None
            boundary_x = None
            boundary_med = None
            gap_px = None
            gap_med = None
            dbg_note = ""

            if state == "INSPECT" and use_H is not None:
                seam_quad_t_pts = [(float(p[0]), float(p[1])) for p in seam_quad_t]
                seam_quad_frame = order_quad_tltrbrbl(perspective_transform_points(use_H, seam_quad_t_pts))
                tpl_quad_frame = order_quad_tltrbrbl(perspective_transform_points(use_H, template_corners_t))

                try:
                    seam_patch = warp_quad_to_rect(frame, seam_quad_frame, warp_w, warp_h)
                except Exception:
                    seam_patch = None
                    dbg_note = "warp_failed"

                if seam_patch is not None:
                    boundary_x, bdbg = detect_boundary_x(seam_patch, baseline=baseline, prev=prev_boundary)
                    last_bdbg = bdbg

                    if boundary_x is not None and prev_boundary is not None and abs(boundary_x - prev_boundary) > JUMP_REJECT_PX:
                        # Tracking should prevent this; still keep a guard
                        boundary_x = None
                        dbg_note = "jump_rejected"

                    if boundary_x is not None:
                        prev_boundary = boundary_x
                        boundary_hist.append(boundary_x)
                        boundary_med = int(np.median(boundary_hist))
                    else:
                        if not dbg_note:
                            dbg_note = bdbg.get("reason", "")

                    if len(boundary_hist) >= MIN_HISTORY_FOR_JITTER:
                        bj = max(boundary_hist) - min(boundary_hist)
                        stable_boundary_count = stable_boundary_count + 1 if bj <= BOUNDARY_JITTER_PX_MAX else 0
                    else:
                        stable_boundary_count = 0

                    if baseline_capture_active:
                        baseline_capture_left -= 1
                        if boundary_x is not None:
                            baseline_samples.append(int(boundary_x))
                        if baseline_capture_left <= 0:
                            baseline_capture_active = False
                            if len(baseline_samples) >= BASELINE_CAPTURE_MIN_SAMPLES:
                                baseline = int(np.median(baseline_samples))
                                cfg["baseline_boundary_x"] = int(baseline)
                                save_config(cfg)
                                reset_part_state(f"Baseline={baseline}")
                            else:
                                toast(f"Baseline failed: {len(baseline_samples)} samples")
                            baseline_samples.clear()

                    if baseline is not None and boundary_med is not None:
                        gap_px = int(max(0, boundary_med - baseline))
                        gap_hist.append(gap_px)
                        gap_med = int(np.median(np.array(gap_hist, dtype=np.int32))) if len(gap_hist) else gap_px

                        if len(gap_hist) >= 3:
                            gj = max(gap_hist) - min(gap_hist)
                            stable_gap_count = stable_gap_count + 1 if gj <= GAP_JITTER_MAX else 0
                        else:
                            stable_gap_count = 0

            if k in (ord('b'), ord('B')):
                if state != "INSPECT" or seam_patch is None:
                    toast("Cannot baseline now")
                else:
                    baseline_capture_active = True
                    baseline_capture_left = BASELINE_CAPTURE_FRAMES
                    baseline_samples.clear()
                    toast("Baseline capture...")

            # Decision
            if state == "INSPECT" and not paused and baseline is not None and gap_med is not None:
                if gap_med >= HARD_FAIL_AT_OR_ABOVE_PX:
                    hard_fail_consec += 1
                else:
                    hard_fail_consec = 0
                if hard_fail_consec >= HARD_FAIL_CONSEC_TO_TRIP:
                    bad = True

                if stable_gap_count >= STABLE_GAP_REQUIRED_FOR_DECISION:
                    is_fail = (gap_med >= GAP_FAIL_AT_OR_ABOVE_PX)
                    if is_fail:
                        fail_consec += 1
                        pass_consec = 0
                    else:
                        pass_consec += 1
                        fail_consec = 0
                    if fail_consec >= FAIL_CONSEC_TO_TRIP:
                        bad = True
                    if (not per_part_latch) and clear_on_reseat and (pass_consec >= PASS_CONSEC_TO_CLEAR) and (not latch_mode):
                        bad = False
                        part_failed = False

                if stable_boundary_count >= STABLE_REQUIRED_FOR_MAXGAP:
                    max_gap_stable = max(max_gap_stable, gap_med)
                if per_part_latch and max_gap_stable >= GAP_FAIL_AT_OR_ABOVE_PX:
                    part_failed = True
                if part_failed:
                    bad = True

            if state == "INSPECT" and bad and not saved_this_part:
                ann = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if seam_quad_frame is not None:
                    cv2.polylines(ann, [np.int32(seam_quad_frame).reshape(-1, 1, 2)], True, (0,255,255), 2)
                if tpl_quad_frame is not None:
                    cv2.polylines(ann, [np.int32(tpl_quad_frame).reshape(-1, 1, 2)], True, (255,0,0), 2)
                info = f"baseline={baseline}, boundary_med={boundary_med}, gap={gap_px}, gap_med={gap_med}, bdbg={last_bdbg}"
                save_fail_images(frame, ann, info_text=info)
                saved_this_part = True

            # MAIN panel
            bg = np.zeros((650, 950, 3), dtype=np.uint8)
            if baseline is None:
                bg[:] = (0,165,255)
                label = "SET BASELINE (press b)"
                txt = (0,0,0)
            else:
                if bad:
                    bg[:] = (0,0,255)
                    label = "BAD"
                    txt = (255,255,255)
                else:
                    bg[:] = (0,255,0)
                    label = "GOOD"
                    txt = (0,0,0)

            if toast_text and cv2.getTickCount() < toast_until:
                cv2.putText(bg, toast_text, (30, 60), FONT, 0.8, txt, 2)
            elif cv2.getTickCount() >= toast_until:
                toast_text = ""

            cv2.putText(bg, label, (30, 140), FONT, 1.8, txt, 6)
            cv2.putText(bg, f"State: {state}", (30, 220), FONT, 1.0, txt, 2)
            cv2.putText(bg, f"BaselineX: {baseline}", (30, 260), FONT, 0.9, txt, 2)
            cv2.putText(bg, f"BoundaryMed: {boundary_med}  Gap: {gap_px}  GapMed({GAP_HISTORY}): {gap_med}", (30, 300), FONT, 0.85, txt, 2)
            cv2.putText(bg, f"TrackNear={TRACK_NEAR_PX}px TrackMinConf={TRACK_MIN_CONF}", (30, 335), FONT, 0.75, txt, 2)
            cv2.putText(bg, f"bdbg: {last_bdbg.get('reason','')} conf={last_bdbg.get('conf','')} idx={last_bdbg.get('idx','')}", (30, 365), FONT, 0.7, txt, 2)
            cv2.putText(bg, f"HardFail>= {HARD_FAIL_AT_OR_ABOVE_PX} hard_ct={hard_fail_consec} | SoftFail>= {GAP_FAIL_AT_OR_ABOVE_PX} fail_ct={fail_consec}",
                        (30, 395), FONT, 0.7, txt, 2)
            cv2.putText(bg, f"GapStable: {stable_gap_count}/{STABLE_GAP_REQUIRED_FOR_DECISION} (jitter<= {GAP_JITTER_MAX})", (30, 425), FONT, 0.7, txt, 2)
            cv2.putText(bg, f"MaxGapStable: {max_gap_stable} PartFailed: {part_failed}", (30, 455), FONT, 0.7, txt, 2)
            cv2.putText(bg, f"ORB inliers={last_orb_dbg.get('inliers',0)} ratio={last_orb_dbg.get('inlier_ratio',0):.2f} hold={hold_frames_left}",
                        (30, 485), FONT, 0.7, txt, 2)
            cv2.putText(bg, f"LatchMode={latch_mode} PerPartLatch={per_part_latch} ClearOnReseat={clear_on_reseat}", (30, 515), FONT, 0.7, txt, 2)
            if dbg_note:
                cv2.putText(bg, f"Note: {dbg_note}", (30, 545), FONT, 0.7, txt, 2)

            cv2.putText(bg, "Keys: b=Baseline u=Clear r=Reset m=PerPart k=ClrReseat d=Debug ESC=Quit", (30, 620), FONT, 0.7, txt, 2)
            cv2.imshow(WINDOW_MAIN, bg)

            if show_debug:
                dbg_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if tpl_quad_frame is not None:
                    cv2.polylines(dbg_img, [np.int32(tpl_quad_frame).reshape(-1,1,2)], True, (255,0,0), 2)
                if seam_quad_frame is not None:
                    cv2.polylines(dbg_img, [np.int32(seam_quad_frame).reshape(-1,1,2)], True, (0,255,255), 2)
                if seam_patch is not None:
                    patch_vis = cv2.cvtColor(seam_patch, cv2.COLOR_GRAY2BGR)
                    if boundary_med is not None:
                        cv2.line(patch_vis, (int(boundary_med), 0), (int(boundary_med), warp_h-1), (255,255,0), 2)
                    if baseline is not None:
                        cv2.line(patch_vis, (int(baseline), 0), (int(baseline), warp_h-1), (255,0,255), 1)
                    scale = 1.6
                    patch_vis = cv2.resize(patch_vis, (int(warp_w*scale), int(warp_h*scale)))
                    ph,pw = patch_vis.shape[:2]
                    dbg_img[0:ph, 0:pw] = patch_vis
                    cv2.rectangle(dbg_img, (0,0), (pw,ph), (255,255,255), 1)
                cv2.imshow(WINDOW_DEBUG, dbg_img)

    except Exception as e:
        print("\nERROR:", e)
        print("If you see 'Device is exclusively opened by another client', close pylon Viewer and retry.\n")
    finally:
        if camera is not None:
            try:
                camera.StopGrabbing()
            except Exception:
                pass
            try:
                camera.Close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
