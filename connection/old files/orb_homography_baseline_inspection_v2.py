#!/usr/bin/env python3
"""orb_homography_baseline_inspection_v2.py

Patch notes vs v1
-----------------
Fixes the symptom you're seeing: BoundaryMed=None and "jump_rejected" even when ORB is perfect.

Changes:
1) Enforces a consistent quad point order (TL,TR,BR,BL) *after* homography transform each frame.
   This prevents the seam warp from occasionally flipping/rotating and confusing boundary tracking.
2) When you capture baseline (b), it clears prev_boundary and boundary history so the next
   measurements don't get rejected as a "jump".
3) Shows boundary debug metrics and allows a slightly larger jump threshold during bring-up.

Everything else is the same: ORB+Homography for pose, warp seam patch, baseline-boundary gap.

Inputs
------
- template.png
- vision_config.json from calibrate_rois_orb.py (seam_quad_in_template + warp_size)

Keys
----
b baseline, u clear baseline, l latch, r reset, p pause, d debug, c exit, ESC quit
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

# Pass / fail
GAP_FAIL_AT_OR_ABOVE_PX = 4
FAIL_CONSEC_TO_TRIP = 2
PASS_CONSEC_TO_CLEAR = 3

LATCH_MODE_DEFAULT = False
ENABLE_PER_PART_FAIL_LATCH = True

# Part present based on ORB confidence
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# Boundary detection params
COLMEAN_SMOOTH_K = 11
GRAD_SMOOTH_K = 9
BOUNDARY_SEARCH_FRAC_PREBASELINE = (0.10, 0.90)
BASELINE_MARGIN_LEFT = 5
BASELINE_MARGIN_RIGHT = 40
BOUNDARY_CONF_MIN = 1.0
BOUNDARY_HISTORY = 9
JITTER_PX_MAX = 3
STABLE_REQUIRED = 6
# Bring-up: allow bigger jumps; once stable you can return to 10
JUMP_REJECT_PX = 25
FAIL_IF_BOUNDARY_MISSING = False

# ORB / Homography
ORB_NFEATURES = 2000
ORB_KEEP_BEST = 100
RANSAC_REPROJ_THRESH = 4.0
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.35
HOLD_LAST_GOOD_FRAMES = 10

# UI
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


def detect_boundary_x(seam_roi_gray, baseline=None):
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
    else:
        a = int(clamp(baseline - BASELINE_MARGIN_LEFT, 0, w - 2))
        b = int(clamp(baseline + BASELINE_MARGIN_RIGHT, a + 2, w))

    search = grad_s[a:b]
    if search.size < 5:
        return None, {"reason": "search_window_too_small", "a": a, "b": b}

    idx = int(np.argmax(search)) + a
    peak = float(grad_s[idx])
    med = float(np.median(search))
    conf = peak - med
    if conf < BOUNDARY_CONF_MIN:
        return None, {"reason": "low_conf", "peak": peak, "med": med, "idx": idx, "a": a, "b": b}

    return idx, {"peak": peak, "med": med, "idx": idx, "a": a, "b": b, "conf": conf}


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
    """Ensure consistent TL,TR,BR,BL order for 4 points."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)           # x+y
    d = np.diff(pts, axis=1)      # y-x
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


def run(camera, converter, cfg):
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError("template.png missing. Run calibrate_rois_orb.py first.")

    seam_quad_t = cfg.get("seam_quad_in_template", None)
    if seam_quad_t is None or len(seam_quad_t) != 4:
        raise ValueError("vision_config.json missing seam_quad_in_template (need 4 points).")

    warp = cfg.get("warp_size", {"w": 420, "h": 200})
    warp_w = int(warp.get("w", 420))
    warp_h = int(warp.get("h", 200))

    baseline = cfg.get("baseline_boundary_x", None)
    baseline = int(baseline) if baseline is not None else None

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, 950, 650)
    cv2.moveWindow(WINDOW_MAIN, 20, 20)

    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_DEBUG, 1400, 900)
    cv2.moveWindow(WINDOW_DEBUG, 1000, 20)

    latch_mode = LATCH_MODE_DEFAULT
    paused = False
    show_debug = True

    state = "WAIT_FOR_PART"
    present_ct = 0
    leave_ct = 0

    bad = False
    part_failed = False
    fail_consec = 0
    pass_consec = 0
    max_gap_stable = 0
    saved_this_part = False

    boundary_hist = deque(maxlen=BOUNDARY_HISTORY)
    prev_boundary = None
    stable_count = 0

    last_good_H = None
    hold_frames_left = 0
    last_orb_dbg = {}

    last_boundary_dbg = {}

    print("\n--- RUN MODE (v2) ---")
    print("Keys: b=Baseline  u=ClearBaseline  l=Latch  r=Reset  p=Pause  d=Debug  c=Exit  ESC=Quit")

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue

        frame = converter.Convert(grab).GetArray()
        grab.Release()
        H_img, W_img = frame.shape

        k = cv2.waitKey(1) & 0xFF

        if k == KEY_ESC:
            return "QUIT"
        if k in (ord('c'), ord('C')):
            return "RECAL"

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
            print("LATCH_MODE =", latch_mode)

        if k in (ord('r'), ord('R')):
            bad = False
            part_failed = False
            fail_consec = pass_consec = 0
            max_gap_stable = 0
            saved_this_part = False
            boundary_hist.clear()
            prev_boundary = None
            stable_count = 0
            print("[RESET] cleared state for current part")

        if k in (ord('u'), ord('U')):
            baseline = None
            cfg["baseline_boundary_x"] = None
            save_config(cfg)
            boundary_hist.clear()
            prev_boundary = None
            stable_count = 0
            print("[BASELINE] cleared")

        # ORB homography
        H_est, orb_dbg = estimate_homography_orb(template, frame)
        last_orb_dbg = orb_dbg

        use_H = None
        if H_est is not None:
            inliers = int(orb_dbg.get("inliers", 0))
            ratio = float(orb_dbg.get("inlier_ratio", 0.0))
            if inliers >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO:
                use_H = H_est
                last_good_H = H_est
                hold_frames_left = HOLD_LAST_GOOD_FRAMES

        if use_H is None:
            if last_good_H is not None and hold_frames_left > 0:
                use_H = last_good_H
                hold_frames_left -= 1

        part_present = (use_H is not None)

        # State machine
        if not paused:
            if state == "WAIT_FOR_PART":
                present_ct = present_ct + 1 if part_present else 0
                if present_ct >= ENTER_PRESENT_FRAMES:
                    state = "INSPECT"
                    present_ct = leave_ct = 0
                    bad = False if not latch_mode else bad
                    part_failed = False
                    fail_consec = pass_consec = 0
                    max_gap_stable = 0
                    saved_this_part = False
                    boundary_hist.clear()
                    prev_boundary = None
                    stable_count = 0
            else:
                leave_ct = leave_ct + 1 if not part_present else 0
                if leave_ct >= LEAVE_FRAMES:
                    state = "WAIT_FOR_PART"
                    leave_ct = present_ct = 0
                    if not latch_mode:
                        bad = False

        seam_patch = None
        seam_quad_frame = None
        boundary_x = None
        boundary_med = None
        gap_px = None
        dbg_note = ""

        if state == "INSPECT" and use_H is not None:
            seam_quad_t_pts = [(float(p[0]), float(p[1])) for p in seam_quad_t]
            seam_quad_frame = perspective_transform_points(use_H, seam_quad_t_pts)
            seam_quad_frame = order_quad_tltrbrbl(seam_quad_frame)

            try:
                seam_patch = warp_quad_to_rect(frame, seam_quad_frame, warp_w, warp_h)
            except Exception:
                seam_patch = None
                dbg_note = "warp_failed"

            if seam_patch is not None:
                boundary_x, bdbg = detect_boundary_x(seam_patch, baseline=baseline)
                last_boundary_dbg = bdbg

                if boundary_x is not None and prev_boundary is not None and abs(boundary_x - prev_boundary) > JUMP_REJECT_PX:
                    boundary_x = None
                    dbg_note = "jump_rejected"

                if boundary_x is not None:
                    prev_boundary = boundary_x
                    boundary_hist.append(boundary_x)
                    boundary_med = int(np.median(boundary_hist))
                else:
                    if not dbg_note:
                        dbg_note = bdbg.get("reason", "")

                if len(boundary_hist) >= 5:
                    jitter = max(boundary_hist) - min(boundary_hist)
                    stable_count = stable_count + 1 if jitter <= JITTER_PX_MAX else 0
                else:
                    stable_count = 0

                if baseline is not None and boundary_med is not None:
                    gap_px = int(max(0, boundary_med - baseline))

        # baseline capture (clears history to prevent jump issues)
        if k in (ord('b'), ord('B')):
            if seam_patch is None:
                print("[BASELINE] seam patch not available; cannot capture.")
            elif boundary_med is None:
                print("[BASELINE] boundary not found; cannot capture.")
            else:
                baseline = int(boundary_med)
                cfg["baseline_boundary_x"] = int(baseline)
                save_config(cfg)
                # Clear boundary tracking so next frames don't get rejected as a jump
                boundary_hist.clear()
                prev_boundary = None
                stable_count = 0
                print("[BASELINE] captured =", baseline)

        # Decision
        if state == "INSPECT" and not paused and baseline is not None:
            if gap_px is None:
                if FAIL_IF_BOUNDARY_MISSING:
                    bad = True
            else:
                is_fail = (gap_px >= GAP_FAIL_AT_OR_ABOVE_PX)
                if is_fail:
                    fail_consec += 1
                    pass_consec = 0
                else:
                    pass_consec += 1
                    fail_consec = 0

                if fail_consec >= FAIL_CONSEC_TO_TRIP:
                    bad = True

                if pass_consec >= PASS_CONSEC_TO_CLEAR and (not latch_mode) and (not ENABLE_PER_PART_FAIL_LATCH):
                    bad = False

                if stable_count >= STABLE_REQUIRED:
                    max_gap_stable = max(max_gap_stable, gap_px)

                if ENABLE_PER_PART_FAIL_LATCH and max_gap_stable >= GAP_FAIL_AT_OR_ABOVE_PX:
                    part_failed = True

                if part_failed:
                    bad = True

        if state == "INSPECT" and bad and not saved_this_part:
            ann = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if seam_quad_frame is not None:
                pts = np.int32(seam_quad_frame).reshape(-1, 1, 2)
                cv2.polylines(ann, [pts], True, (0, 255, 255), 2)
            info = (
                f"baseline={baseline}, boundary_med={boundary_med}, gap={gap_px}, "
                f"max_gap_stable={max_gap_stable}, stable_count={stable_count}, "
                f"orb={last_orb_dbg}, boundary_dbg={last_boundary_dbg}"
            )
            save_fail_images(frame, ann, info_text=info)
            saved_this_part = True

        # MAIN panel
        bg = np.zeros((650, 950, 3), dtype=np.uint8)
        if baseline is None:
            bg[:] = (0, 165, 255)
            label = "SET BASELINE (press b)"
            txt = (0, 0, 0)
        else:
            if bad:
                bg[:] = (0, 0, 255)
                label = "BAD"
                txt = (255, 255, 255)
            else:
                bg[:] = (0, 255, 0)
                label = "GOOD"
                txt = (0, 0, 0)

        cv2.putText(bg, label, (30, 120), FONT, 1.8, txt, 6)
        cv2.putText(bg, f"State: {state}", (30, 200), FONT, 1.0, txt, 2)
        cv2.putText(bg, f"BaselineX: {baseline}", (30, 240), FONT, 0.9, txt, 2)
        cv2.putText(bg, f"BoundaryMed: {boundary_med}   Gap: {gap_px}", (30, 280), FONT, 0.9, txt, 2)
        cv2.putText(bg, f"MaxGap(stable): {max_gap_stable} (Fail>= {GAP_FAIL_AT_OR_ABOVE_PX})", (30, 320), FONT, 0.85, txt, 2)
        cv2.putText(bg, f"StableCount: {stable_count}/{STABLE_REQUIRED}  PartFailed: {part_failed}", (30, 360), FONT, 0.85, txt, 2)

        inliers = last_orb_dbg.get("inliers", 0)
        ratio = last_orb_dbg.get("inlier_ratio", 0.0)
        matches = last_orb_dbg.get("matches", 0)
        reason = last_orb_dbg.get("reason", "")
        cv2.putText(bg, f"ORB matches={matches} inliers={inliers} ratio={ratio:.2f} hold={hold_frames_left}",
                    (30, 410), FONT, 0.75, txt, 2)
        if reason:
            cv2.putText(bg, f"ORB note: {reason}", (30, 445), FONT, 0.75, txt, 2)
        if dbg_note:
            cv2.putText(bg, f"Boundary note: {dbg_note}", (30, 480), FONT, 0.75, txt, 2)

        cv2.imshow(WINDOW_MAIN, bg)

        # DEBUG
        if show_debug:
            dbg_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if seam_quad_frame is not None:
                pts = np.int32(seam_quad_frame).reshape(-1, 1, 2)
                cv2.polylines(dbg_img, [pts], True, (0, 255, 255), 2)

            if seam_patch is not None:
                patch_vis = cv2.cvtColor(seam_patch, cv2.COLOR_GRAY2BGR)
                if boundary_med is not None:
                    cv2.line(patch_vis, (int(boundary_med), 0), (int(boundary_med), warp_h - 1), (255, 255, 0), 2)
                if baseline is not None:
                    cv2.line(patch_vis, (int(baseline), 0), (int(baseline), warp_h - 1), (255, 0, 255), 1)
                ph, pw = patch_vis.shape[:2]
                maxw = 420
                if pw > maxw:
                    scale = maxw / pw
                    patch_vis = cv2.resize(patch_vis, (int(pw * scale), int(ph * scale)))
                ph, pw = patch_vis.shape[:2]
                dbg_img[0:ph, 0:pw] = patch_vis
                cv2.rectangle(dbg_img, (0, 0), (pw, ph), (255, 255, 255), 1)

            cv2.putText(dbg_img, f"State={state} Present={part_present} ORB inliers={inliers} ratio={ratio:.2f}",
                        (30, H_img - 20), FONT, 0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW_DEBUG, dbg_img)

    return "QUIT"


def main():
    print("NOTE: close Basler pylon Viewer before running (exclusive camera access).")
    cfg = load_config()
    if cfg is None:
        print("Missing vision_config.json or template.png. Run calibrate_rois_orb.py first.")
        return

    camera = None
    try:
        camera, converter = start_camera()
        result = run(camera, converter, cfg)
        if result == "RECAL":
            print("Exit requested. Re-run calibration if needed.")
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
