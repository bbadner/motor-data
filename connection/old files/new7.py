#!/usr/bin/env python3
"""orb_homography_inspection_v19_unstable_hysteresis_tuned2.py

Connector seating inspection using ORB homography + seam patch profile matching.

UI states
---------
- GOOD (green): stable pose + no defect indicators.
- UNSTABLE - REPOSITION (orange): pose mismatch / tracking lost / seam out of view.
  (UNSTABLE does NOT latch.)
- BAD (red): defect indicators (shift/residual/hard limits). BAD latches if PerPartLatch=True.

UNSTABLE hysteresis (tuned for your observed max ScoreMed ≈ 0.92)
---------------------------------------------------------------
- Enter UNSTABLE when ScoreMed <= 0.87
- Exit UNSTABLE only when ScoreMed >= 0.91

Rationale:
- Your best achievable seated ScoreMed is ~0.92. Exit must be comfortably below that
  so the system can return to GOOD.
- A 0.04 hysteresis band (0.87↔0.91) plus SCORE_HISTORY=9 reduces flicker.

Baseline capture
----------------
- Timeboxed baseline: 25 samples over 90 seconds, minimum 20 samples.
- No GOOD/BAD/UNSTABLE decisions while capturing baseline.

Files
-----
- template.png
- vision_config.json (must include seam_quad_in_template and warp_size)

Keys
----
b  capture baseline profile
u  clear baseline
r  reset BAD state (and clear per-part latch)
m  toggle per-part latch
k  toggle clear-on-reseat
l  toggle latch mode
p  pause
d  debug
ESC quit
"""

import os
import json
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from pypylon import pylon

CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"
FAIL_DIR = "fails"

# -------------------------- Decision thresholds --------------------------
FAIL_AT_OR_ABOVE_PX = 4
FAIL_CONSEC_TO_TRIP = 3
PASS_CONSEC_TO_CLEAR = 6
HARD_SHIFT_FAIL_AT_OR_ABOVE = 12

SHIFT_HISTORY = 7
SCORE_HISTORY = 9
RESID_HISTORY = 7

MAX_SHIFT_PX = 80

# -------------------------- UNSTABLE hysteresis (pose) --------------------------
SCORE_UNSTABLE_ENTER = 0.85
SCORE_UNSTABLE_EXIT  = 0.90

# -------------------------- Defect thresholds (true BAD) --------------------------
RESID_FAIL_AT_OR_ABOVE = 0.65
RESID_CONSEC_TO_TRIP = 3
HARD_RESID_FAIL_AT_OR_ABOVE = 1.10

# -------------------------- Latch behavior --------------------------
LATCH_MODE_DEFAULT = False
PER_PART_LATCH_DEFAULT = True
CLEAR_ON_RESEAT_DEFAULT = False

# -------------------------- Part present via ORB --------------------------
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# -------------------------- ORB / Homography --------------------------
ORB_NFEATURES = 2000
ORB_KEEP_BEST = 100
RANSAC_REPROJ_THRESH = 4.0
MIN_INLIERS = 15
MIN_INLIER_RATIO = 0.25
HOLD_LAST_GOOD_FRAMES = 10

# -------------------------- Profile construction --------------------------
PROFILE_SMOOTH_K = 11
PROFILE_GRAD_SMOOTH_K = 9

# -------------------------- Baseline capture (timeboxed) --------------------------
BASELINE_CAPTURE_SAMPLES = 25
BASELINE_CAPTURE_DURATION_SEC = 90
BASELINE_CAPTURE_MIN_SAMPLES = 20

# -------------------------- UI --------------------------
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


def smooth_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x, kernel, mode="same")


def build_profile(patch_gray: np.ndarray) -> np.ndarray | None:
    col = patch_gray.mean(axis=0).astype(np.float32)
    col_s = smooth_1d(col, PROFILE_SMOOTH_K)
    grad = np.diff(col_s, prepend=col_s[0])
    grad = np.abs(grad)
    grad_s = smooth_1d(grad, PROFILE_GRAD_SMOOTH_K)
    s = float(np.std(grad_s))
    if s < 1e-6:
        return None
    z = (grad_s - float(np.mean(grad_s))) / s
    return z.astype(np.float32)


def best_shift_signed(base_prof: np.ndarray, cur_prof: np.ndarray, max_shift: int) -> tuple[int, float]:
    w = len(base_prof)
    max_shift = int(min(max_shift, w - 2))
    best_s = 0
    best_score = -1e9

    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a = base_prof[: w - s]
            b = cur_prof[s:]
        else:
            ss = -s
            a = base_prof[ss:]
            b = cur_prof[: w - ss]
        if len(a) < 10:
            continue
        score = float(np.dot(a, b)) / float(len(a))
        if score > best_score:
            best_score = score
            best_s = s

    return int(best_s), float(best_score)


def aligned_residual(base_prof: np.ndarray, cur_prof: np.ndarray, shift: int) -> float:
    w = len(base_prof)
    s = int(shift)
    if s >= 0:
        a = base_prof[: w - s]
        b = cur_prof[s:]
    else:
        ss = -s
        a = base_prof[ss:]
        b = cur_prof[: w - ss]
    if len(a) < 10:
        return 1e9
    return float(np.mean(np.abs(a - b)))


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


def perspective_transform_points(H: np.ndarray, pts: list[tuple[float, float]]):
    pts_np = np.float32(pts).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts_np, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in out]


def order_quad_tltrbrbl(pts: list[tuple[float, float]]):
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


def quad_in_frame(quad: list[tuple[float, float]], w: int, h: int, margin: int = 2) -> bool:
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(xs) >= -margin) and (min(ys) >= -margin) and (max(xs) <= (w - 1 + margin)) and (max(ys) <= (h - 1 + margin))


def warp_quad_to_rect(gray: np.ndarray, quad_pts: list[tuple[float, float]], out_w: int, out_h: int) -> np.ndarray:
    src = np.float32(quad_pts)
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (out_w, out_h))


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
        print("Missing vision_config.json or template.png. Run the ROI calibration tool first.")
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

    base_profile = cfg.get("baseline_profile", None)
    base_profile = np.array(base_profile, dtype=np.float32) if base_profile is not None else None

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, 950, 650)
    cv2.moveWindow(WINDOW_MAIN, 20, 20)

    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_DEBUG, 1400, 900)
    cv2.moveWindow(WINDOW_DEBUG, 1000, 20)

    latch_mode = LATCH_MODE_DEFAULT
    per_part_latch = PER_PART_LATCH_DEFAULT
    clear_on_reseat = CLEAR_ON_RESEAT_DEFAULT

    state = "WAIT_FOR_PART"
    present_ct = 0
    leave_ct = 0

    bad = False
    prev_bad = False
    part_failed = False

    fail_consec = 0
    pass_consec = 0
    resid_fail_consec = 0

    shift_hist = deque(maxlen=SHIFT_HISTORY)
    score_hist = deque(maxlen=SCORE_HISTORY)
    resid_hist = deque(maxlen=RESID_HISTORY)

    cap_active = False
    cap_profiles = []
    cap_start_t = 0.0
    cap_next_sample_t = 0.0
    cap_interval = max(0.05, BASELINE_CAPTURE_DURATION_SEC / max(1, BASELINE_CAPTURE_SAMPLES))

    last_good_H = None
    hold_left = 0
    last_orb = {}

    paused = False
    show_debug = True

    unstable = False
    unstable_reason = ""

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

            if k in (ord('l'), ord('L')):
                latch_mode = not latch_mode

            if k in (ord('m'), ord('M')):
                per_part_latch = not per_part_latch

            if k in (ord('k'), ord('K')):
                clear_on_reseat = not clear_on_reseat

            if k in (ord('r'), ord('R')):
                bad = False
                part_failed = False
                fail_consec = pass_consec = 0
                resid_fail_consec = 0
                shift_hist.clear(); score_hist.clear(); resid_hist.clear()

            if k in (ord('u'), ord('U')):
                base_profile = None
                cfg["baseline_profile"] = None
                save_config(cfg)
                bad = False
                part_failed = False
                fail_consec = pass_consec = 0
                resid_fail_consec = 0
                shift_hist.clear(); score_hist.clear(); resid_hist.clear()

            # ORB homography
            H, orb = estimate_homography_orb(template, frame)
            last_orb = orb

            use_H = None
            H_is_fresh = False

            if H is not None and int(orb.get("inliers", 0)) >= MIN_INLIERS and float(orb.get("inlier_ratio", 0.0)) >= MIN_INLIER_RATIO:
                use_H = H
                H_is_fresh = True
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
                        unstable = False
                        unstable_reason = ""
                        bad = False if not latch_mode else bad
                        part_failed = False
                        fail_consec = pass_consec = 0
                        resid_fail_consec = 0
                        shift_hist.clear(); score_hist.clear(); resid_hist.clear()
                else:
                    leave_ct = leave_ct + 1 if not part_present else 0
                    if leave_ct >= LEAVE_FRAMES:
                        state = "WAIT_FOR_PART"
                        leave_ct = present_ct = 0
                        unstable = False
                        unstable_reason = ""
                        if not latch_mode:
                            bad = False
                            part_failed = False

            # seam patch
            seam_patch = None
            seam_quad_frame = None
            prof = None

            shift_signed = None
            shift_abs = None
            shift_med = None

            score = None
            score_med = None

            resid = None
            resid_med = None

            now = time.monotonic()

            if state == "INSPECT" and use_H is not None:
                seam_quad_frame = order_quad_tltrbrbl(
                    perspective_transform_points(use_H, [(float(p[0]), float(p[1])) for p in seam_quad_t])
                )
                fh, fw = frame.shape[:2]
                if quad_in_frame(seam_quad_frame, fw, fh, margin=5):
                    seam_patch = warp_quad_to_rect(frame, seam_quad_frame, warp_w, warp_h)
                    prof = build_profile(seam_patch)

            # baseline capture
            if k in (ord('b'), ord('B')) and (not cap_active):
                cap_active = True
                cap_profiles = []
                cap_start_t = now
                cap_interval = max(0.05, float(BASELINE_CAPTURE_DURATION_SEC) / max(1, int(BASELINE_CAPTURE_SAMPLES)))
                cap_next_sample_t = now

                bad = False
                part_failed = False
                unstable = False
                unstable_reason = ""
                fail_consec = pass_consec = 0
                resid_fail_consec = 0
                shift_hist.clear(); score_hist.clear(); resid_hist.clear()

            if cap_active:
                elapsed = now - cap_start_t
                if (prof is not None) and (now >= cap_next_sample_t) and (len(cap_profiles) < BASELINE_CAPTURE_SAMPLES):
                    cap_profiles.append(prof)
                    cap_next_sample_t = now + cap_interval

                if (len(cap_profiles) >= BASELINE_CAPTURE_SAMPLES) or (elapsed >= BASELINE_CAPTURE_DURATION_SEC):
                    cap_active = False
                    if len(cap_profiles) >= BASELINE_CAPTURE_MIN_SAMPLES:
                        stack = np.stack(cap_profiles, axis=0)
                        base_profile = np.median(stack, axis=0).astype(np.float32)
                        cfg["baseline_profile"] = base_profile.tolist()
                        save_config(cfg)
                    else:
                        print(f"[BASELINE] Not enough samples: {len(cap_profiles)} collected in {elapsed:.1f}s")

                    cap_profiles = []
                    bad = False
                    part_failed = False
                    unstable = False
                    unstable_reason = ""
                    fail_consec = pass_consec = 0
                    resid_fail_consec = 0
                    shift_hist.clear(); score_hist.clear(); resid_hist.clear()

            # metrics
            if (not cap_active) and base_profile is not None and prof is not None and len(base_profile) == len(prof):
                shift_signed, score = best_shift_signed(base_profile, prof, MAX_SHIFT_PX)
                shift_abs = abs(int(shift_signed))
                resid = aligned_residual(base_profile, prof, shift_signed)

                if H_is_fresh:
                    shift_hist.append(int(shift_abs))
                    score_hist.append(float(score))
                    resid_hist.append(float(resid))

                if len(shift_hist) > 0:
                    shift_med = int(np.median(np.array(shift_hist, dtype=np.int32)))
                if len(score_hist) > 0:
                    score_med = float(np.median(np.array(score_hist, dtype=np.float32)))
                if len(resid_hist) > 0:
                    resid_med = float(np.median(np.array(resid_hist, dtype=np.float32)))

            # UNSTABLE hysteresis
            if (not cap_active) and state == "INSPECT" and base_profile is not None:
                tracking_ok = (use_H is not None) and (prof is not None)
                if not tracking_ok:
                    unstable = True
                    unstable_reason = "TRACKING LOST" if use_H is None else "SEAM OUT OF VIEW"
                else:
                    if score_med is None:
                        unstable = True
                        unstable_reason = "REPOSITION"
                    else:
                        if not unstable:
                            if score_med <= SCORE_UNSTABLE_ENTER:
                                unstable = True
                                unstable_reason = "REPOSITION"
                        else:
                            if score_med >= SCORE_UNSTABLE_EXIT:
                                unstable = False
                                unstable_reason = ""
                            else:
                                unstable_reason = "REPOSITION"
            else:
                if state != "INSPECT":
                    unstable = False
                    unstable_reason = ""

            # BAD only when NOT unstable
            prev_bad = bool(bad)

            if (not cap_active) and (not paused) and state == "INSPECT" and (not unstable) and base_profile is not None and shift_med is not None:
                if shift_med >= HARD_SHIFT_FAIL_AT_OR_ABOVE:
                    bad = True
                if resid_med is not None and resid_med >= HARD_RESID_FAIL_AT_OR_ABOVE:
                    bad = True

                if not bad:
                    if shift_med >= FAIL_AT_OR_ABOVE_PX:
                        fail_consec += 1
                        pass_consec = 0
                    else:
                        pass_consec += 1
                        fail_consec = 0

                    if resid_med is not None and resid_med >= RESID_FAIL_AT_OR_ABOVE:
                        resid_fail_consec += 1
                    else:
                        resid_fail_consec = 0

                    if fail_consec >= FAIL_CONSEC_TO_TRIP:
                        bad = True
                    if resid_fail_consec >= RESID_CONSEC_TO_TRIP:
                        bad = True

                if (
                    (not per_part_latch)
                    and clear_on_reseat
                    and (pass_consec >= PASS_CONSEC_TO_CLEAR)
                    and (not latch_mode)
                ):
                    bad = False
                    part_failed = False

                if per_part_latch and (
                    (shift_med is not None and shift_med >= FAIL_AT_OR_ABOVE_PX)
                    or (resid_med is not None and resid_med >= RESID_FAIL_AT_OR_ABOVE)
                ):
                    part_failed = True

                if part_failed:
                    bad = True

            if (not prev_bad) and bad and seam_quad_frame is not None:
                ann = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.polylines(ann, [np.int32(seam_quad_frame).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
                info = (
                    f"State={state}\n"
                    f"shift_signed={shift_signed} shift_abs={shift_abs} shift_med={shift_med}\n"
                    f"score={score} score_med={score_med}\n"
                    f"resid={resid} resid_med={resid_med}\n"
                    f"unstable={unstable} reason={unstable_reason}\n"
                    f"ORB matches={last_orb.get('matches', 0)} inliers={last_orb.get('inliers', 0)} ratio={last_orb.get('inlier_ratio', 0):.3f} hold={hold_left} fresh={H_is_fresh}\n"
                )
                save_fail_images(frame, ann, info)

            # UI
            bg = np.zeros((650, 950, 3), dtype=np.uint8)
            txt = (0, 0, 0)

            if cap_active or base_profile is None:
                bg[:] = (0, 165, 255)
                if cap_active:
                    remaining_s = max(0.0, BASELINE_CAPTURE_DURATION_SEC - (now - cap_start_t))
                    remaining_n = max(0, BASELINE_CAPTURE_SAMPLES - len(cap_profiles))
                    label = f"CAPTURING BASELINE... ({remaining_n} samples, {remaining_s:.0f}s left)"
                else:
                    label = "SET BASELINE (press b)"
            else:
                if state != "INSPECT":
                    bg[:] = (0, 165, 255)
                    label = "WAIT_FOR_PART - POSITION PART"
                elif unstable:
                    bg[:] = (0, 165, 255)
                    label = f"UNSTABLE - {unstable_reason}"
                elif bad:
                    bg[:] = (0, 0, 255)
                    label = "BAD"
                    txt = (255, 255, 255)
                else:
                    bg[:] = (0, 255, 0)
                    label = "GOOD"

            cv2.putText(bg, label, (30, 110), FONT, 1.5, txt, 5)
            cv2.putText(bg, f"State: {state}", (30, 185), FONT, 1.0, txt, 2)

            cv2.putText(bg, f"ShiftAbs: {shift_abs}  ShiftMed({SHIFT_HISTORY}): {shift_med}  Fail>= {FAIL_AT_OR_ABOVE_PX}",
                        (30, 225), FONT, 0.85, txt, 2)

            cv2.putText(bg, f"ScoreMed({SCORE_HISTORY}): {None if score_med is None else f'{score_med:.3f}'}  Enter<= {SCORE_UNSTABLE_ENTER}  Exit>= {SCORE_UNSTABLE_EXIT}",
                        (30, 260), FONT, 0.85, txt, 2)

            cv2.putText(bg, f"ResidMed({RESID_HISTORY}): {None if resid_med is None else f'{resid_med:.3f}'}  Fail>= {RESID_FAIL_AT_OR_ABOVE}",
                        (30, 295), FONT, 0.85, txt, 2)

            cv2.putText(bg, f"ORB matches={last_orb.get('matches',0)} inliers={last_orb.get('inliers',0)} ratio={last_orb.get('inlier_ratio',0):.2f} hold={hold_left} fresh={H_is_fresh}",
                        (30, 335), FONT, 0.75, txt, 2)

            cv2.putText(bg, f"LatchMode={latch_mode} PerPartLatch={per_part_latch} ClearOnReseat={clear_on_reseat}",
                        (30, 370), FONT, 0.75, txt, 2)

            cv2.putText(bg, "Keys: b=Baseline  u=Clear  r=Reset  m=PerPart  d=Debug  p=Pause  ESC=Quit",
                        (30, 620), FONT, 0.7, txt, 2)

            cv2.imshow(WINDOW_MAIN, bg)

            if show_debug:
                dbg = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if seam_quad_frame is not None:
                    cv2.polylines(dbg, [np.int32(seam_quad_frame).reshape(-1, 1, 2)], True, (0, 255, 255), 2)

                if seam_patch is not None:
                    patch_vis = cv2.cvtColor(seam_patch, cv2.COLOR_GRAY2BGR)
                    scale = 1.6
                    patch_vis = cv2.resize(patch_vis, (int(warp_w * scale), int(warp_h * scale)))
                    ph, pw = patch_vis.shape[:2]
                    dbg[0:ph, 0:pw] = patch_vis
                    cv2.rectangle(dbg, (0, 0), (pw, ph), (255, 255, 255), 1)

                cv2.imshow(WINDOW_DEBUG, dbg)

    except Exception as e:
        print("\nERROR:", e)
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
