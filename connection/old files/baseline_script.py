import os
import json
import cv2
import numpy as np
from collections import deque
from datetime import datetime
from pypylon import pylon

print("RUNNING FILE:", os.path.abspath(__file__))

# ============================================================
# FILES
# ============================================================
CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"
FAIL_DIR = "fails"

# ============================================================
# PASS/FAIL
# ============================================================
GAP_FAIL_AT_OR_ABOVE_PX = 4

# Part-present thresholds (template match)
MATCH_MIN_PRESENT = 0.65
MATCH_MIN_LEAVE   = 0.55
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# Debounce
FAIL_CONSEC_TO_TRIP = 2
PASS_CONSEC_TO_CLEAR = 3

# Option B: stable behavior + per-part latch once truly stable
LATCH_MODE_DEFAULT = False
ENABLE_PER_PART_FAIL_LATCH = True

# ============================================================
# BOUNDARY DETECTION (Option B stable)
# ============================================================
COLMEAN_SMOOTH_K = 11
GRAD_SMOOTH_K = 9
BOUNDARY_SEARCH_FRAC_PREBASELINE = (0.10, 0.90)

BASELINE_MARGIN_LEFT = 5
BASELINE_MARGIN_RIGHT = 240

BOUNDARY_CONF_MIN = 1.0

BOUNDARY_HISTORY = 9
JITTER_PX_MAX = 3
STABLE_REQUIRED = 6
JUMP_REJECT_PX = 10

FAIL_IF_BOUNDARY_MISSING = False

# ============================================================
# ROI SAFETY / AUTO-EXPANSION (THIS IS THE IMPORTANT FIX)
# ============================================================
# If your seam ROI is too skinny, the boundary cannot move when you pull out.
MIN_SEAM_W = 260
MIN_SEAM_H = 120

# Auto-expand seam ROI at runtime (gives "runway" to the right)
EXPAND_LEFT_PX  = 40
EXPAND_RIGHT_PX = 320
EXPAND_UP_PX    = 10
EXPAND_DOWN_PX  = 10

# Warn if baseline is too close to the left edge (means not enough runway)
BASELINE_MIN_FROM_LEFT_EDGE = 30

# ============================================================
# UI / KEYS
# ============================================================
WINDOW_MAIN = "Trial Inspection"
WINDOW_DEBUG = "DEBUG VIEW"
WINDOW_CAL = "CALIBRATION"
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_ESC = 27

# ============================================================
# UTILS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def smooth_1d(x, k):
    if k < 3: k = 3
    if k % 2 == 0: k += 1
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

def delete_calibration_files():
    for f in (CONFIG_PATH, TEMPLATE_PATH):
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass

# ============================================================
# ROI SELECTOR
# ============================================================
class ROISelector:
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

# ============================================================
# BOUNDARY DETECTION
# ============================================================
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

    return idx, {"peak": peak, "med": med, "idx": idx, "a": a, "b": b}

# ============================================================
# CAMERA
# ============================================================
def start_camera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Lock auto exposure/gain off (reduces hand/reflection effects)
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

# ============================================================
# CONFIG
# ============================================================
def load_config():
    if not (os.path.exists(CONFIG_PATH) and os.path.exists(TEMPLATE_PATH)):
        return None
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print("[CFG] Saved", CONFIG_PATH)

# ============================================================
# CALIBRATION
# ============================================================
def calibrate(camera, converter):
    cv2.namedWindow(WINDOW_CAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CAL, 1400, 900)
    cv2.moveWindow(WINDOW_CAL, 20, 20)

    selector = ROISelector(WINDOW_CAL)
    cv2.setMouseCallback(WINDOW_CAL, selector.callback)

    mode = "TEMPLATE"
    template_roi = None
    seam_roi = None

    print("\n--- CALIBRATION ---")
    print("t = TEMPLATE ROI (controller only)")
    print("s = SEAM ROI (controller + connector) -- MUST BE WIDE")
    print("w = Write")
    print("ESC = Cancel")

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue
        frame = converter.Convert(grab).GetArray()
        grab.Release()

        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if selector.last_roi is not None:
            if mode == "TEMPLATE":
                template_roi = selector.last_roi
            else:
                seam_roi = selector.last_roi

        if template_roi is not None:
            x, y, w, h = template_roi
            cv2.rectangle(disp, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(disp, "TEMPLATE (controller only)", (x, max(20, y-8)),
                        FONT, 0.8, (255,0,0), 2)

        if seam_roi is not None:
            x, y, w, h = seam_roi
            color = (0, 255, 255) if (w >= MIN_SEAM_W and h >= MIN_SEAM_H) else (0, 0, 255)
            cv2.rectangle(disp, (x, y), (x+w, y+h), color, 2)
            cv2.putText(disp, f"SEAM ROI  w={w} h={h}", (x, max(20, y-8)),
                        FONT, 0.8, color, 2)
            if w < MIN_SEAM_W or h < MIN_SEAM_H:
                cv2.putText(disp, f"WARNING: seam ROI too small (min {MIN_SEAM_W}x{MIN_SEAM_H})",
                            (20, 80), FONT, 0.9, (0,0,255), 2)

        cv2.putText(disp, f"Mode: {mode} (t/s switch)  w=Write  ESC=Exit",
                    (20, 40), FONT, 0.9, (255,255,255), 2)
        cv2.imshow(WINDOW_CAL, disp)

        k = cv2.waitKey(1) & 0xFF
        if k == KEY_ESC:
            cv2.destroyWindow(WINDOW_CAL)
            return False

        if k in (ord('t'), ord('T')):
            mode = "TEMPLATE"
        if k in (ord('s'), ord('S')):
            mode = "SEAM"

        if k in (ord('w'), ord('W')):
            if template_roi is None or seam_roi is None:
                print("[CAL] Need BOTH template ROI and seam ROI.")
                continue

            sx, sy, sw, sh = seam_roi
            if sw < MIN_SEAM_W or sh < MIN_SEAM_H:
                print(f"[CAL] Seam ROI too small. Make it at least {MIN_SEAM_W}x{MIN_SEAM_H}.")
                continue

            tx, ty, tw, th = template_roi

            template_img = frame[ty:ty+th, tx:tx+tw].copy()
            cv2.imwrite(TEMPLATE_PATH, template_img)
            print("[CAL] Saved", TEMPLATE_PATH)

            seam_offset = {"x": int(sx-tx), "y": int(sy-ty), "w": int(sw), "h": int(sh)}

            cfg = {
                "seam_offset_from_template": seam_offset,
                "baseline_boundary_x": None
            }
            save_config(cfg)
            cv2.destroyWindow(WINDOW_CAL)
            return True

# ============================================================
# RUN
# ============================================================
def run(camera, converter, cfg):
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError("template.png missing. Recalibrate.")
    th, tw = template.shape[:2]

    seam_off = cfg["seam_offset_from_template"]
    baseline = cfg.get("baseline_boundary_x", None)
    if baseline is not None:
        baseline = int(baseline)

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

    print("\n--- RUN MODE ---")
    print("Keys: b=Baseline u=ClearBaseline l=Latch r=Reset p=Pause d=Debug c=Recal ESC=Quit")
    print("TIP: click an OpenCV window before pressing keys.\n")

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue
        frame = converter.Convert(grab).GetArray()
        grab.Release()

        H, W = frame.shape
        k = cv2.waitKey(1) & 0xFF

        # keys
        if k == KEY_ESC:
            return "QUIT"
        if k in (ord('c'), ord('C')):
            return "RECAL"
        if k in (ord('l'), ord('L')):
            latch_mode = not latch_mode
            print("LATCH_MODE =", latch_mode)
        if k in (ord('p'), ord('P')):
            paused = not paused
        if k in (ord('d'), ord('D')):
            show_debug = not show_debug
        if k in (ord('r'), ord('R')):
            bad = False
            part_failed = False
            fail_consec = pass_consec = 0
            max_gap_stable = 0
            saved_this_part = False
            boundary_hist.clear()
            prev_boundary = None
            stable_count = 0
        if k in (ord('u'), ord('U')):
            baseline = None
            cfg["baseline_boundary_x"] = None
            save_config(cfg)
            print("[BASELINE] cleared")

        # template match
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, match_score, _, match_loc = cv2.minMaxLoc(res)
        mx, my = match_loc

        # seam ROI from offsets + AUTO-EXPANSION
        rx = int(clamp(mx + seam_off["x"] - EXPAND_LEFT_PX, 0, W-1))
        ry = int(clamp(my + seam_off["y"] - EXPAND_UP_PX,   0, H-1))
        rw = int(seam_off["w"] + EXPAND_LEFT_PX + EXPAND_RIGHT_PX)
        rh = int(seam_off["h"] + EXPAND_UP_PX + EXPAND_DOWN_PX)
        rx2 = int(clamp(rx + rw, 0, W))
        ry2 = int(clamp(ry + rh, 0, H))

        # part present state
        if not paused:
            if state == "WAIT_FOR_PART":
                present_ct = present_ct + 1 if match_score >= MATCH_MIN_PRESENT else 0
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
                leave_ct = leave_ct + 1 if match_score < MATCH_MIN_LEAVE else 0
                if leave_ct >= LEAVE_FRAMES:
                    state = "WAIT_FOR_PART"
                    leave_ct = present_ct = 0

        # measurement
        boundary_x = None
        boundary_med = None
        gap_px = None
        dbg_note = ""

        if state == "INSPECT":
            roi = frame[ry:ry2, rx:rx2]
            boundary_x, dbg = detect_boundary_x(roi, baseline=baseline)

            # reject big jumps
            if boundary_x is not None and prev_boundary is not None and abs(boundary_x - prev_boundary) > JUMP_REJECT_PX:
                boundary_x = None
                dbg_note = "jump_rejected"
            if boundary_x is not None:
                prev_boundary = boundary_x
                boundary_hist.append(boundary_x)
                boundary_med = int(np.median(boundary_hist))
            else:
                if not dbg_note:
                    dbg_note = dbg.get("reason", "")

            # stability
            if len(boundary_hist) >= 5:
                jitter = max(boundary_hist) - min(boundary_hist)
                stable_count = stable_count + 1 if jitter <= JITTER_PX_MAX else 0
            else:
                stable_count = 0

            # baseline capture
            if k in (ord('b'), ord('B')):
                if boundary_med is not None:
                    baseline = boundary_med
                    cfg["baseline_boundary_x"] = int(baseline)
                    save_config(cfg)
                    print("[BASELINE] captured =", baseline)
                else:
                    print("[BASELINE] boundary not found; cannot capture baseline.")

            if baseline is not None and boundary_med is not None:
                gap_px = int(max(0, boundary_med - baseline))

            # decision
            if not paused and baseline is not None:
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
                    if pass_consec >= PASS_CONSEC_TO_CLEAR and not latch_mode and not ENABLE_PER_PART_FAIL_LATCH:
                        bad = False

                    # update stable max-gap only when stable
                    if stable_count >= STABLE_REQUIRED:
                        max_gap_stable = max(max_gap_stable, gap_px)

                    # per-part latch only from stable max-gap
                    if ENABLE_PER_PART_FAIL_LATCH and max_gap_stable >= GAP_FAIL_AT_OR_ABOVE_PX:
                        part_failed = True
                    if part_failed:
                        bad = True

                    # save fail once per part
                    if bad and not saved_this_part:
                        ann = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(ann, (mx, my), (mx+tw, my+th), (255,0,0), 2)
                        cv2.rectangle(ann, (rx, ry), (rx2, ry2), (0,255,255), 2)
                        if boundary_med is not None:
                            cv2.line(ann, (rx + boundary_med, ry), (rx + boundary_med, ry2), (255,255,0), 2)
                        if baseline is not None:
                            cv2.line(ann, (rx + baseline, ry), (rx + baseline, ry2), (255,0,255), 1)
                        info = f"baseline={baseline}, boundary_med={boundary_med}, gap={gap_px}, max_gap_stable={max_gap_stable}, stable_count={stable_count}"
                        save_fail_images(frame, ann, info_text=info)
                        saved_this_part = True

        # MAIN screen
        bg = np.zeros((650, 950, 3), dtype=np.uint8)
        if baseline is None:
            bg[:] = (0, 165, 255)
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

        cv2.putText(bg, label, (30, 120), FONT, 1.8, txt, 6)
        cv2.putText(bg, f"State: {state}", (30, 200), FONT, 1.0, txt, 2)
        cv2.putText(bg, f"MatchScore: {match_score:.2f}", (30, 240), FONT, 0.9, txt, 2)
        cv2.putText(bg, f"BaselineX: {baseline}", (30, 280), FONT, 0.9, txt, 2)
        cv2.putText(bg, f"BoundaryMed: {boundary_med}  Gap: {gap_px}", (30, 320), FONT, 0.85, txt, 2)
        cv2.putText(bg, f"MaxGap(stable): {max_gap_stable} (Fail>= {GAP_FAIL_AT_OR_ABOVE_PX})", (30, 360), FONT, 0.85, txt, 2)
        cv2.putText(bg, f"StableCount: {stable_count}/{STABLE_REQUIRED}  PartFailed: {part_failed}", (30, 400), FONT, 0.85, txt, 2)

        if baseline is not None and baseline < BASELINE_MIN_FROM_LEFT_EDGE:
            cv2.putText(bg, "WARNING: Baseline too close to left edge - move seam ROI left or widen",
                        (30, 440), FONT, 0.8, (0,0,0), 2)

        if dbg_note:
            cv2.putText(bg, f"Boundary note: {dbg_note}", (30, 480), FONT, 0.8, txt, 2)

        cv2.putText(bg, "Keys: b=Baseline u=Clear l=Latch r=Reset p=Pause d=Debug c=Recal ESC=Quit",
                    (30, 620), FONT, 0.7, txt, 2)

        cv2.imshow(WINDOW_MAIN, bg)

        # DEBUG view
        if show_debug:
            dbg_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(dbg_img, (mx, my), (mx+tw, my+th), (255,0,0), 2)

            if state == "INSPECT":
                cv2.rectangle(dbg_img, (rx, ry), (rx2, ry2), (0,255,255), 2)
                if boundary_med is not None:
                    cv2.line(dbg_img, (rx + boundary_med, ry), (rx + boundary_med, ry2), (255,255,0), 2)
                if baseline is not None:
                    cv2.line(dbg_img, (rx + baseline, ry), (rx + baseline, ry2), (255,0,255), 1)

            cv2.imshow(WINDOW_DEBUG, dbg_img)

    return "QUIT"

# ============================================================
# MAIN
# ============================================================
def main():
    print("NOTE: close Basler pylon Viewer before running (exclusive camera access).")
    camera = None
    try:
        camera, converter = start_camera()

        while True:
            cfg = load_config()
            if cfg is None:
                ok = calibrate(camera, converter)
                if not ok:
                    print("Calibration cancelled. Exiting.")
                    break
            else:
                result = run(camera, converter, cfg)
                if result == "RECAL":
                    delete_calibration_files()
                    try:
                        cv2.destroyWindow(WINDOW_MAIN)
                        cv2.destroyWindow(WINDOW_DEBUG)
                    except Exception:
                        pass
                    print("[INFO] Recalibration requested -> returning to CALIBRATION.")
                    continue
                break

    except Exception as e:
        print("\nERROR:", e)
        print("If you see 'Device is exclusively opened by another client', close pylon Viewer and retry.\n")

    finally:
        if camera is not None:
            try: camera.StopGrabbing()
            except: pass
            try: camera.Close()
            except: pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()