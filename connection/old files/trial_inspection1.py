import os
import json
import cv2
import numpy as np
from datetime import datetime
from pypylon import pylon

import os
print("RUNNING FILE:", os.path.abspath(__file__))

# ============================================================
# PROOF YOU ARE RUNNING THE RIGHT FILE
# ============================================================
print("RUNNING FILE:", os.path.abspath(__file__))

# ============================================================
# FILES / OUTPUT
# ============================================================
CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"
FAIL_DIR = "fails"

# ============================================================
# STARTING SETTINGS (YOU CAN TUNE LATER)
# ============================================================
# Threshold: FAIL if gap >= this many pixels
GAP_FAIL_AT_OR_ABOVE_PX = 4

# Debounce (consecutive frames)
FAIL_CONSEC_TO_TRIP = 2
PASS_CONSEC_TO_CLEAR = 3  # only used in LIVE mode

# Template match thresholds (part present / part leave)
MATCH_MIN_PRESENT = 0.65
MATCH_MIN_LEAVE = 0.55
ENTER_PRESENT_FRAMES = 3
LEAVE_FRAMES = 6

# Edge profile settings (robust seated + pulled-out)
EDGE_PROFILE_SMOOTH_K = 9            # smoothing kernel (odd)
LEFT_REGION_END_FRAC = 0.30          # search xA in [0 .. 0.30*w]
RIGHT_REGION_START_FRAC = 0.55       # search xB in [0.55*w .. w]
SUPPRESS_FRAC = 0.03                 # suppress neighborhood around xA
MIN_EDGE_CONF_REL = 0.20             # min confidence for xA/xB edges

# Default mode for development
LATCH_MODE_DEFAULT = False           # False = Live behavior; True = latched until reset

# ============================================================
# UI / KEYS
# ============================================================
WINDOW_MAIN = "Trial Inspection"
WINDOW_DEBUG = "DEBUG VIEW"
WINDOW_CAL = "CALIBRATION"
FONT = cv2.FONT_HERSHEY_SIMPLEX

KEY_ESC = 27
KEY_RESET = ord('r')
KEY_TOGGLE_DEBUG = ord('d')
KEY_TOGGLE_LATCH = ord('l')
KEY_PAUSE = ord('p')
KEY_RECAL = ord('c')
KEY_TEMPLATE_MODE = ord('t')
KEY_SEAM_MODE = ord('s')
KEY_WRITE = ord('w')

# ============================================================
# UTILS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

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

# ============================================================
# ROI SELECTOR (mouse drag, auto-accept)
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
# GAP MEASUREMENT (robust xA + suppression + xB)
# ============================================================
def measure_gap_edges_suppressed(seam_roi_gray):
    """
    Measures gap by:
      - find strongest vertical edge in LEFT region -> xA (controller edge)
      - suppress around xA so xB can't pick same edge
      - find strongest vertical edge in RIGHT region -> xB (connector edge)
      - gap = max(0, xB - xA)

    Returns:
      gap_px (int), xA (int|None), xB (int|None), conf_ok (bool)
    """
    roi = cv2.GaussianBlur(seam_roi_gray, (5, 5), 0)
    sx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    edge = np.abs(sx)
    profile = edge.sum(axis=0).astype(np.float32)

    w = len(profile)
    if w < 10:
        return 0, None, None, False

    # Smooth
    k = EDGE_PROFILE_SMOOTH_K
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    prof = np.convolve(profile, kernel, mode="same")

    # Regions
    left_end = int(LEFT_REGION_END_FRAC * w)
    right_start = int(RIGHT_REGION_START_FRAC * w)
    left_end = clamp(left_end, 3, w - 3)
    right_start = clamp(right_start, 0, w - 4)

    left_region = prof[:left_end]
    if left_region.size == 0:
        return 0, None, None, False
    xA = int(np.argmax(left_region))

    # Suppress around xA
    suppress = prof.copy()
    half = max(4, int(SUPPRESS_FRAC * w))
    a0 = max(0, xA - half)
    a1 = min(w, xA + half + 1)
    suppress[a0:a1] = 0.0

    right_region = suppress[right_start:]
    if right_region.size == 0:
        return 0, xA, None, False
    xB = int(np.argmax(right_region)) + right_start

    mx = float(np.max(prof)) if prof.size else 0.0
    if mx <= 0:
        return 0, xA, xB, False

    confA = float(prof[xA]) / mx
    confB = float(prof[xB]) / mx
    conf_ok = (confA >= MIN_EDGE_CONF_REL) and (confB >= MIN_EDGE_CONF_REL)

    if not conf_ok:
        # treat as seated / not measurable -> gap 0
        return 0, xA, xB, False

    if xB < xA:
        xA, xB = xB, xA

    gap_px = int(max(0, xB - xA))
    return gap_px, xA, xB, True

# ============================================================
# CAMERA STARTUP
# ============================================================
def start_camera():
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Recommended: tune exposure in pylon Viewer, then keep these Off
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
# CONFIG LOAD/SAVE
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
    """
    Calibration:
      - Press T: drag TEMPLATE ROI (include controller + connector + stable surroundings)
      - Press S: drag SEAM ROI (span both sides of the interface)
      - Press W: write template.png and vision_config.json
    """
    cv2.namedWindow(WINDOW_CAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CAL, 1100, 750)
    cv2.moveWindow(WINDOW_CAL, 20, 20)

    selector = ROISelector(WINDOW_CAL)
    cv2.setMouseCallback(WINDOW_CAL, selector.callback)

    mode = "TEMPLATE"
    template_roi = None
    seam_roi = None

    print("\n--- CALIBRATION ---")
    print("Click the CALIBRATION window so it has focus.")
    print("T = Template ROI (controller + connector, stable features)")
    print("S = Seam ROI (spans both sides of interface)")
    print("W = Write config + template")
    print("ESC = Exit\n")

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
            cv2.rectangle(disp, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(disp, "TEMPLATE ROI", (x, max(20, y-8)), FONT, 0.8, (255,0,0), 2)

        if seam_roi is not None:
            x, y, w, h = seam_roi
            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(disp, "SEAM ROI", (x, max(20, y-8)), FONT, 0.8, (0,255,255), 2)

        cv2.putText(disp, f"Mode: {mode}   (T=Template  S=Seam  W=Write  ESC=Exit)",
                    (20, 40), FONT, 0.9, (255,255,255), 2)
        cv2.putText(disp, "Drag ROI with mouse (auto-accept on release).",
                    (20, 75), FONT, 0.75, (255,255,255), 2)

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
                print("[CAL] Need BOTH Template ROI and Seam ROI before writing.")
                continue

            tx, ty, tw, th = template_roi
            sx, sy, sw, sh = seam_roi

            template_img = frame[ty:ty+th, tx:tx+tw].copy()
            cv2.imwrite(TEMPLATE_PATH, template_img)
            print("[CAL] Saved", TEMPLATE_PATH)

            seam_offset = {
                "x": int(sx - tx),
                "y": int(sy - ty),
                "w": int(sw),
                "h": int(sh)
            }

            cfg = {
                "seam_offset_from_template": seam_offset,
                "gap_fail_at_or_above_px": int(GAP_FAIL_AT_OR_ABOVE_PX),
                "match_min_present": float(MATCH_MIN_PRESENT),
                "match_min_leave": float(MATCH_MIN_LEAVE),
                "enter_present_frames": int(ENTER_PRESENT_FRAMES),
                "leave_frames": int(LEAVE_FRAMES)
            }
            save_config(cfg)
            cv2.destroyWindow(WINDOW_CAL)
            return True

# ============================================================
# RUN MODE (TWO WINDOWS GUARANTEED)
# ============================================================
def run(camera, converter, cfg):
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError("template.png missing. Run calibration again.")

    th, tw = template.shape[:2]
    seam_off = cfg["seam_offset_from_template"]

    gap_thr = int(cfg.get("gap_fail_at_or_above_px", GAP_FAIL_AT_OR_ABOVE_PX))
    match_present = float(cfg.get("match_min_present", MATCH_MIN_PRESENT))
    match_leave = float(cfg.get("match_min_leave", MATCH_MIN_LEAVE))
    enter_frames = int(cfg.get("enter_present_frames", ENTER_PRESENT_FRAMES))
    leave_frames = int(cfg.get("leave_frames", LEAVE_FRAMES))

    # Create windows and force them on-screen
    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MAIN, 950, 650)
    cv2.moveWindow(WINDOW_MAIN, 20, 20)

    cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_DEBUG, 1100, 750)
    cv2.moveWindow(WINDOW_DEBUG, 1000, 20)

    latch_mode = LATCH_MODE_DEFAULT
    paused = False
    show_debug = True

    # State
    state = "WAIT_FOR_PART"
    present_ct = 0
    leave_ct = 0

    # Decision
    bad_latched = False
    fail_consec = 0
    pass_consec = 0
    max_gap_seen = 0
    saved_this_part = False

    print("\n--- RUN MODE ---")
    print("Click a window to give it focus for keys.")
    print("Keys: L=ToggleLatch  R=Reset  P=Pause  D=Debug  C=Recalibrate  ESC=Quit\n")

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue
        frame = converter.Convert(grab).GetArray()
        grab.Release()

        H, W = frame.shape

        # Key handling (must have focus on an OpenCV window)
        k = cv2.waitKey(1) & 0xFF
        if k == KEY_ESC:
            break
        if k == KEY_TOGGLE_LATCH:
            latch_mode = not latch_mode
            print("LATCH_MODE =", latch_mode)
        if k == KEY_RESET:
            bad_latched = False
            fail_consec = 0
            pass_consec = 0
            max_gap_seen = 0
            saved_this_part = False
            print("[RESET] BAD cleared")
        if k == KEY_PAUSE:
            paused = not paused
            print("[PAUSE]" if paused else "[RESUME]")
        if k == KEY_TOGGLE_DEBUG:
            show_debug = not show_debug
            if not show_debug:
                try:
                    cv2.destroyWindow(WINDOW_DEBUG)
                except Exception:
                    pass
            else:
                cv2.namedWindow(WINDOW_DEBUG, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_DEBUG, 1100, 750)
                cv2.moveWindow(WINDOW_DEBUG, 1000, 20)
        if k == KEY_RECAL:
            # Return to main loop and delete config to force recalibration
            return "RECAL"

        # Template match
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, match_score, _, match_loc = cv2.minMaxLoc(res)
        mx, my = match_loc

        # Compute seam ROI from offsets
        rx = int(clamp(mx + seam_off["x"], 0, W - 1))
        ry = int(clamp(my + seam_off["y"], 0, H - 1))
        rw = int(seam_off["w"])
        rh = int(seam_off["h"])
        rx2 = int(clamp(rx + rw, 0, W))
        ry2 = int(clamp(ry + rh, 0, H))

        # State machine (only update when not paused)
        if not paused:
            if state == "WAIT_FOR_PART":
                if match_score >= match_present:
                    present_ct += 1
                else:
                    present_ct = 0

                if present_ct >= enter_frames:
                    state = "INSPECT"
                    present_ct = 0
                    leave_ct = 0
                    fail_consec = 0
                    pass_consec = 0
                    max_gap_seen = 0
                    saved_this_part = False
                    if not latch_mode:
                        bad_latched = False

            else:  # INSPECT
                if match_score < match_leave:
                    leave_ct += 1
                else:
                    leave_ct = 0

                if leave_ct >= leave_frames:
                    state = "WAIT_FOR_PART"
                    leave_ct = 0
                    present_ct = 0
                    if not latch_mode:
                        bad_latched = False

        # Gap measurement (only meaningful in INSPECT)
        gap_px = 0
        xA = xB = None
        conf_ok = False

        if state == "INSPECT":
            roi = frame[ry:ry2, rx:rx2]
            gap_px, xA, xB, conf_ok = measure_gap_edges_suppressed(roi)

            if not paused:
                max_gap_seen = max(max_gap_seen, gap_px)

                is_fail = (gap_px >= gap_thr)
                if is_fail:
                    fail_consec += 1
                    pass_consec = 0
                else:
                    pass_consec += 1
                    fail_consec = 0

                if latch_mode:
                    if fail_consec >= FAIL_CONSEC_TO_TRIP:
                        bad_latched = True
                    # Optional conservative latch using max gap seen:
                    if max_gap_seen >= gap_thr:
                        bad_latched = True
                else:
                    # LIVE mode: allow returning to GOOD after stable PASS
                    if fail_consec >= FAIL_CONSEC_TO_TRIP:
                        bad_latched = True
                    if pass_consec >= PASS_CONSEC_TO_CLEAR:
                        bad_latched = False

                # Save FAIL images once per part when BAD first occurs
                if bad_latched and (not saved_this_part):
                    annotated_for_save = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(annotated_for_save, (mx, my), (mx+tw, my+th), (255, 0, 0), 2)
                    cv2.rectangle(annotated_for_save, (rx, ry), (rx2, ry2), (0, 255, 255), 2)
                    info = (f"gap_px={gap_px}, max_gap_seen={max_gap_seen}, match={match_score:.3f}, "
                            f"latch_mode={latch_mode}, state={state}")
                    save_fail_images(frame, annotated_for_save, info_text=info)
                    saved_this_part = True

        # ---------------------------
        # BUILD MAIN GREEN/RED WINDOW
        # ---------------------------
        bg = np.zeros((650, 950, 3), dtype=np.uint8)

        if bad_latched:
            bg[:] = (0, 0, 255)  # red
            label = "BAD (LATCHED)" if latch_mode else "BAD"
            txt = (255, 255, 255)
        else:
            bg[:] = (0, 255, 0)  # green
            label = "GOOD"
            txt = (0, 0, 0)

        cv2.putText(bg, label, (30, 140), FONT, 3.0, txt, 10)
        cv2.putText(bg, f"State: {state}", (30, 220), FONT, 1.2, txt, 3)
        cv2.putText(bg, f"Gap: {gap_px}px   MaxGapSeen: {max_gap_seen}px   (Fail>= {gap_thr})",
                    (30, 280), FONT, 1.0, txt, 3)
        cv2.putText(bg, f"MatchScore: {match_score:.2f}", (30, 330), FONT, 1.0, txt, 3)
        cv2.putText(bg, f"LatchMode: {latch_mode}   Paused: {paused}", (30, 380), FONT, 0.9, txt, 2)

        cv2.putText(bg, "Keys: L=Latch  R=Reset  P=Pause  D=Debug  C=Recal  ESC=Quit",
                    (30, 620), FONT, 0.8, txt, 2)

        cv2.imshow(WINDOW_MAIN, bg)

        # ---------------------------
        # BUILD DEBUG VIEW WINDOW
        # ---------------------------
        if show_debug:
            dbg = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            # Template match box (blue)
            cv2.rectangle(dbg, (mx, my), (mx+tw, my+th), (255, 0, 0), 2)
            cv2.putText(dbg, f"Match: {match_score:.2f}", (mx, max(20, my-10)),
                        FONT, 0.8, (255, 0, 0), 2)

            # Seam ROI (yellow) only when in INSPECT
            if state == "INSPECT":
                cv2.rectangle(dbg, (rx, ry), (rx2, ry2), (0, 255, 255), 2)

                if xA is not None and xB is not None:
                    # red = controller edge, green = connector edge
                    cv2.line(dbg, (rx + xA, ry), (rx + xA, ry2), (0, 0, 255), 2)
                    cv2.line(dbg, (rx + xB, ry), (rx + xB, ry2), (0, 255, 0), 2)

                color = (0, 0, 255) if gap_px >= gap_thr else (0, 255, 0)
                cv2.putText(dbg, f"gap_px={gap_px} (Fail>= {gap_thr})", (30, 60),
                            FONT, 1.2, color, 3)

                if not conf_ok:
                    cv2.putText(dbg, "NOTE: edges low confidence (often seated) -> gap treated as 0",
                                (30, 95), FONT, 0.8, (0, 255, 255), 2)

            cv2.putText(dbg, f"LatchMode={latch_mode}  Paused={paused}  State={state}",
                        (30, H - 20), FONT, 0.8, (255, 255, 255), 2)

            cv2.imshow(WINDOW_DEBUG, dbg)

    return "QUIT"

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print("NOTE: Close Basler pylon Viewer before running this script (exclusive camera access).")
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
                    # delete config/template so calibration is forced next loop
                    try:
                        if os.path.exists(CONFIG_PATH):
                            os.remove(CONFIG_PATH)
                        if os.path.exists(TEMPLATE_PATH):
                            os.remove(TEMPLATE_PATH)
                        print("[INFO] Deleted config/template for recalibration.")
                    except Exception:
                        pass
                    continue
                break

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