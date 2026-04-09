import cv2
import json
import os
import numpy as np
from datetime import datetime
from pypylon import pylon

# ============================================================
# FILES / FOLDERS
# ============================================================
CONFIG_PATH = "vision_config.json"
TEMPLATE_PATH = "template.png"
FAIL_DIR = "fails"

# ============================================================
# TRIAL SETTINGS (TUNE AFTER FIRST RUN)
# ============================================================
# Decision threshold: FAIL if gap >= this many pixels
GAP_FAIL_AT_OR_ABOVE_PX = 4

# Conservative logic (bad part passing is worse)
FAIL_CONSEC_TO_TRIP = 2          # consecutive bad frames required to latch BAD
USE_MAX_GAP_LATCH = True         # if max gap seen crosses threshold => latch BAD

# Part present / part leave based on template match score
MATCH_MIN_PRESENT = 0.65         # must be >= for part present
MATCH_MIN_LEAVE = 0.55           # must be <  for part left (lower than present)
ENTER_PRESENT_FRAMES = 3         # consecutive frames >= present threshold to arm
LEAVE_FRAMES = 6                 # consecutive frames < leave threshold to re-arm

# Edge detection tuning
EDGE_PROFILE_SMOOTH_K = 9        # smoothing kernel length (odd number)
PEAK_REL_THRESH = 0.35           # peaks must be >= this fraction of max profile
PAIR_MAX_SEP_PX = 80             # max distance between the two edges we consider as the "gap"

# UI
WINDOW_MAIN = "Trial Inspection"
WINDOW_DEBUG = "DEBUG VIEW"
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Keys
KEY_ESC = 27
KEY_RESET = ord('r')
KEY_TOGGLE_DEBUG = ord('d')
KEY_RECAL = ord('c')


# ============================================================
# UTILS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def save_fail_images(frame_gray, annotated_bgr, info_text=""):
    os.makedirs(FAIL_DIR, exist_ok=True)
    ts = now_stamp()
    raw_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_raw.png")
    ann_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_annotated.png")
    cv2.imwrite(raw_path, frame_gray)
    cv2.imwrite(ann_path, annotated_bgr)
    print(f"[SAVE] {raw_path}")
    print(f"[SAVE] {ann_path}")
    if info_text:
        txt_path = os.path.join(FAIL_DIR, f"FAIL_{ts}_info.txt")
        with open(txt_path, "w") as f:
            f.write(info_text)
        print(f"[SAVE] {txt_path}")


# ============================================================
# ROI SELECTOR (mouse drag)
# ============================================================
class ROISelector:
    """
    Drag a rectangle with the mouse.
    After mouse release, last_roi is updated (auto-accept).
    """
    def __init__(self, window_name):
        self.window_name = window_name
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

    def draw(self, img, color=(0, 255, 255), label="ROI"):
        if self.dragging:
            cv2.rectangle(img, (self.x0, self.y0), (self.x1, self.y1), color, 2)
        if self.last_roi is not None:
            x, y, w, h = self.last_roi
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, max(20, y - 8)), FONT, 0.8, color, 2)


# ============================================================
# GAP MEASUREMENT (EDGE-TO-EDGE)
# ============================================================
def find_peaks_1d(profile, rel_thresh=0.35, search_end=None):
    """
    Simple local maxima detection (no scipy).
    Returns list of peak indices.
    """
    if profile is None or len(profile) < 5:
        return []

    mx = float(np.max(profile))
    if mx <= 0:
        return []

    thr = rel_thresh * mx
    n = len(profile)
    end = search_end if (search_end is not None) else n

    peaks = []
    for i in range(2, min(end, n - 2)):
        if profile[i] >= thr and profile[i] > profile[i - 1] and profile[i] > profile[i + 1]:
            peaks.append(i)
    return peaks

def measure_gap_edges(seam_roi_gray):
    """
    Measures gap using two strong vertical edges within the seam ROI:
      - controller right face edge
      - connector left face edge
    Returns:
      gap_px (int), xA (controller edge idx), xB (connector edge idx), profile_s (smoothed profile)
    """
    roi = cv2.GaussianBlur(seam_roi_gray, (5, 5), 0)

    # Sobel X emphasizes vertical edges
    sx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    edge = np.abs(sx)

    # Collapse to 1D profile by summing over rows
    profile = edge.sum(axis=0)

    # Smooth
    k = EDGE_PROFILE_SMOOTH_K
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    profile_s = np.convolve(profile, kernel, mode="same")

    w = len(profile_s)
    if w < 10:
        return 0, None, None, profile_s

    # Search peaks in most relevant band to avoid unrelated edges:
    # You can adjust search_end if needed.
    search_end = int(0.75 * w)

    peaks = find_peaks_1d(profile_s, rel_thresh=PEAK_REL_THRESH, search_end=search_end)
    if len(peaks) < 2:
        return 0, None, None, profile_s

    # Choose a pair of peaks that are close enough to represent the gap.
    # We score by edge strength and penalize large separation.
    best_pair = None
    best_score = -1e9

    # Consider top candidates by strength
    peaks_sorted = sorted(peaks, key=lambda i: profile_s[i], reverse=True)[:10]

    # Evaluate pairs
    for i in range(len(peaks_sorted)):
        for j in range(i + 1, len(peaks_sorted)):
            xA = peaks_sorted[i]
            xB = peaks_sorted[j]
            if xB < xA:
                xA, xB = xB, xA
            sep = xB - xA
            if 2 <= sep <= PAIR_MAX_SEP_PX:
                score = float(profile_s[xA] + profile_s[xB]) - 0.01 * sep
                if score > best_score:
                    best_score = score
                    best_pair = (xA, xB)

    if best_pair is None:
        return 0, None, None, profile_s

    xA, xB = best_pair
    gap_px = int(max(0, xB - xA))
    return gap_px, xA, xB, profile_s


# ============================================================
# CAMERA STARTUP
# ============================================================
def start_camera():
    """
    Starts Basler camera using pypylon and returns (camera, converter).
    """
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # Recommended to disable auto once exposure is tuned in pylon Viewer.
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
def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[CFG] Saved {CONFIG_PATH}")

def load_config():
    if not os.path.exists(CONFIG_PATH) or not os.path.exists(TEMPLATE_PATH):
        return None
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ============================================================
# CALIBRATION
# ============================================================
def calibrate(camera, converter):
    """
    Calibration collects:
      - Template ROI (for part-present detection and location)
      - Seam ROI offset relative to template match (for gap measurement)
    """
    win = "CALIBRATION"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    selector = ROISelector(win)
    cv2.setMouseCallback(win, selector.callback)

    mode = "TEMPLATE"  # TEMPLATE or SEAM
    template_roi = None  # (x,y,w,h)
    seam_roi = None      # (x,y,w,h)

    print("\n--- CALIBRATION ---")
    print("1) Put a GOOD part under camera.")
    print("2) Press T and drag a TEMPLATE ROI that includes BOTH:")
    print("   - stable controller features AND connector area (not connector only).")
    print("3) Press S and drag a SEAM ROI that spans BOTH sides of the interface")
    print("   (some controller area on left + some connector face on right).")
    print("4) Press W to write config + template.png.")
    print("ESC to exit.\n")

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue
        frame = converter.Convert(grab).GetArray()
        grab.Release()

        disp = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Auto-accept ROI after mouse drag
        if selector.last_roi is not None:
            if mode == "TEMPLATE":
                template_roi = selector.last_roi
            else:
                seam_roi = selector.last_roi

        # Draw current ROIs
        if template_roi is not None:
            x, y, w, h = template_roi
            cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(disp, "TEMPLATE ROI", (x, max(20, y - 8)), FONT, 0.9, (255, 0, 0), 2)

        if seam_roi is not None:
            x, y, w, h = seam_roi
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(disp, "SEAM ROI", (x, max(20, y - 8)), FONT, 0.9, (0, 255, 255), 2)

        # Help text
        cv2.putText(disp, f"Mode: {mode}  (T=Template, S=Seam, W=Write, ESC=Exit)",
                    (20, 40), FONT, 0.9, (255, 255, 255), 2)
        cv2.putText(disp, "Drag ROI with mouse. ROI auto-accepts on mouse release.",
                    (20, 75), FONT, 0.75, (255, 255, 255), 2)

        cv2.imshow(win, disp)
        k = cv2.waitKey(1) & 0xFF

        if k == KEY_ESC:
            cv2.destroyWindow(win)
            return False

        if k in (ord('t'), ord('T')):
            mode = "TEMPLATE"

        if k in (ord('s'), ord('S')):
            mode = "SEAM"

        if k in (ord('w'), ord('W')):
            if template_roi is None or seam_roi is None:
                print("[CAL] Need BOTH template ROI and seam ROI before writing.")
                continue

            tx, ty, tw, th = template_roi
            template_img = frame[ty:ty + th, tx:tx + tw].copy()
            cv2.imwrite(TEMPLATE_PATH, template_img)
            print(f"[CAL] Saved {TEMPLATE_PATH}")

            sx, sy, sw, sh = seam_roi

            # Store seam ROI as offset relative to template ROI top-left
            seam_offset = dict(
                x=int(sx - tx),
                y=int(sy - ty),
                w=int(sw),
                h=int(sh)
            )

            cfg = dict(
                template_roi=dict(x=int(tx), y=int(ty), w=int(tw), h=int(th)),
                seam_offset_from_template=seam_offset,
                # thresholds
                gap_fail_at_or_above_px=int(GAP_FAIL_AT_OR_ABOVE_PX),
                fail_consec_to_trip=int(FAIL_CONSEC_TO_TRIP),
                use_max_gap_latch=bool(USE_MAX_GAP_LATCH),
                match_min_present=float(MATCH_MIN_PRESENT),
                match_min_leave=float(MATCH_MIN_LEAVE),
                enter_present_frames=int(ENTER_PRESENT_FRAMES),
                leave_frames=int(LEAVE_FRAMES),
                # edge settings
                peak_rel_thresh=float(PEAK_REL_THRESH),
                pair_max_sep_px=int(PAIR_MAX_SEP_PX),
                edge_profile_smooth_k=int(EDGE_PROFILE_SMOOTH_K),
            )

            save_config(cfg)
            cv2.destroyWindow(win)
            return True


# ============================================================
# RUN TRIAL
# ============================================================
def run_trial(camera, converter, cfg):
    template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError("template.png not found. Run calibration first.")

    th, tw = template.shape[:2]
    seam_off = cfg["seam_offset_from_template"]

    # load tuned settings
    gap_thr = cfg["gap_fail_at_or_above_px"]
    fail_consec_to_trip = cfg["fail_consec_to_trip"]
    use_max_gap = cfg["use_max_gap_latch"]

    match_present = cfg["match_min_present"]
    match_leave = cfg["match_min_leave"]
    enter_frames = cfg["enter_present_frames"]
    leave_frames = cfg["leave_frames"]

    # apply edge params to globals (or use cfg directly)
    global PEAK_REL_THRESH, PAIR_MAX_SEP_PX, EDGE_PROFILE_SMOOTH_K
    PEAK_REL_THRESH = cfg.get("peak_rel_thresh", PEAK_REL_THRESH)
    PAIR_MAX_SEP_PX = cfg.get("pair_max_sep_px", PAIR_MAX_SEP_PX)
    EDGE_PROFILE_SMOOTH_K = cfg.get("edge_profile_smooth_k", EDGE_PROFILE_SMOOTH_K)

    cv2.namedWindow(WINDOW_MAIN, cv2.WINDOW_NORMAL)

    state = "WAIT_FOR_PART"
    present_ct = 0
    leave_ct = 0

    bad_latched = False
    fail_consec = 0
    max_gap_seen = 0
    saved_this_part = False

    show_debug = True

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            grab.Release()
            continue

        frame = converter.Convert(grab).GetArray()
        grab.Release()

        H, W = frame.shape

        # ---- Template match (full frame) ----
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, match_score, _, match_loc = cv2.minMaxLoc(res)
        mx, my = match_loc

        # ---- Keyboard ----
        k = cv2.waitKey(1) & 0xFF

        if k == KEY_ESC:
            break

        if k == KEY_RESET:
            bad_latched = False
            fail_consec = 0
            max_gap_seen = 0
            saved_this_part = False
            print("[RESET] Latched BAD cleared (dev reset).")

        if k == KEY_TOGGLE_DEBUG:
            show_debug = not show_debug
            if not show_debug:
                try:
                    cv2.destroyWindow(WINDOW_DEBUG)
                except Exception:
                    pass

        if k == KEY_RECAL:
            # quick re-calibration shortcut
            print("[INFO] Recalibration requested. Exiting run mode.")
            break

        # ---- State machine: part present ----
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
                max_gap_seen = 0
                saved_this_part = False
                # Note: don't auto-clear bad_latched here; dev reset controls it
        else:
            # state == INSPECT
            if match_score < match_leave:
                leave_ct += 1
            else:
                leave_ct = 0

        # ---- Build annotated debug image ----
        annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Draw template match
        cv2.rectangle(annotated, (mx, my), (mx + tw, my + th), (255, 0, 0), 2)
        cv2.putText(annotated, f"Match: {match_score:.2f}",
                    (mx, max(20, my - 10)), FONT, 0.8, (255, 0, 0), 2)

        gap_px = 0
        xA = xB = None
        rx = ry = rx2 = ry2 = None

        if state == "INSPECT":
            # Seam ROI relative to template match
            rx = int(clamp(mx + seam_off["x"], 0, W - 1))
            ry = int(clamp(my + seam_off["y"], 0, H - 1))
            rw = int(seam_off["w"])
            rh = int(seam_off["h"])
            rx2 = int(clamp(rx + rw, 0, W))
            ry2 = int(clamp(ry + rh, 0, H))

            roi = frame[ry:ry2, rx:rx2]
            cv2.rectangle(annotated, (rx, ry), (rx2, ry2), (0, 255, 255), 2)

            # --- GAP measurement (edge-to-edge) ---
            gap_px, xA, xB, prof = measure_gap_edges(roi)
            max_gap_seen = max(max_gap_seen, gap_px)

            # Conservative decision logic
            fail_frame = (gap_px >= gap_thr)

            if fail_frame:
                fail_consec += 1
            else:
                fail_consec = 0

            if fail_consec >= fail_consec_to_trip:
                bad_latched = True

            if use_max_gap and (max_gap_seen >= gap_thr):
                bad_latched = True

            # If part leaves, re-arm
            if leave_ct >= leave_frames:
                state = "WAIT_FOR_PART"
                leave_ct = 0
                present_ct = 0

        # ---- Save FAIL images once per part when BAD first latches ----
        if bad_latched and (not saved_this_part) and state == "INSPECT":
            info = f"gap_px={gap_px}, max_gap_seen={max_gap_seen}, match_score={match_score:.3f}"
            save_fail_images(frame, annotated, info_text=info)
            saved_this_part = True

        # ---- Big Green/Red screen ----
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        if bad_latched:
            bg[:] = (0, 0, 255)  # RED
            label = "BAD"
            txt_color = (255, 255, 255)
        else:
            bg[:] = (0, 255, 0)  # GREEN
            label = "GOOD"
            txt_color = (0, 0, 0)

        cv2.putText(bg, label, (50, 180), FONT, 4.5, txt_color, 12)
        cv2.putText(bg, f"State: {state}", (50, 260), FONT, 1.4, txt_color, 4)

        if state == "INSPECT":
            cv2.putText(bg, f"Gap: {gap_px}px  MaxGapSeen: {max_gap_seen}px  (Fail>= {gap_thr})",
                        (50, 320), FONT, 1.2, txt_color, 4)
            cv2.putText(bg, f"MatchScore: {match_score:.2f}", (50, 370),
                        FONT, 1.2, txt_color, 4)
        else:
            cv2.putText(bg, f"Waiting for part... MatchScore: {match_score:.2f}", (50, 320),
                        FONT, 1.2, txt_color, 4)

        cv2.putText(bg, "R=Reset  D=Debug  C=Recalibrate  ESC=Quit",
                    (50, H - 50), FONT, 1.1, txt_color, 4)

        cv2.imshow(WINDOW_MAIN, bg)

        # ---- Debug window (optional but strongly recommended) ----
        if show_debug:
            dbg = annotated.copy()
            if state == "INSPECT" and xA is not None and xB is not None:
                # Draw detected edges within seam ROI
                cv2.line(dbg, (rx + xA, ry), (rx + xA, ry2), (0, 0, 255), 2)   # controller edge
                cv2.line(dbg, (rx + xB, ry), (rx + xB, ry2), (0, 255, 0), 2)   # connector edge
                cv2.putText(dbg, f"gap_px={gap_px} (Fail>= {gap_thr})",
                            (30, 60), FONT, 1.1,
                            (0, 0, 255) if gap_px >= gap_thr else (0, 255, 0), 3)

            cv2.putText(dbg, "DEBUG: template box (blue), seam ROI (yellow), edges (red/green)",
                        (30, H - 30), FONT, 0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW_DEBUG, dbg)

    # If user pressed C (recal), return special code
    return (k == KEY_RECAL)


# ============================================================
# MAIN
# ============================================================
def main():
    print("NOTE: Close Basler pylon Viewer before running this script.")
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
                recal = run_trial(camera, converter, cfg)
                if recal:
                    # delete config/template to force calibration again
                    print("[INFO] Recalibration requested.")
                    try:
                        if os.path.exists(CONFIG_PATH):
                            os.remove(CONFIG_PATH)
                        if os.path.exists(TEMPLATE_PATH):
                            os.remove(TEMPLATE_PATH)
                    except Exception:
                        pass
                    continue
                else:
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





