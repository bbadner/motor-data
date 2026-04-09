"""
Gap inspection for a fixed Basler camera view.

Decision logic:
- Detect housing edge.
- Check for dark-pixel presence in a narrow window immediately to its right.
- FAIL if dark region is present (gap), PASS otherwise.
"""

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from pypylon import pylon
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pypylon is required. Install with: pip install pypylon") from exc


ROI_CONFIG_PATH = Path("roi_config.json")


@dataclass
class InspectionConfig:
    # Default ROI in full-frame pixel coordinates.
    roi_x: int = 460
    roi_y: int = 180
    roi_w: int = 130
    roi_h: int = 120

    # Housing search zone as ROI-width ratios.
    housing_search_start_ratio: float = 0.45
    housing_search_end_ratio: float = 0.72

    # Presence window right of housing edge.
    presence_window_px: int = 10  # 8-12 px typical

    # Dark-pixel presence test (fixed threshold, not adaptive).
    dark_pixel_threshold: int = 65
    dark_ratio_fail_threshold: float = 0.18

    # Housing-edge quality checks.
    blur_ksize: int = 5
    canny_low: int = 60
    canny_high: int = 140
    min_edge_strength: float = 0.01  # fraction of ROI height
    housing_min_gradient_strength: float = 6.0
    housing_min_vertical_run_px: int = 8
    housing_min_local_contrast: float = 3.0

    # Evaluate in mating band only.
    eval_ignore_top_ratio: float = 0.10
    eval_ignore_bottom_ratio: float = 0.05

    # Temporal smoothing on dark ratio.
    dark_ratio_median_window_frames: int = 5


@dataclass
class RoiEditorState:
    """State machine for interactive ROI editing."""

    edit_mode: bool = False
    dragging: bool = False
    frame_w: int = 0
    frame_h: int = 0
    drag_start: Optional[Tuple[int, int]] = None
    drag_end: Optional[Tuple[int, int]] = None
    candidate_roi: Optional[Tuple[int, int, int, int]] = None

    def set_frame_size(self, frame_w: int, frame_h: int) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h

    def start_edit_mode(self) -> None:
        self.edit_mode = True
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.candidate_roi = None
        print("ROI edit mode enabled. Drag with left mouse button to define ROI.")

    def stop_edit_mode(self) -> None:
        self.edit_mode = False
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.candidate_roi = None
        print("ROI edit mode disabled.")

    def on_mouse(self, event: int, x: int, y: int, _flags: int) -> None:
        if not self.edit_mode:
            return
        if self.frame_w <= 0 or self.frame_h <= 0:
            return

        x = max(0, min(x, self.frame_w - 1))
        y = max(0, min(y, self.frame_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)
            return

        if event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.drag_end = (x, y)
            roi = normalized_drag_rect(
                self.drag_start, self.drag_end, self.frame_w, self.frame_h
            )
            if roi is not None:
                self.candidate_roi = roi
                print(f"ROI candidate set: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
            else:
                print("ROI drag too small; drag a larger rectangle.")
            self.drag_start = None
            self.drag_end = None

    def drag_preview(self) -> Optional[Tuple[int, int, int, int]]:
        if not self.edit_mode or not self.dragging:
            return None
        return normalized_drag_rect(
            self.drag_start, self.drag_end, self.frame_w, self.frame_h, min_size_px=1
        )


def clamp_roi(frame_w: int, frame_h: int, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(1, min(w, frame_w - x))
    h = max(1, min(h, frame_h - y))
    return x, y, w, h


def normalized_drag_rect(
    start: Optional[Tuple[int, int]],
    end: Optional[Tuple[int, int]],
    frame_w: int,
    frame_h: int,
    min_size_px: int = 5,
) -> Optional[Tuple[int, int, int, int]]:
    if start is None or end is None or frame_w <= 0 or frame_h <= 0:
        return None

    x1, y1 = start
    x2, y2 = end
    x = min(x1, x2)
    y = min(y1, y2)
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    if w < min_size_px or h < min_size_px:
        return None

    return clamp_roi(frame_w, frame_h, x, y, w, h)


def load_roi_config(path: Path) -> Optional[Tuple[int, int, int, int]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        x = int(data["x"])
        y = int(data["y"])
        w = int(data["w"])
        h = int(data["h"])
        if w <= 0 or h <= 0:
            return None
        return x, y, w, h
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_roi_config(path: Path, roi: Tuple[int, int, int, int]) -> None:
    x, y, w, h = roi
    payload = {"x": x, "y": y, "w": w, "h": h}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_rightmost_housing_edge(
    gray: np.ndarray,
    edges: np.ndarray,
    grad_x: np.ndarray,
    cfg: InspectionConfig,
) -> Tuple[Optional[int], list[int], int, int]:
    """
    Collect valid housing edge candidates and return rightmost x.
    We keep all candidates for debug; final housing reference is max(candidates).
    """
    h, w = gray.shape[:2]
    y0 = int(round(cfg.eval_ignore_top_ratio * h))
    y1 = h - int(round(cfg.eval_ignore_bottom_ratio * h))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    x0 = int(round(cfg.housing_search_start_ratio * w))
    x1 = int(round(cfg.housing_search_end_ratio * w))
    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))

    min_edge_sum = cfg.min_edge_strength * (y1 - y0)
    candidates: list[int] = []

    for x in range(x0, x1):
        col_edge = edges[y0:y1, x] > 0
        if float(np.sum(col_edge)) < min_edge_sum:
            continue

        # Housing reference uses dark-to-bright polarity.
        col_polarity = grad_x[y0:y1, x] >= cfg.housing_min_gradient_strength
        col_valid = np.logical_and(col_edge, col_polarity)

        run = 0
        max_run = 0
        for v in col_valid:
            if v:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0
        if max_run < cfg.housing_min_vertical_run_px:
            continue

        xl = max(0, x - 2)
        xr = min(w - 1, x + 2)
        local_contrast = abs(
            float(np.mean(gray[y0:y1, xr])) - float(np.mean(gray[y0:y1, xl]))
        )
        if local_contrast < cfg.housing_min_local_contrast:
            continue

        candidates.append(x)

    if not candidates:
        return None, candidates, y0, y1
    return max(candidates), candidates, y0, y1


def detect_gap_presence(
    roi_bgr: np.ndarray, cfg: InspectionConfig
) -> Tuple[
    np.ndarray,
    Optional[int],
    list[int],
    int,
    int,
    int,
    int,
    float,
]:
    """
    Returns:
    - edges image (for debug panel)
    - housing_edge_x
    - housing candidate x list
    - eval band y0, y1
    - presence window start/end x
    - dark ratio in presence window
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)
    edges = cv2.Canny(blur, cfg.canny_low, cfg.canny_high)
    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)

    housing_edge_x, housing_candidates, y0, y1 = find_rightmost_housing_edge(
        gray, edges, grad_x, cfg
    )

    if housing_edge_x is None:
        return edges, None, housing_candidates, y0, y1, 0, 0, 0.0

    h, w = gray.shape[:2]
    presence_start_x = max(0, min(w - 1, housing_edge_x + 1))
    presence_end_x = max(
        presence_start_x + 1,
        min(w, presence_start_x + max(1, cfg.presence_window_px)),
    )

    win = gray[y0:y1, presence_start_x:presence_end_x]
    if win.size == 0:
        dark_ratio = 0.0
    else:
        dark_ratio = float(np.mean(win < cfg.dark_pixel_threshold))

    return (
        edges,
        housing_edge_x,
        housing_candidates,
        y0,
        y1,
        presence_start_x,
        presence_end_x,
        dark_ratio,
    )


def configure_camera(camera: "pylon.InstantCamera") -> None:
    camera.Open()
    if hasattr(camera, "AcquisitionMode"):
        camera.AcquisitionMode.SetValue("Continuous")
    if hasattr(camera, "TriggerMode"):
        camera.TriggerMode.SetValue("Off")
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


def main() -> None:
    cfg = InspectionConfig()
    roi_editor = RoiEditorState()

    loaded_roi = load_roi_config(ROI_CONFIG_PATH)
    if loaded_roi is not None:
        cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h = loaded_roi
        print(
            "Loaded ROI from roi_config.json: "
            f"x={cfg.roi_x}, y={cfg.roi_y}, w={cfg.roi_w}, h={cfg.roi_h}"
        )
    else:
        print("No roi_config.json found. Using default ROI from script.")

    tl_factory = pylon.TlFactory.GetInstance()
    devices = tl_factory.EnumerateDevices()
    if not devices:
        raise SystemExit("No Basler camera found.")

    camera = pylon.InstantCamera(tl_factory.CreateFirstDevice())
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    configure_camera(camera)

    window_name = "Gap Inspection (R edit ROI | S save ROI | ESC/Q quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def on_mouse(event: int, x: int, y: int, flags: int, _param: object) -> None:
        roi_editor.on_mouse(event, x, y, flags)

    cv2.setMouseCallback(window_name, on_mouse)

    dark_ratio_history: deque[float] = deque(maxlen=max(1, cfg.dark_ratio_median_window_frames))

    try:
        while camera.IsGrabbing():
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if not grab_result.GrabSucceeded():
                grab_result.Release()
                continue

            image = converter.Convert(grab_result)
            frame = image.GetArray()
            grab_result.Release()

            fh, fw = frame.shape[:2]
            roi_editor.set_frame_size(fw, fh)
            rx, ry, rw, rh = clamp_roi(fw, fh, cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h)
            roi = frame[ry : ry + rh, rx : rx + rw]

            (
                edges,
                housing_x,
                housing_candidates,
                y0,
                y1,
                presence_start_x,
                presence_end_x,
                dark_ratio,
            ) = detect_gap_presence(roi, cfg)

            # Main overlays.
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"ROI: x={rx}, y={ry}, w={rw}, h={rh}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Presence-window decision.
            status_text = "FAIL"
            status_color = (0, 0, 255)
            result_text = "dark_ratio: N/A"

            if housing_x is not None and presence_end_x > presence_start_x:
                dark_ratio_history.append(dark_ratio)
                dark_ratio_med = float(np.median(np.array(dark_ratio_history, dtype=np.float32)))
                is_fail = dark_ratio_med >= cfg.dark_ratio_fail_threshold
                status_text = "FAIL" if is_fail else "PASS"
                status_color = (0, 0, 255) if is_fail else (0, 220, 0)
                result_text = (
                    f"dark_ratio: {dark_ratio:.3f} med:{dark_ratio_med:.3f} "
                    f"(fail>={cfg.dark_ratio_fail_threshold:.3f})"
                )
            else:
                dark_ratio_history.clear()
                result_text = "dark_ratio: N/A (housing missing/window invalid)"

            cv2.putText(
                frame,
                result_text,
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                status_color,
                3,
                cv2.LINE_AA,
            )

            mode_text = (
                "ROI edit mode: ON (drag to set, S to save)"
                if roi_editor.edit_mode
                else "Press R to enter ROI edit mode"
            )
            cv2.putText(
                frame,
                mode_text,
                (10, 118),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            # Debug text.
            debug_housing = f"housing_edge_x: {housing_x if housing_x is not None else 'N/A'}"
            debug_win = (
                f"presence_window_x: [{presence_start_x}, {presence_end_x})"
                if housing_x is not None
                else "presence_window_x: N/A"
            )
            debug_dark = f"dark_ratio: {dark_ratio:.3f}"
            cv2.putText(
                frame,
                debug_housing,
                (10, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                debug_win,
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                debug_dark,
                (10, 194),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # ROI edit preview.
            preview = roi_editor.drag_preview()
            if preview is not None:
                px, py, pw, ph = preview
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 180, 0), 2)
            elif roi_editor.edit_mode and roi_editor.candidate_roi is not None:
                cx, cy, cw, ch = roi_editor.candidate_roi
                cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 180, 0), 2)

            # Draw housing candidates (thin red debug).
            for x in housing_candidates:
                cv2.line(frame, (rx + x, ry), (rx + x, ry + rh), (0, 0, 255), 1)

            # Draw selected housing edge (blue).
            if housing_x is not None:
                cv2.line(frame, (rx + housing_x, ry), (rx + housing_x, ry + rh), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    "HOUSING EDGE",
                    (rx + housing_x + 4, max(18, ry - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Presence window (yellow box) + evaluation vertical band.
            if housing_x is not None and presence_end_x > presence_start_x:
                cv2.rectangle(
                    frame,
                    (rx + presence_start_x, ry),
                    (rx + presence_end_x, ry + rh),
                    (0, 255, 255),
                    1,
                )
                cv2.putText(
                    frame,
                    "PRESENCE WINDOW",
                    (rx + presence_start_x + 2, min(fh - 10, ry + rh + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(
                    frame,
                    (rx + presence_start_x, ry + y0),
                    (rx + presence_end_x, ry + y1),
                    (0, 200, 255),
                    1,
                )

            # Edge debug panel.
            edges_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            panel_h = 160
            panel_w = int(panel_h * (rw / max(1, rh)))
            edges_vis = cv2.resize(edges_vis, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST)
            frame[10 : 10 + panel_h, fw - panel_w - 10 : fw - 10] = edges_vis
            cv2.rectangle(frame, (fw - panel_w - 10, 10), (fw - 10, 10 + panel_h), (255, 255, 255), 1)
            cv2.putText(
                frame,
                "ROI edges",
                (fw - panel_w - 10, 10 + panel_h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("r"), ord("R")):
                if roi_editor.edit_mode:
                    print("ROI edit mode is already enabled.")
                else:
                    roi_editor.start_edit_mode()

            elif key in (ord("s"), ord("S")):
                if not roi_editor.edit_mode:
                    print("Press R to enable ROI edit mode before saving.")
                elif roi_editor.candidate_roi is None:
                    print("No ROI candidate defined. Drag to create ROI first.")
                else:
                    cfg.roi_x, cfg.roi_y, cfg.roi_w, cfg.roi_h = roi_editor.candidate_roi
                    save_roi_config(ROI_CONFIG_PATH, roi_editor.candidate_roi)
                    print(
                        "ROI saved to roi_config.json: "
                        f"x={cfg.roi_x}, y={cfg.roi_y}, w={cfg.roi_w}, h={cfg.roi_h}"
                    )
                    roi_editor.stop_edit_mode()

            elif key in (27, ord("q"), ord("Q")):
                break

    finally:
        if camera.IsGrabbing():
            camera.StopGrabbing()
        if camera.IsOpen():
            camera.Close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
