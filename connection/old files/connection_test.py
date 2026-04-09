import cv2
import numpy as np
from pypylon import pylon

# -----------------------------
# Display scaling
# -----------------------------
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800

# -----------------------------
# FIXED ROI (full resolution)
# -----------------------------
ROI_X = 1100
ROI_Y = 900
ROI_W = 400
ROI_H = 350

THRESHOLD_PX = 15  # adjust later

def measure_horizontal_gap(gray_roi):
    sobelx = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = np.absolute(sobelx)
    sobelx = np.uint8(sobelx)

    column_profile = np.mean(sobelx, axis=0)

    # Find strong edges
    edges = np.where(column_profile > 40)[0]

    if len(edges) < 2:
        return None, None, None

    left_edge = edges[0]
    right_edge = edges[-1]

    gap = right_edge - left_edge

    return gap, left_edge, right_edge


def run():
    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    while camera.IsGrabbing():
        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab.GrabSucceeded():
            image = converter.Convert(grab)
            frame = image.GetArray()

            # ROI
            roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

            gap, left, right = measure_horizontal_gap(roi)

            display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            scale_x = frame.shape[1] / DISPLAY_WIDTH
            scale_y = frame.shape[0] / DISPLAY_HEIGHT

            # Draw ROI
            dx = int(ROI_X / scale_x)
            dy = int(ROI_Y / scale_y)
            dw = int(ROI_W / scale_x)
            dh = int(ROI_H / scale_y)

            cv2.rectangle(display, (dx, dy),
                          (dx + dw, dy + dh), (0,255,255), 2)

            if gap is not None:
                # Draw edge lines
                lx = int((ROI_X + left) / scale_x)
                rx = int((ROI_X + right) / scale_x)

                cv2.line(display, (lx, dy), (lx, dy+dh), (0,255,0), 2)
                cv2.line(display, (rx, dy), (rx, dy+dh), (0,0,255), 2)

                status = "PASS" if gap < THRESHOLD_PX else "FAIL"
                color = (0,255,0) if gap < THRESHOLD_PX else (0,0,255)

                cv2.putText(display, f"Gap: {gap}px",
                            (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, color, 3)

                cv2.putText(display, status,
                            (50, 140),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            2, color, 4)

            cv2.imshow("Inspection", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        grab.Release()

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
