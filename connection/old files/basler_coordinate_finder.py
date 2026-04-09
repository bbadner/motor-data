import cv2
import numpy as np
from pypylon import pylon
import time

clicks = []  # store (x,y)

def mouse_callback(event, x, y, flags, param):
    global clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))

        # Write to file immediately (no console needed)
        with open("click_points.txt", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  X={x}, Y={y}\n")

# Camera setup
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

cv2.namedWindow("View", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("View", mouse_callback)

print("Click on the image window. ESC=quit, R=reset. Points are saved to click_points.txt")

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = np.uint8(image.GetArray())

        # Draw clicks
        for i, (cx, cy) in enumerate(clicks[-2:], start=1):
            cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
            cv2.putText(frame, f"P{i}: ({cx},{cy})",
                        (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # If we have 2 points, draw ROI rectangle too
        if len(clicks) >= 2:
            (x1, y1) = clicks[-2]
            (x2, y2) = clicks[-1]
            tl = (min(x1, x2), min(y1, y2))
            br = (max(x1, x2), max(y1, y2))
            cv2.rectangle(frame, tl, br, (255, 255, 255), 2)

            roi_x, roi_y = tl
            roi_w = br[0] - tl[0]
            roi_h = br[1] - tl[1]
            cv2.putText(frame, f"ROI: X={roi_x} Y={roi_y} W={roi_w} H={roi_h}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("View", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:      # ESC
            break
        elif k in (ord('r'), ord('R')):
            clicks = []  # reset

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
