import cv2
import numpy as np
from pypylon import pylon

# ==============================
# SETTINGS
# ==============================

THRESHOLD_NON_BLACK = 40
PASS_THRESHOLD = 3  # pixels

# Vertical scan area (adjust if needed)
SCAN_TOP = 200
SCAN_BOTTOM = 800

# ==============================
# START CAMERA
# ==============================

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print("Camera started.")

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():

        image = converter.Convert(grabResult)
        frame = image.GetArray()

        height, width = frame.shape

        # Make sure scan region is valid
        scan_top = max(0, min(SCAN_TOP, height - 1))
        scan_bottom = max(0, min(SCAN_BOTTOM, height))

        # Use rightmost area for scanning
        x = width - 120

        # Make sure x is valid
        if x >= width:
            x = width - 1

        # ==============================
        # GAP MEASUREMENT
        # ==============================

        gap = 0

        for y in range(scan_top, scan_bottom):
            if frame[y, x] > THRESHOLD_NON_BLACK:
                gap = y - scan_top
                break

        # ==============================
        # PASS / FAIL
        # ==============================

        if gap <= PASS_THRESHOLD:
            status = "PASS"
            color = (0, 255, 0)
        else:
            status = "FAIL"
            color = (0, 0, 255)

        # ==============================
        # CONVERT FOR DISPLAY
        # ==============================

        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Draw scan line
        cv2.line(display, (x, scan_top), (x, scan_bottom), (255, 0, 0), 2)

        # Overlay text
        cv2.putText(display, f"Gap: {gap}px",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color,
                    4)

        cv2.putText(display, status,
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    color,
                    4)

        # ==============================
        # SHOW RAW FRAME (NO RESIZE)
        # ==============================

        cv2.imshow("Inspection", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()







