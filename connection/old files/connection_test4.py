import cv2
import numpy as np
from pypylon import pylon

# ==============================
# SETTINGS
# ==============================

THRESHOLD = 60
PASS_THRESHOLD = 6   # px gap allowed

# ROI around connector area (adjust if needed)
ROI_X1 = 700
ROI_X2 = 1100
ROI_Y1 = 350
ROI_Y2 = 650

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

        # Safe ROI bounds
        x1 = max(0, min(ROI_X1, width-1))
        x2 = max(0, min(ROI_X2, width))
        y1 = max(0, min(ROI_Y1, height-1))
        y2 = max(0, min(ROI_Y2, height))

        roi = frame[y1:y2, x1:x2]

        # ==============================
        # FIND DARK GAP STRIP
        # ==============================

        # Average each column brightness
        column_mean = np.mean(roi, axis=0)

        # Find dark columns
        dark_columns = np.where(column_mean < THRESHOLD)[0]

        gap = 0

        if len(dark_columns) > 0:
            gap = dark_columns[-1] - dark_columns[0]

        # ==============================
        # PASS / FAIL
        # ==============================

        if gap <= PASS_THRESHOLD:
            status = "PASS"
            color = (0, 255, 0)
        else:
            status = "FAIL"
            color = (0, 0, 255)

        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Draw ROI
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)

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

        cv2.imshow("Inspection", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()






