import cv2
import numpy as np
from pypylon import pylon

# ==============================
# FIXED HOUSING EDGE (LOCKED)
# ==============================
HOUSING_RIGHT_EDGE_X = 1077
PASS_TOLERANCE = 2   # 0–2 px = PASS

# ==============================
# BASLER CAMERA SETUP
# ==============================
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print("Camera Started")

# ==============================
# MAIN LOOP
# ==============================
while camera.IsGrabbing():

    grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab.GrabSucceeded():

        image = converter.Convert(grab)
        frame = image.GetArray()

        display = frame.copy()

        # ---------------------------------------------------
        # DEFINE SEARCH ROI (adjust width if needed)
        # ---------------------------------------------------
        roi_x1 = 1000
        roi_x2 = 1250
        roi_y1 = 900
        roi_y2 = 1300

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Strong threshold to isolate connector face
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

        # Clean noise
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        connector_left_x = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)

            connector_left_x = roi_x1 + x

            # Draw box
            cv2.rectangle(display,
                          (roi_x1 + x, roi_y1 + y),
                          (roi_x1 + x + w, roi_y1 + y + h),
                          (0,255,255), 2)

            # Draw connector edge line
            cv2.line(display,
                     (connector_left_x, roi_y1),
                     (connector_left_x, roi_y2),
                     (0,0,255), 2)

        # Draw housing fixed reference line
        cv2.line(display,
                 (HOUSING_RIGHT_EDGE_X, 0),
                 (HOUSING_RIGHT_EDGE_X, frame.shape[0]),
                 (255,0,0), 2)

        # ---------------------------------------------------
        # GAP CALCULATION
        # ---------------------------------------------------
        gap = 0

        if connector_left_x is not None:
            gap = connector_left_x - HOUSING_RIGHT_EDGE_X

        status = "PASS"
        color = (0,255,0)

        if gap > PASS_TOLERANCE:
            status = "FAIL"
            color = (0,0,255)

        # Display info
        cv2.putText(display, f"Gap: {gap}px",
                    (40,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    color,
                    3)

        cv2.putText(display, status,
                    (40,140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    4)

        # ---------------------------------------
        # Resize for viewing (CRITICAL FIX)
        # ---------------------------------------
        scale_percent = 50
        width = int(display.shape[1] * scale_percent / 100)
        height = int(display.shape[0] * scale_percent / 100)
        resized = cv2.resize(display, (width, height))

        cv2.imshow("Inspection", resized)

    grab.Release()

    key = cv2.waitKey(1)
    if key == 27:  # ESC to quit
        break

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
