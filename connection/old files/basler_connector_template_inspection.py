import cv2
import numpy as np
from pypylon import pylon

# ----------------------------
# USER SETTINGS
# ----------------------------

TEMPLATE_PATH = "template_good.png"

# ROI around connector (adjust once and lock)
#ROI_X = 950
#ROI_Y = 800
#ROI_W = 250
#ROI_H = 250
ROI_X = 1500
ROI_Y = 900
ROI_W = 300
ROI_H = 300



# Threshold for failure (pixels shift)
FAIL_THRESHOLD = 5  # adjust after testing

REFERENCE_X = None

# ----------------------------
# LOAD TEMPLATE
# ----------------------------

template = cv2.imread(TEMPLATE_PATH, 0)
if template is None:
    print("ERROR: template_good.png not found.")
    exit()

t_h, t_w = template.shape

# ----------------------------
# BASLER CAMERA INIT
# ----------------------------

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_Mono8
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

print("Camera running... Press ESC to exit.")

# ----------------------------
# MAIN LOOP
# ----------------------------

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        frame = image.GetArray()

        # Extract ROI
        roi = frame[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

        # Blur to stabilize
        roi_blur = cv2.GaussianBlur(roi, (5,5), 0)

        # Template Matching
        result = cv2.matchTemplate(roi_blur, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        match_x = max_loc[0]

        # Lock reference automatically first frame
        if REFERENCE_X is None:
            REFERENCE_X = match_x
            print(f"Reference X locked at: {REFERENCE_X}")

        shift = match_x - REFERENCE_X

        # Determine status
        if abs(shift) > FAIL_THRESHOLD:
            status = "FAIL"
            color = (0, 0, 255)
        else:
            status = "GOOD"
            color = (0, 255, 0)

        # Draw match rectangle
        top_left = (ROI_X + match_x, ROI_Y + max_loc[1])
        bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
        cv2.rectangle(frame, top_left, bottom_right, (0,255,255), 2)

        # Overlay text
        cv2.putText(frame, f"Shift: {shift}px",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)

        cv2.putText(frame, status,
                    (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, color, 4)

        cv2.imshow("Inspection", frame)

        if cv2.waitKey(1) == 27:
            break

    grabResult.Release()

camera.StopGrabbing()
camera.Close()
cv2.destroyAllWindows()
