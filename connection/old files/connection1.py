import cv2
import numpy as np
from pypylon import pylon

# ---------------------------
# FIXED ROI (ADJUST ONCE)
# ---------------------------
ROI_X = 1100
ROI_Y = 950
ROI_W = 350
ROI_H = 300

PASS_THRESHOLD = 8   # pixels of horizontal shift allowed


def measure_gap(gray):

    roi = gray[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

    # Detect vertical edges only
    sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    sobel_8u = np.uint8(abs_sobelx)

    # Threshold edges
    _, thresh = cv2.threshold(sobel_8u, 40, 255, cv2.THRESH_BINARY)

    # Sum columns to find strong vertical edges
    col_sum = np.sum(thresh, axis=0)

    # Require minimum edge strength
    edge_indices = np.where(col_sum > 2000)[0]

    if len(edge_indices) < 2:
        return None, None, None

    # Force left edge to be in left half
    left_candidates = edge_indices[edge_indices < ROI_W * 0.5]
    right_candidates = edge_indices[edge_indices > ROI_W * 0.5]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return None, None, None

    left_edge = left_candidates[-1]
    right_edge = right_candidates[0]

    gap = right_edge - left_edge

    return gap, left_edge, right_edge


def run():

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    reference_gap = None

    print("Press 'g' to set reference")
    print("Press 'q' to quit")

    while camera.IsGrabbing():

        grab = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab.GrabSucceeded():

            image = converter.Convert(grab)
            frame = image.GetArray()

            gray = frame.copy()

            gap, left_edge, right_edge = measure_gap(gray)

            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Draw ROI
            cv2.rectangle(display,
                          (ROI_X, ROI_Y),
                          (ROI_X+ROI_W, ROI_Y+ROI_H),
                          (0,255,255), 2)

            if gap is not None:

                # Draw detected edges
                cv2.line(display,
                         (ROI_X + left_edge, ROI_Y),
                         (ROI_X + left_edge, ROI_Y+ROI_H),
                         (0,255,0), 2)

                cv2.line(display,
                         (ROI_X + right_edge, ROI_Y),
                         (ROI_X + right_edge, ROI_Y+ROI_H),
                         (0,0,255), 2)

                if reference_gap is not None:

                    shift = gap - reference_gap

                    if abs(shift) <= PASS_THRESHOLD:
                        status = "PASS"
                        color = (0,255,0)
                    else:
                        status = "FAIL"
                        color = (0,0,255)

                    cv2.putText(display,
                                f"Shift: {shift}px",
                                (50,100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (255,255,255),
                                3)

                    cv2.putText(display,
                                status,
                                (50,170),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2,
                                color,
                                4)

            cv2.imshow("Inspection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('g') and gap is not None:
                reference_gap = gap
                print("Reference set to:", reference_gap)

            elif key == ord('q'):
                break

        grab.Release()

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()

