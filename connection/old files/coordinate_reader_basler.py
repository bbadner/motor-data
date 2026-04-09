from pypylon import pylon
import cv2

current_pos = (0, 0)
scale_factor = 1.0

def mouse_callback(event, x, y, flags, param):
    global current_pos, scale_factor
    # Convert display coords back to real image coords
    real_x = int(x / scale_factor)
    real_y = int(y / scale_factor)
    current_pos = (real_x, real_y)

def run_basler():

    global scale_factor

    print("Opening Basler camera...")

    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    cv2.namedWindow("Basler Coordinate Reader")
    cv2.setMouseCallback("Basler Coordinate Reader", mouse_callback)

    print("\nControls:")
    print("  Move mouse to read X/Y")
    print("  Press 's' to freeze/unfreeze frame")
    print("  Press 'q' to quit\n")

    freeze = False
    frozen_frame = None

    while camera.IsGrabbing():

        if not freeze:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                image = converter.Convert(grabResult)
                frame = image.GetArray()
                grabResult.Release()
            else:
                grabResult.Release()
                continue
        else:
            frame = frozen_frame.copy()

        h, w = frame.shape[:2]

        # -------- SCALE FOR DISPLAY --------
        max_display_width = 1200
        if w > max_display_width:
            scale_factor = max_display_width / w
        else:
            scale_factor = 1.0

        display = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        disp_h, disp_w = display.shape[:2]

        x, y = current_pos

        # Draw crosshair (scaled)
        cv2.line(display, (int(x * scale_factor), 0),
                 (int(x * scale_factor), disp_h), (0,255,0), 1)

        cv2.line(display, (0, int(y * scale_factor)),
                 (disp_w, int(y * scale_factor)), (0,255,0), 1)

        cv2.putText(display,
                    f"Resolution: {w} x {h}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2)

        cv2.putText(display,
                    f"X: {x}  Y: {y}",
                    (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.imshow("Basler Coordinate Reader", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            freeze = not freeze
            if freeze:
                frozen_frame = frame.copy()
                print("Frame frozen")
            else:
                print("Live resumed")

    camera.StopGrabbing()
    camera.Close()
    cv2.destroyAllWindows()
    print("Camera closed.")

if __name__ == "__main__":
    run_basler()
