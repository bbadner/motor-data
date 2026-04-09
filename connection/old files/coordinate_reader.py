import cv2

current_pos = (0, 0)

def mouse_callback(event, x, y, flags, param):
    global current_pos
    current_pos = (x, y)

def run_live_camera(camera_index=0):

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("ERROR: Could not open camera")
        return

    cv2.namedWindow("Live Coordinate Reader")
    cv2.setMouseCallback("Live Coordinate Reader", mouse_callback)

    print("\nControls:")
    print("  Move mouse to read coordinates")
    print("  Press 's' to freeze/unfreeze frame")
    print("  Press 'q' to quit\n")

    freeze = False
    frozen_frame = None

    while True:

        if not freeze:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = frozen_frame.copy()

        h, w = frame.shape[:2]

        display = frame.copy()
        x, y = current_pos

        # Draw crosshair
        cv2.line(display, (x, 0), (x, h), (0,255,0), 1)
        cv2.line(display, (0, y), (w, y), (0,255,0), 1)

        cv2.putText(display, f"X: {x}  Y: {y}",
                    (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,0),
                    2)

        cv2.imshow("Live Coordinate Reader", display)

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_camera()
