from pypylon import pylon
import cv2
import numpy as np

def open_locked_basler_camera():
    # Connect to first Basler camera found
    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()

    print("Connected to:", camera.GetDeviceInfo().GetModelName())

    # -------------------------------
    # 🔒 HARD CAMERA LOCK SETTINGS
    # -------------------------------

    # Disable all auto features
    if camera.ExposureAuto.IsWritable():
        camera.ExposureAuto.SetValue("Off")

    if camera.GainAuto.IsWritable():
        camera.GainAuto.SetValue("Off")

    if camera.BalanceWhiteAuto.IsWritable():
        camera.BalanceWhiteAuto.SetValue("Off")

    # Set fixed exposure (microseconds)
    camera.ExposureTime.SetValue(8000.0)   # <-- adjust later

    # Set fixed gain (dB)
    camera.Gain.SetValue(0.0)

    # Lock frame rate
    if camera.AcquisitionFrameRateEnable.IsWritable():
        camera.AcquisitionFrameRateEnable.SetValue(True)
        camera.AcquisitionFrameRate.SetValue(20.0)

    # Pixel format (critical for OpenCV)
    camera.PixelFormat.SetValue("Mono8")

    # -------------------------------
    # Image format converter
    # -------------------------------
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_Mono8
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    return camera, converter


def live_view():
    camera, converter = open_locked_basler_camera()

    try:
        while camera.IsGrabbing():
            grab = camera.RetrieveResult(2000, pylon.TimeoutHandling_ThrowException)

            if grab.GrabSucceeded():
                img = converter.Convert(grab).GetArray()

                # Optional: normalize for display only
                display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

                cv2.imshow("Basler Locked Live View", display)

            grab.Release()

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    finally:
        camera.StopGrabbing()
        camera.Close()
        cv2.destroyAllWindows()
        print("Camera closed cleanly")


if __name__ == "__main__":
    live_view()
