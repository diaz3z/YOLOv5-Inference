from time import time
import cv2

class Camera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)  # Open the webcam

    def __del__(self):
        self.video.release()  # Release the webcam when done

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            print("Failed to capture frame")  # Debugging output
            return None  # Return None if frame capture fails

        return frame  # Return the captured frame