import cv2
import numpy as np
import configuration

class Captureimage():
    def __init__(self):
        self.camera_resolution = configuration.resolution
        self.background_image = None
        self.lower_value = configuration.lower_value
        self.higher_value = configuration.higher_value

    def set_resolution(self):
        vid = cv2.VideoCapture(0)
        ret, frame = vid.read()
        self.camera_resolution = (frame.shape[0]. frame.shape[1])

    def select_background_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        self.background_image = cv2.resize(img, self.camera_resolution)
        return img

    def add_background(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self.lower_value, self.higher_value)
        res = cv2.bitwise_and(frame, frame, mask = mask)
        filtered = frame - res
        filtered = np.where(filtered == 0, self.background_image, filtered)
        return filtered

    def capture_webcam(self):
        vid = cv2.VideoCapture(0)
        while(True):
            ret, frame = vid.read()
            modified_frame = self.add_background(frame)
            cv2.imshow("Masked image", modified_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    image_capture  = Captureimage()
    image_capture.select_background_image("../files/white_bg.jpg")
    image_capture.capture_webcam()