import cv2
import mediapipe
import numpy as np
import pdb

mp_drawing = mediapipe.solutions.drawing_utils
mp_selfie_segmentation = mediapipe.solutions.selfie_segmentation

class MediaPipeSegmentation():
    def __init__(self):
        self.resolution = (192, 192, 192)
        self.background_image = None

    def get_camera_resolution(self):
        cap = cv2.VideoCapture(0)
        self.camera_resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    def set_background_image(self, file_path):
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        print(self.camera_resolution)
        self.background_image = cv2.resize(img, self.camera_resolution)

    def format_image(self,frame):
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        return frame

    def add_background(self, frame, selfie_segmentation):
        frame = self.format_image(frame)
        frame.flags.writeable = False
        results = selfie_segmentation.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        if self.background_image is None:
            self.background_image = np.zeros(frame.shape, dtype=np.uint8)
            self.background_image[:] = self.resolution
        output_image = np.where(condition, frame, self.background_image)
        return output_image
        
    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
            while cap.isOpened():
                ret, frame = cap.read()
                modified_frame = self.add_background(frame, selfie_segmentation)
                cv2.imshow("Segmented image", modified_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__== "__main__":

    segmenter = MediaPipeSegmentation()
    segmenter.get_camera_resolution()
    segmenter.set_background_image("../files/wood.jpg")
    segmenter.capture_webcam()