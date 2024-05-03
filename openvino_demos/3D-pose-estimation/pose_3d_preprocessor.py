import cv2
import numpy as np

class Pose3DPreprocessor:
    def __init__(self, inp_H=256, inp_W=448):
        """OCR text detection preprocessor"""
        super().__init__()
        self.inp_H = inp_H
        self.inp_W = inp_W
        self.stride = 8
    def load_frame(self, frame):
        "Reads an image file and outputs a numpy array"
        resized_frame = cv2.resize(frame, dsize=(self.inp_W, self.inp_H))
        scaled_frame = resized_frame[
        0 : resized_frame.shape[0] - (resized_frame.shape[0] % self.stride),
        0 : resized_frame.shape[1] - (resized_frame.shape[1] % self.stride),
        ]
        scaled_frame = np.transpose(scaled_frame, (2, 0, 1))[None,]
        scaled_frame = scaled_frame.astype('float32')
        return resized_frame, scaled_frame

