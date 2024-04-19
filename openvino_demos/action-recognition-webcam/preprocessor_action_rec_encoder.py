import cv2
import numpy as np

class ActionRecEncoderPreprocessor:
    def __init__(self, size=224):
        """OCR text recognition preprocessor"""
        self.size = size

    def _center_crop(self, preprocessed):
        """
        Center crop squared the original frame to standardize the input image to the encoder model

        :param frame: input frame
        :returns: center-crop-squared frame
        """
        img_h, img_w, _ = preprocessed.shape
        min_dim = min(img_h, img_w)
        start_x = int((img_w - min_dim) / 2.0)
        start_y = int((img_h - min_dim) / 2.0)
        roi = [start_y, (start_y + min_dim), start_x, (start_x + min_dim)]
        return preprocessed[start_y : (start_y + min_dim), start_x : (start_x + min_dim), ...], roi


    def _adaptive_resize(self, frame):
        """
        The frame going to be resized to have a height of size or a width of size

        :param frame: input frame
        :param size: input size to encoder model
        :returns: resized frame, np.array type
        """
        h, w, _ = frame.shape
        scale = self.size / min(h, w)
        w_scaled, h_scaled = int(w * scale), int(h * scale)
        if w_scaled == w and h_scaled == h:
            return frame
        return cv2.resize(frame, (w_scaled, h_scaled))
    
    def preprocess_frame_for_encoder(self, frame):
        # Adaptive resize
        preprocessed = self._adaptive_resize(frame)
        # Center crop
        (preprocessed, roi) = self._center_crop(preprocessed)
        # Transpose frame HWC -> CHW
        preprocessed = preprocessed.transpose((2, 0, 1))[None,]  # HWC -> CHW
        preprocessed = preprocessed.astype(np.float32)
        return preprocessed
            