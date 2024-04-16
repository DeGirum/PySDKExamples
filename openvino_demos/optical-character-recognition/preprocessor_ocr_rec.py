import cv2
import numpy as np

class OCRTextRecPreprocessor:
    def __init__(self, rec_H=32, rec_W=100):
        """OCR text recognition preprocessor"""
        self.rec_H = rec_H
        self.rec_W = rec_W
        self.text_det_res = None
        self.grayscale_image = None


    def initialize(self, text_det_res):
        if self.text_det_res is None and self.grayscale_image is None:
            self.text_det_res = text_det_res
            self.grayscale_image = self.load_image_as_grayscale(self.text_det_res.image)

        return self.preprocess_crop()

    def load_image_as_grayscale(self, input_image):
        "Reads an image file and outputs a grayscale image"
        grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    def run_preprocesing_on_crop(self, crop):
        "Resizes the crop, reshapes it, and converts it to float32"
        resized_crop = cv2.resize(crop, (self.rec_W, self.rec_H))  # swapped rec_H and rec_W
        reshaped_crop = resized_crop.reshape((1,) * 2 + resized_crop.shape)
        float32_crop = reshaped_crop.astype(np.float32)
        return float32_crop
    
    def preprocess_crop(self):
        for index, det in enumerate(self.text_det_res.results):
            bbox = [int(coord) for coord in det["bbox"]]
            (x_min, y_min, x_max, y_max) = bbox
            "Crops the image and runs preprocessing on the crop"
            crop = self.grayscale_image[y_min:y_max, x_min:x_max]
            yield (self.run_preprocesing_on_crop(crop),index)