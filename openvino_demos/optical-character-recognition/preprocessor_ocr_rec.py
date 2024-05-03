import cv2
import numpy as np

class OCRTextRecPreprocessor:
    def __init__(self, rec_H=32, rec_W=100):
        """OCR text recognition preprocessor"""
        self.rec_H = rec_H
        self.rec_W = rec_W

    def initialize(self, text_det_res, image_backend="opencv"):
        self.text_det_res = text_det_res
        self.image_backend = image_backend
        self.grayscale_image = self.load_image_as_grayscale(self.text_det_res.image)
        return self.preprocess_crop()

    def load_image_as_grayscale(self, input_image):
        "Reads an image file and outputs a grayscale image"

        if self.image_backend == "opencv":
            grayscale_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

        elif self.image_backend == "pil":
            grayscale_image = input_image.convert('L')
        
        return grayscale_image
    
    def run_preprocesing_on_crop(self, crop):
        "Resizes the crop, reshapes it, and converts it to float32"

        if self.image_backend == "opencv":
            resized_crop = cv2.resize(crop, (self.rec_W, self.rec_H))  

        elif self.image_backend == "pil":
            resized_crop = crop.resize((self.rec_W, self.rec_H))
            resized_crop = np.array(resized_crop)

        reshaped_crop = resized_crop.reshape((1,) * 2 + resized_crop.shape)
        float32_crop = reshaped_crop.astype(np.float32)
        return float32_crop
    
    def preprocess_crop(self):
        for index, det in enumerate(self.text_det_res.results):
            bbox = [int(coord) for coord in det["bbox"]]
            (x_min, y_min, x_max, y_max) = bbox

            "Crops the image and runs preprocessing on the crop"
            if self.image_backend == "opencv":
                crop = self.grayscale_image[y_min:y_max, x_min:x_max]

            elif self.image_backend == "pil":
                crop = self.grayscale_image.crop((x_min, y_min, x_max, y_max))

            yield (self.run_preprocesing_on_crop(crop),index)