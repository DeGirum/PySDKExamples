import cv2
import numpy as np
import importlib
import importlib.util
class VehicleRecPreprocessor:
    def __init__(self, rec_H=72, rec_W=72):
        """OCR text recognition preprocessor"""
        self.rec_H = rec_H
        self.rec_W = rec_W
        self.vehicle_det_res = None


    def initialize(self, vehicle_det_res, image_backend ="opencv"):
        if self.vehicle_det_res is None:
            self.vehicle_det_res = vehicle_det_res

        self.image_backend = image_backend
        return self.preprocess_crop()
    
    def run_preprocesing_on_crop(self, crop):
        "Resizes the crop, expands the dim and converts it to float32"

        if self.image_backend == "opencv":
            self._backend = importlib.import_module("cv2")
            resized_crop = cv2.resize(crop, (self.rec_W, self.rec_H))

        elif self.image_backend == "pil":
            self._backend = importlib.import_module("PIL")
            resized_crop = crop.resize((self.rec_W, self.rec_H))
            resized_crop = np.array(resized_crop)
            resized_crop = resized_crop[...,::-1]

        resized_crop = np.expand_dims(resized_crop.transpose(2, 0, 1), 0)
        float32_crop = resized_crop.astype(np.float32)
        return float32_crop
    
    def preprocess_crop(self):
        for index, det in enumerate(self.vehicle_det_res.results):
            bbox = [int(coord) for coord in det["bbox"]]

            "Crops the vehicle and runs preprocessing on the crop"
            if self.image_backend == "opencv":
                crop = self.vehicle_det_res.image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            elif self.image_backend == "pil":
                crop = self.vehicle_det_res.image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            yield (self.run_preprocesing_on_crop(crop), index)