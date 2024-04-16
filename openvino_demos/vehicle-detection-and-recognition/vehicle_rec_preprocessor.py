import cv2
import numpy as np

class VehicleRecPreprocessor:
    def __init__(self, rec_H=72, rec_W=72):
        """OCR text recognition preprocessor"""
        self.rec_H = rec_H
        self.rec_W = rec_W
        self.vehicle_det_res = None


    def initialize(self, vehicle_det_res):
        if self.vehicle_det_res is None:
            self.vehicle_det_res = vehicle_det_res
        return self.preprocess_crop()
    
    def run_preprocesing_on_crop(self, crop):
        "Resizes the crop, expands the dim and converts it to float32"
        resized_crop = cv2.resize(crop, (self.rec_W, self.rec_H))
        resized_crop = np.expand_dims(resized_crop.transpose(2, 0, 1), 0)
        float32_crop = resized_crop.astype(np.float32)
        return float32_crop
    
    def preprocess_crop(self):
        for index, det in enumerate(self.vehicle_det_res.results):
            bbox = [int(coord) for coord in det["bbox"]]
            "Crops the vehicle and runs preprocessing on the crop"
            crop = self.vehicle_det_res.image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            yield (self.run_preprocesing_on_crop(crop), index)