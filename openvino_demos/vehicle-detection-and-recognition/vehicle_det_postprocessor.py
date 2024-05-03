from degirum import postprocessor
import numpy as np
import importlib
import importlib.util

class VehicleDetPostprocessor(postprocessor.DetectionResults):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    
        boxes = self._inference_results[0]["data"]
        # Delete the dim of 0, 1.
        boxes = np.squeeze(boxes, (0, 1))
        # Remove zero only boxes.
        boxes = boxes[~np.all(boxes == 0, axis=1)]
        self._inference_results = self._post_processing_bbox(boxes)
        
    def _post_processing_bbox(self, boxes, threshold=0.6):
        """
        Use boxes from detection model to find the absolute car position
        
        :param: bgr_image: raw image
        :param: resized_image: resized image
        :param: boxes: detection model returns rectangle position
        :param: threshold: confidence threshold
        :returns: car_position: car's absolute position
        """
        # Fetch image shapes to calculate ratio
        new_inference_results = []
        if (
            isinstance(self.image, np.ndarray)
            and len(self.image.shape) == 3
            and importlib.util.find_spec("cv2")
        ):
            self._backend = importlib.import_module("cv2")
            self._backend_name = 'cv2'
            (real_y, real_x) = self._input_image.shape[:2]

        elif importlib.util.find_spec("PIL"):
            self._backend = importlib.import_module("PIL")
            if self._backend and isinstance(self.image, self._backend.Image.Image):
                self._backend_name  = 'pil'
                (real_x, real_y) = self._input_image.size[:2]

        resized_y, resized_x = self._model_params.InputH[0], self._model_params.InputW[0]
        ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

        # Find the boxes ratio
        boxes = boxes[:, 2:]
        # Iterate through non-zero boxes
        for box in boxes:
            # Pick confidence factor from last place in array
            conf = box[0]
            if conf > threshold:
                # Convert float to int and multiply corner position of each box by x and y ratio
                # In case that bounding box is found at the top of the image, 
                # upper box  bar should be positioned a little bit lower to make it visible on image 
                (x_min, y_min, x_max, y_max) = [
                    int(max(corner_position * ratio_y * resized_y, 10)) if idx % 2 
                    else int(corner_position * ratio_x * resized_x)
                    for idx, corner_position in enumerate(box[1:])
                ]
                
                box = [x_min, y_min, x_max, y_max]
                result = {"bbox":box, "score":conf, "category_id":0, "label":"vehicle"}
                new_inference_results.append(result)
            
        return new_inference_results